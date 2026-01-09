# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Qwen3 model with layer skipping via trained jump heads.

This model extends Qwen3ForCausalLM to support dynamic layer skipping
during inference based on trained jump head predictions.

Note: Currently, jump head predictions are computed but actual layer skipping
is not implemented due to torch.compile constraints. All layers are executed.
Statistics collection is only available with --enforce-eager flag.
"""

import atexit
import glob
import os
import time
from collections.abc import Iterable
from typing import Any

import torch
from torch import nn
import torch.compiler

from vllm.config import VllmConfig
from vllm.distributed import get_pp_group
from vllm.logger import init_logger
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.sequence import IntermediateTensors

from .interfaces import SupportsLoRA, SupportsPP
from .qwen3 import Qwen3Model, Qwen3ForCausalLM
from .utils import AutoWeightsLoader, PPMissingLayer, maybe_prefix

logger = init_logger(__name__)


class JumpLayerStats:
    """Statistics collector for jump layer predictions and actual skips.
    
    Only works when torch.compile is disabled (--enforce-eager).
    Logs statistics periodically.
    """
    
    def __init__(self, log_interval: float = 10.0):
        self.log_interval = log_interval
        self._total_calls = 0
        self._total_tokens = 0
        self._total_layer_executions = 0
        self._total_layers_skipped = 0
        # Per-layer: layer_idx -> {jump_value -> count}
        self._jump_counts: dict[int, dict[int, int]] = {}
        # Actual skips per layer
        self._skips_per_layer: dict[int, int] = {}
        self._enabled = False
        self._last_log_time = time.time()
        self._registered_atexit = False
        self._num_layers = 0
        
    def enable(self, layer_indices: list[int]):
        """Enable statistics collection for given layers."""
        self._enabled = True
        for layer_idx in layer_indices:
            self._jump_counts[layer_idx] = {}
        if not self._registered_atexit:
            atexit.register(self._log_final_stats)
            self._registered_atexit = True
        logger.info("Jump layer statistics collection ENABLED (--enforce-eager mode)")
    
    def set_num_layers(self, num_layers: int):
        """Set the total number of layers for skip ratio calculation."""
        self._num_layers = num_layers
    
    def record_layer_skip(self, layer_idx: int, num_skipped_tokens: int, total_tokens: int):
        """Record actual layer skipping statistics."""
        if not self._enabled:
            return
        self._total_layers_skipped += num_skipped_tokens
        self._total_layer_executions += total_tokens
        self._skips_per_layer[layer_idx] = self._skips_per_layer.get(layer_idx, 0) + num_skipped_tokens
    
    def record(self, layer_idx: int, jump_predictions: torch.Tensor):
        """Record jump predictions for a layer."""
        if not self._enabled:
            return
        
        # Skip if we're inside torch.compile (shouldn't happen with enforce-eager)
        if torch.compiler.is_compiling():
            return
        
        try:
            predictions = jump_predictions.detach().cpu().tolist()
        except Exception:
            return
        
        self._total_calls += 1
        self._total_tokens += len(predictions)
        
        if layer_idx not in self._jump_counts:
            self._jump_counts[layer_idx] = {}
        
        for pred in predictions:
            pred_int = int(pred)
            self._jump_counts[layer_idx][pred_int] = (
                self._jump_counts[layer_idx].get(pred_int, 0) + 1
            )
        
        # Check if we should log
        current_time = time.time()
        if current_time - self._last_log_time >= self.log_interval:
            self._log_stats()
            self._last_log_time = current_time
    
    def _log_stats(self):
        """Log current statistics."""
        if self._total_calls == 0:
            return
        
        # Calculate actual skip ratio
        actual_skip_ratio = 0.0
        if self._total_layer_executions > 0:
            actual_skip_ratio = self._total_layers_skipped / self._total_layer_executions
        
        lines = [
            "",
            "=" * 60,
            "JUMP LAYER STATISTICS (ACTUAL SKIPPING ENABLED)",
            "=" * 60,
            f"Total forward calls: {self._total_calls}",
            f"Total tokens processed: {self._total_tokens:,}",
            f"Total layer executions: {self._total_layer_executions:,}",
            f"Total tokens skipped: {self._total_layers_skipped:,}",
            f"ACTUAL SKIP RATIO: {actual_skip_ratio*100:.2f}%",
            "",
            "Per-layer skip counts:",
        ]
        
        for layer_idx in sorted(self._skips_per_layer.keys()):
            skips = self._skips_per_layer[layer_idx]
            lines.append(f"  Layer {layer_idx}: {skips:,} token-layers skipped")
        
        lines.append("")
        lines.append("Jump head predictions:")
        
        for layer_idx in sorted(self._jump_counts.keys()):
            counts = self._jump_counts[layer_idx]
            if not counts:
                continue
            
            total = sum(counts.values())
            if total == 0:
                continue
            
            # Calculate weighted average
            weighted_sum = sum(jump * count for jump, count in counts.items())
            avg_jump = weighted_sum / total
            
            # Format distribution
            dist_parts = []
            for jump in sorted(counts.keys()):
                count = counts[jump]
                pct = 100 * count / total
                dist_parts.append(f"{jump}:{count}({pct:.1f}%)")
            
            lines.append(
                f"  Layer {layer_idx}: avg_jump={avg_jump:.2f}, "
                f"dist=[{', '.join(dist_parts)}]"
            )
        
        lines.append("=" * 60)
        logger.info("\n".join(lines))
    
    def _log_final_stats(self):
        """Log final statistics at program exit."""
        if self._total_calls > 0:
            logger.info("=== FINAL JUMP LAYER STATISTICS ===")
            self._log_stats()


# Global statistics instance
_jump_stats = JumpLayerStats(log_interval=10.0)


class JumpHead(nn.Module):
    """Small head that predicts jump length for layer skipping."""

    def __init__(self, hidden_size: int, max_jump: int):
        super().__init__()
        self.max_jump = max_jump
        self.linear = nn.Linear(hidden_size, max_jump)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Returns jump_len (B, S) in [1, max_jump]."""
        logits = self.linear(hidden_states)
        actions = logits.argmax(dim=-1)
        return actions + 1


class Qwen3ModelWithJump(Qwen3Model):
    """Qwen3 model with layer skipping support.
    
    Inherits from Qwen3Model to maintain weight loading compatibility.
    Jump heads are added as separate modules and do not interfere with
    standard weight loading.
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        
        config = vllm_config.model_config.hf_config
        self.num_layers = config.num_hidden_layers
        self.hidden_size = config.hidden_size
        
        # Jump layer configuration
        self.jump_layers_enabled = False
        self.jump_layer_indices: list[int] = []
        
        # Jump heads stored as regular dict (NOT nn.ModuleDict) to avoid
        # being registered as model parameters that vLLM expects from checkpoint.
        # We load these separately from our own checkpoints.
        self._jump_heads: dict[int, JumpHead] = {}
        
        # Load jump layers if configured
        jump_config = vllm_config.jump_layers_config
        if jump_config is not None and jump_config.is_enabled:
            self._load_jump_heads(
                jump_config.jump_layers_path,
                jump_config.jump_layer_indices,
            )

    def _load_jump_heads(
        self,
        checkpoint_dir: str,
        layer_indices: list[int],
    ):
        """Load trained jump heads from checkpoint directory."""
        pattern = os.path.join(checkpoint_dir, "stage*_step*.pt")
        all_ckpts = glob.glob(pattern)

        if not all_ckpts:
            logger.warning(
                "No checkpoints found in %s. Jump layers will be disabled.",
                checkpoint_dir
            )
            return

        # Group by stage and take latest for each
        stage_ckpts: dict[int, tuple[str, int]] = {}
        for ckpt_path in all_ckpts:
            basename = os.path.basename(ckpt_path)
            parts = basename.replace(".pt", "").split("_")
            try:
                stage_num = int(parts[0].replace("stage", ""))
                step_num = int(parts[1].replace("step", ""))

                if stage_num not in stage_ckpts or step_num > stage_ckpts[stage_num][1]:
                    stage_ckpts[stage_num] = (ckpt_path, step_num)
            except (IndexError, ValueError):
                continue

        logger.info(
            "Loading jump heads from %s for layers %s (found %d checkpoints)",
            checkpoint_dir, layer_indices, len(stage_ckpts)
        )

        # Load by matching stage_idx to layer_idx
        for layer_idx in layer_indices:
            if layer_idx in stage_ckpts:
                # Found a checkpoint with stage_idx matching the requested layer_idx
                if layer_idx >= self.num_layers - 1:
                    logger.warning(
                        "Layer index %d is out of range or is the last layer. Skipping.",
                        layer_idx
                    )
                    continue

                ckpt_path, _ = stage_ckpts[layer_idx]
                try:
                    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)

                    if "train_head_state" in ckpt:
                        head_state = ckpt["train_head_state"]
                    else:
                        head_state = ckpt

                    head_max_jump = head_state["linear.weight"].shape[0]

                    head = JumpHead(self.hidden_size, head_max_jump)
                    head.load_state_dict(head_state)
                    head.eval()  # Set to eval mode
                    
                    # Store in regular dict with int key
                    self._jump_heads[layer_idx] = head
                    self.jump_layer_indices.append(layer_idx)

                    logger.info(
                        "Loaded jump head for layer %d (max_jump=%d) from %s",
                        layer_idx, head_max_jump, os.path.basename(ckpt_path)
                    )
                except Exception as e:
                    logger.error(
                        "Failed to load checkpoint %s: %s",
                        ckpt_path, e
                    )
            else:
                logger.warning(
                    "No checkpoint found for layer %d (stage_%d_*.pt)",
                    layer_idx, layer_idx
                )

        self.jump_layers_enabled = len(self.jump_layer_indices) > 0
        logger.info(
            "Jump layers enabled: %s, active layers: %s",
            self.jump_layers_enabled, self.jump_layer_indices
        )
        
        # Verify jump heads work with dummy input and enable statistics
        if self.jump_layers_enabled:
            self._verify_jump_heads()
            # Enable statistics collection (only works with --enforce-eager)
            _jump_stats.enable(self.jump_layer_indices)
            _jump_stats.set_num_layers(self.num_layers)
    
    def _verify_jump_heads(self):
        """Test jump heads with dummy data to verify they're working."""
        logger.info("Verifying jump heads with dummy data...")
        for layer_idx, head in self._jump_heads.items():
            try:
                # Create dummy input (batch_size=10, hidden_size)
                dummy_input = torch.randn(10, self.hidden_size)
                with torch.no_grad():
                    predictions = head(dummy_input)
                # Calculate distribution
                pred_list = predictions.tolist()
                unique_preds = {}
                for p in pred_list:
                    unique_preds[p] = unique_preds.get(p, 0) + 1
                dist_str = ", ".join(f"{k}:{v}" for k, v in sorted(unique_preds.items()))
                avg_jump = sum(pred_list) / len(pred_list)
                logger.info(
                    "Jump head layer %d test: avg_jump=%.2f, max_jump=%d, distribution=[%s]",
                    layer_idx, avg_jump, head.max_jump, dist_str
                )
            except Exception as e:
                logger.error("Jump head layer %d verification failed: %s", layer_idx, e)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        """Forward pass with jump layer support.
        
        When jump heads are enabled, tokens can skip layers based on predictions.
        For tokens in skip mode, their hidden states are not updated by skipped layers.
        """
        
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_tokens(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        # Initialize skip tracking for jump layers
        # skip_until_layer[i] = the layer index that token i should resume at
        # If current_layer < skip_until_layer[i], token i skips this layer
        skip_until_layer: torch.Tensor | None = None
        if self.jump_layers_enabled:
            if hidden_states.dim() == 3:
                num_tokens = hidden_states.shape[0] * hidden_states.shape[1]
            else:
                num_tokens = hidden_states.shape[0]
            # Initialize: all tokens start at layer 0 (no skipping)
            skip_until_layer = torch.zeros(
                num_tokens, 
                device=hidden_states.device, 
                dtype=torch.long
            )

        # Iterate through layers
        for layer_idx, decoder_layer in enumerate(
            self.layers[self.start_layer:self.end_layer]
        ):
            actual_layer_idx = self.start_layer + layer_idx
            
            # Determine which tokens should execute this layer vs skip
            should_skip_layer = False
            skip_mask = None
            if skip_until_layer is not None:
                # Tokens with skip_until_layer > actual_layer_idx should skip
                skip_mask = skip_until_layer > actual_layer_idx
                should_skip_layer = skip_mask.any().item()
            
            if should_skip_layer and not torch.compiler.is_compiling():
                # ACTUAL LAYER SKIPPING (only in eager mode)
                # Save states for tokens that will skip
                orig_shape = hidden_states.shape
                if hidden_states.dim() == 3:
                    hidden_flat = hidden_states.view(-1, hidden_states.shape[-1])
                    residual_flat = residual.view(-1, residual.shape[-1]) if residual is not None else None
                else:
                    hidden_flat = hidden_states
                    residual_flat = residual
                
                # Count skipped tokens for statistics
                num_skipped = skip_mask.sum().item()
                total_tokens = skip_mask.numel()
                _jump_stats.record_layer_skip(actual_layer_idx, num_skipped, total_tokens)
                
                # Save hidden states for skipping tokens
                saved_hidden = hidden_flat[skip_mask].clone()
                saved_residual = residual_flat[skip_mask].clone() if residual_flat is not None else None
                
                # Execute layer for all tokens (needed for KV cache)
                hidden_states, residual = decoder_layer(
                    positions=positions,
                    hidden_states=hidden_states,
                    residual=residual,
                )
                
                # Restore saved states for skipping tokens
                if hidden_states.dim() == 3:
                    hidden_flat = hidden_states.view(-1, hidden_states.shape[-1])
                    residual_flat = residual.view(-1, residual.shape[-1]) if residual is not None else None
                else:
                    hidden_flat = hidden_states
                    residual_flat = residual
                
                hidden_flat[skip_mask] = saved_hidden
                if residual_flat is not None and saved_residual is not None:
                    residual_flat[skip_mask] = saved_residual
                
                # Reshape back if needed
                if len(orig_shape) == 3:
                    hidden_states = hidden_flat.view(orig_shape)
                    if residual is not None:
                        residual = residual_flat.view(orig_shape)
            else:
                # Normal execution (no skipping or compiled mode)
                hidden_states, residual = decoder_layer(
                    positions=positions,
                    hidden_states=hidden_states,
                    residual=residual,
                )
                
                # Record layer execution for statistics (all tokens executed)
                if skip_mask is not None and not torch.compiler.is_compiling():
                    total_tokens = skip_mask.numel()
                    _jump_stats.record_layer_skip(actual_layer_idx, 0, total_tokens)
            
            # Apply jump head if this layer has one
            if self.jump_layers_enabled and actual_layer_idx in self._jump_heads:
                jump_head = self._jump_heads[actual_layer_idx]
                
                # Move jump head to correct device on first use  
                if next(jump_head.parameters()).device != hidden_states.device:
                    self._jump_heads[actual_layer_idx] = jump_head.to(hidden_states.device)
                    jump_head = self._jump_heads[actual_layer_idx]
                
                # Compute full hidden state (hidden + residual)
                full_hidden = hidden_states + residual if residual is not None else hidden_states
                
                # Flatten for jump head
                if full_hidden.dim() == 3:
                    full_hidden_flat = full_hidden.view(-1, full_hidden.shape[-1])
                else:
                    full_hidden_flat = full_hidden
                
                # Get jump predictions
                with torch.no_grad():
                    jump_predictions = jump_head(full_hidden_flat)
                    
                    # Update skip_until_layer based on predictions
                    # jump=1 means execute next layer (no skip)
                    # jump=N means skip to layer (current + N)
                    active_mask = None
                    if skip_until_layer is not None:
                        # Only update for tokens that are currently active (not already skipping)
                        active_mask = skip_until_layer <= actual_layer_idx
                        new_skip_target = actual_layer_idx + jump_predictions
                        skip_until_layer = torch.where(
                            active_mask,
                            new_skip_target,
                            skip_until_layer
                        )
                    
                    # Record statistics only in eager mode (not compiled)
                    # Only record predictions for ACTIVE tokens (not skipping this layer)
                    if not torch.compiler.is_compiling():
                        if active_mask is not None and not active_mask.all():
                            # Filter to only active tokens
                            active_predictions = jump_predictions[active_mask]
                            _jump_stats.record(actual_layer_idx, active_predictions)
                        else:
                            _jump_stats.record(actual_layer_idx, jump_predictions)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({
                "hidden_states": hidden_states,
                "residual": residual
            })

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Qwen3ForCausalLMWithJump(nn.Module, SupportsLoRA, SupportsPP):
    """Qwen3 for causal language modeling with layer skipping support.
    
    Uses standard Qwen3 architecture for weight loading compatibility,
    with jump heads loaded separately from checkpoints.
    """

    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config

        self.config = config
        self.quant_config = quant_config
        
        # Use our custom model with jump support
        self.model = Qwen3ModelWithJump(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )

        if get_pp_group().is_last_rank:
            if config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(
                    config.vocab_size,
                    config.hidden_size,
                    quant_config=quant_config,
                    prefix=maybe_prefix(prefix, "lm_head"),
                )
        else:
            self.lm_head = PPMissingLayer()

        self.logits_processor = LogitsProcessor(config.vocab_size)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        hidden_states = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head."] if self.config.tie_word_embeddings else None),
        )
        return loader.load_weights(weights)
