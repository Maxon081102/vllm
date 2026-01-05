# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Qwen3 model with layer skipping via trained jump heads.

This model extends Qwen3ForCausalLM to support dynamic layer skipping
during inference based on trained jump head predictions.
"""

import glob
import os
from collections.abc import Iterable
from typing import Any

import torch
from torch import nn
from transformers import Qwen3Config

from vllm.attention.backends.abstract import AttentionType
from vllm.attention.layer import Attention
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import QKVParallelLinear, RowParallelLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.config import set_default_rope_theta

from .interfaces import SupportsLoRA, SupportsPP
from .qwen2 import Qwen2MLP as Qwen3MLP
from .qwen3 import Qwen3Attention, Qwen3DecoderLayer, Qwen3Model
from .utils import AutoWeightsLoader, PPMissingLayer, maybe_prefix

logger = init_logger(__name__)


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


class Qwen3DecoderLayerWithJump(nn.Module):
    """Qwen3 decoder layer with optional jump head for layer skipping."""

    def __init__(
        self,
        config: Qwen3Config,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        layer_idx: int = 0,
        num_layers: int = 0,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.num_layers = num_layers
        self.hidden_size = config.hidden_size
        set_default_rope_theta(config, default_theta=1000000)
        dual_chunk_attention_config = getattr(
            config, "dual_chunk_attention_config", None
        )

        if getattr(config, "is_causal", True):
            attn_type = AttentionType.DECODER
        else:
            attn_type = AttentionType.ENCODER_ONLY

        self.self_attn = Qwen3Attention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            max_position=config.max_position_embeddings,
            num_kv_heads=config.num_key_value_heads,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, "attention_bias", False),
            head_dim=getattr(config, "head_dim", None),
            cache_config=cache_config,
            quant_config=quant_config,
            rope_parameters=config.rope_parameters,
            prefix=f"{prefix}.self_attn",
            attn_type=attn_type,
            dual_chunk_attention_config=dual_chunk_attention_config,
        )
        self.mlp = Qwen3MLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        # Jump head (will be loaded from checkpoint if this layer is in jump_layers)
        self.jump_head: JumpHead | None = None

    def set_jump_head(self, jump_head: JumpHead):
        """Set the jump head for this layer."""
        self.jump_head = jump_head

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        jump_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """
        Forward pass with optional jump state tracking.

        Args:
            positions: Position tensor
            hidden_states: Hidden states from previous layer
            residual: Residual connection
            jump_state: Tensor tracking which layer each token should process next.
                       Shape: (batch_size * seq_len,)

        Returns:
            Tuple of (hidden_states, residual, jump_state)
        """
        # Determine which tokens are active at this layer
        if jump_state is not None:
            # Tokens where jump_state == layer_idx should be processed
            active_mask = (jump_state == self.layer_idx)
            num_active = active_mask.sum().item()

            if num_active == 0:
                # Skip this layer entirely - no tokens need it
                return hidden_states, residual if residual is not None else hidden_states, jump_state
        else:
            active_mask = None

        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states_attn = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )

        # Apply mask if we have active tokens
        if active_mask is not None:
            # Reshape for broadcasting
            active_mask_expanded = active_mask.unsqueeze(-1).to(hidden_states_attn.dtype)
            hidden_states_attn = hidden_states_attn * active_mask_expanded

        hidden_states = hidden_states_attn

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states_mlp = self.mlp(hidden_states)

        if active_mask is not None:
            hidden_states_mlp = hidden_states_mlp * active_mask_expanded

        hidden_states = hidden_states_mlp

        # Update jump state using jump head
        if self.jump_head is not None and jump_state is not None:
            # Get jump predictions from the head
            # Use the hidden states after residual connection
            full_hidden = hidden_states + residual
            jump_len = self.jump_head(full_hidden)
            jump_len = jump_len.view(-1)

            # Clamp to remaining layers
            remaining = self.num_layers - self.layer_idx - 1
            jump_len = torch.clamp(jump_len, max=remaining)

            # Update jump state for active tokens
            new_jump_target = self.layer_idx + jump_len
            jump_state = torch.where(active_mask, new_jump_target, jump_state)
        elif jump_state is not None:
            # No jump head, just advance by 1 for active tokens
            if active_mask is not None:
                jump_state = torch.where(
                    active_mask,
                    jump_state + 1,
                    jump_state
                )

        return hidden_states, residual, jump_state


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
    }
)
class Qwen3ModelWithJump(nn.Module):
    """Qwen3 model with layer skipping support."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.num_layers = config.num_hidden_layers

        from vllm.model_executor.layers.vocab_parallel_embedding import (
            VocabParallelEmbedding,
        )

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
        )

        # Create layers with jump support
        self.layers = nn.ModuleList([
            Qwen3DecoderLayerWithJump(
                config,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=f"{prefix}.layers.{i}",
                layer_idx=i,
                num_layers=config.num_hidden_layers,
            )
            for i in range(config.num_hidden_layers)
        ])

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Jump layer configuration
        self.jump_layers_enabled = False
        self.jump_layer_indices: list[int] = []

        # Load jump layers if configured
        jump_config = vllm_config.jump_layers_config
        if jump_config is not None and jump_config.is_enabled:
            self._load_jump_heads(
                jump_config.jump_layers_path,
                jump_config.jump_layer_indices,
                config.hidden_size,
            )

    def _load_jump_heads(
        self,
        checkpoint_dir: str,
        layer_indices: list[int],
        hidden_size: int,
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

        # Two modes of operation:
        # 1. If layer_indices matches stage indices in checkpoints, use stage_idx as layer_idx
        # 2. Otherwise, map layer_indices[i] -> stage_ckpts[i]
        
        # First, try to load by matching stage_idx to layer_idx
        for layer_idx in layer_indices:
            if layer_idx in stage_ckpts:
                # Found a checkpoint with stage_idx matching the requested layer_idx
                if layer_idx >= len(self.layers) - 1:
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

                    head = JumpHead(hidden_size, head_max_jump)
                    head.load_state_dict(head_state)

                    self.layers[layer_idx].set_jump_head(head)
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

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.embed_tokens(input_ids)

        residual = None

        # Initialize jump state
        batch_size = hidden_states.shape[0]
        seq_len = hidden_states.shape[1] if hidden_states.dim() == 3 else 1
        total_tokens = batch_size * seq_len

        if self.jump_layers_enabled:
            # All tokens start at layer 0
            jump_state = torch.zeros(
                total_tokens,
                device=hidden_states.device,
                dtype=torch.long
            )
        else:
            jump_state = None

        for decoder_layer in self.layers:
            hidden_states, residual, jump_state = decoder_layer(
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
                jump_state=jump_state,
            )

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Qwen3ForCausalLMWithJump(nn.Module, SupportsLoRA, SupportsPP):
    """Qwen3 for causal language modeling with layer skipping support."""

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

