# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configuration for jump layers (layer skipping via trained jump heads)."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class JumpLayersConfig:
    """Configuration for jump layers (layer skipping via trained jump heads).
    
    Jump layers allow dynamic layer skipping during inference based on 
    trained jump head predictions. Each jump head predicts how many layers
    to skip for each token.
    
    Attributes:
        jump_layers_path: Path to directory containing trained jump head 
            checkpoints (e.g., stage0_step100.pt, stage5_step200.pt).
        jump_layers: Comma-separated list of layer indices where jump heads
            are applied (e.g., "0,5,10,15,20").
        enable_logging: Whether to log layer skipping statistics.
        log_file: Path to log file for jump statistics (JSON lines format).
    """
    
    jump_layers_path: Optional[str] = None
    jump_layers: Optional[str] = None
    enable_logging: bool = False
    log_file: Optional[str] = None
    
    def __post_init__(self):
        # Parse jump_layers string into list of integers
        self._jump_layer_indices: list[int] = []
        if self.jump_layers:
            try:
                self._jump_layer_indices = [
                    int(x.strip()) for x in self.jump_layers.split(",")
                    if x.strip()
                ]
            except ValueError as e:
                raise ValueError(
                    f"Invalid jump_layers format: {self.jump_layers}. "
                    "Expected comma-separated integers (e.g., '0,5,10')."
                ) from e
    
    @property
    def jump_layer_indices(self) -> list[int]:
        """Get list of layer indices where jump heads are applied."""
        return self._jump_layer_indices
    
    @property
    def is_enabled(self) -> bool:
        """Check if jump layers are enabled."""
        return bool(self.jump_layers_path and self._jump_layer_indices)
    
    def validate(self, num_layers: int):
        """Validate configuration against model's number of layers.
        
        Args:
            num_layers: Total number of layers in the model.
            
        Raises:
            ValueError: If any jump layer index is out of range.
        """
        if not self.is_enabled:
            return
            
        for idx in self._jump_layer_indices:
            if idx < 0 or idx >= num_layers - 1:
                raise ValueError(
                    f"Jump layer index {idx} is out of range. "
                    f"Valid range is [0, {num_layers - 2}] "
                    f"(model has {num_layers} layers, last layer cannot have jump head)."
                )

