"""
SwiGLU Feed-Forward Network module for ChakrView.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from chakrview.brain.config import ModelConfig


class SwiGLU(nn.Module):
    """
    SwiGLU activation feed-forward sub-layer.
    
    Formula:
        SwiGLU(x) = (SiLU(x * W_gate) * (x * W_up)) * W_down
        
    Strictly bias-free, intermediate dimension = 512.
    """
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.d_model, config.hidden_dim, bias=config.use_bias)
        self.up_proj = nn.Linear(config.d_model, config.hidden_dim, bias=config.use_bias)
        self.down_proj = nn.Linear(config.hidden_dim, config.d_model, bias=config.use_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [B, T, d_model].
            
        Returns:
            Output tensor of shape [B, T, d_model].
        """
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)
