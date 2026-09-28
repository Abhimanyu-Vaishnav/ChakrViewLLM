"""
TransformerBlock module for ChakrView.
"""

from typing import Optional
import torch
import torch.nn as nn
from chakrview.brain.config import ModelConfig
from chakrview.brain.normalization import RMSNorm
from chakrview.brain.attention import MultiHeadAttention
from chakrview.brain.feedforward import SwiGLU


class TransformerBlock(nn.Module):
    """
    Pre-RMSNorm Causal Transformer Block.
    
    Formula:
        x = x + Attention(RMSNorm1(x))
        x = x + SwiGLU(RMSNorm2(x))
    """
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.norm_1 = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.attn = MultiHeadAttention(config)
        self.norm_2 = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.ffn = SwiGLU(config)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Any] = None,
        layer_idx: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Residual state tensor of shape [B, T, d_model].
            attention_mask: Optional mask tensor of shape [B, T].
            kv_cache: Optional KVCache instance.
            layer_idx: Optional index of this transformer block.
            
        Returns:
            Updated state tensor of shape [B, T, d_model].
        """
        # Pre-norm attention residual
        x = x + self.attn(
            self.norm_1(x),
            attention_mask=attention_mask,
            kv_cache=kv_cache,
            layer_idx=layer_idx,
        )
        # Pre-norm FFN residual
        x = x + self.ffn(self.norm_2(x))
        return x

