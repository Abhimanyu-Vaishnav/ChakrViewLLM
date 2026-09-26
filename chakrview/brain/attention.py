"""
Simple Multi-Head Attention (MHA) module for ChakrView.
"""

import math
from typing import Optional
import torch
import torch.nn as nn
from chakrview.brain.config import ModelConfig
from chakrview.brain.rotary import RotaryEmbedding
from chakrview.brain.masking import CausalMask


class MultiHeadAttention(nn.Module):
    """
    Simple Multi-Head Attention (MHA) with bias-free projections,
    RoPE applied to Q and K, and strict causal masking.
    """
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        # Bias-free linear projections
        self.q_proj = nn.Linear(self.d_model, self.d_model, bias=config.use_bias)
        self.k_proj = nn.Linear(self.d_model, self.d_model, bias=config.use_bias)
        self.v_proj = nn.Linear(self.d_model, self.d_model, bias=config.use_bias)
        self.out_proj = nn.Linear(self.d_model, self.d_model, bias=config.use_bias)
        
        self.rotary = RotaryEmbedding(
            dim=self.head_dim,
            max_seq_len=config.max_seq_len,
            theta=config.rope_theta
        )
        self.causal_mask = CausalMask(max_seq_len=config.max_seq_len)
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [B, T, d_model].
            attention_mask: Optional mask tensor of shape [B, T] (1 for valid, 0 for pad)
                           or additive mask of shape [B, 1, 1, T] / [B, 1, T, T].
            
        Returns:
            Output tensor of shape [B, T, d_model].
        """
        B, T, C = x.shape
        
        # 1. Projections: [B, T, d_model]
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # 2. Reshape into heads: [B, H, T, d_head]
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        
        # 3. Apply RoPE to Q and K (V remains untouched)
        q = self.rotary(q, T)
        k = self.rotary(k, T)
        
        # 4. Scaled Dot-Product Attention: [B, H, T, T]
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # 5. Add causal mask and optional attention_mask
        mask = self.causal_mask(T)
        scores = scores + mask
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                pad_mask = (attention_mask == 0).unsqueeze(1).unsqueeze(2)  # [B, 1, 1, T]
                scores = scores.masked_fill(pad_mask, -1e9)
            else:
                scores = scores + attention_mask
        
        # 6. Softmax & Dropout
        probs = torch.softmax(scores, dim=-1)
        probs = self.dropout(probs)
        
        # 7. Weighted sum over values: [B, H, T, d_head]
        context = torch.matmul(probs, v)
        
        # 8. Recombine heads: [B, T, d_model]
        context = context.transpose(1, 2).contiguous().view(B, T, C)
        
        # 9. Output projection
        return self.out_proj(context)
