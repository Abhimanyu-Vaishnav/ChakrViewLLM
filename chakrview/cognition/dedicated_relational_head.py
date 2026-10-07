"""Step 333: Dedicated Relational Attention Head Architecture.

Implements an integrated relational attention head that:
- Operates on the exact same hidden states x ∈ R^(B, T, d_model)
- Obeys RoPE and strict causal masking
- Uses learned relational Q_rel, K_rel, V_rel projections (d_head = 32 or 64)
- Produces actual attention probabilities over the premise sequence
- Projects back to the residual stream with zero-initialized blend gate:
  x = x + tanh(beta) * Out_rel(RelationalAttention(RMSNorm(x)))

Properties:
- Parameter budget: ~24,576 parameters (< 50,000 budget)
- Starts at exact identity / zero-op initialization (beta = 0.0)
- Fully differentiable (no Python dictionaries, no symbolic mappings)
- Preserves canonical baseline freeze (Delta W = 0)
"""

from __future__ import annotations

import dataclasses
import hashlib
import math
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from chakrview.brain.model import ChakrMicro
from chakrview.brain.config import ModelConfig
from chakrview.brain.normalization import RMSNorm
from chakrview.brain.rotary import RotaryEmbedding
from chakrview.brain.masking import CausalMask
from chakrview.runtime.interactive import compute_model_hash, EXPECTED_WEIGHT_HASH


class DedicatedRelationalAttentionHead(nn.Module):
    """
    Dedicated relational attention head operating at a target transformer block.
    """

    def __init__(
        self,
        d_model: int = 192,
        head_dim: int = 32,
        max_seq_len: int = 512,
        rope_theta: float = 10000.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.head_dim = head_dim
        self.scale = 1.0 / math.sqrt(head_dim)

        self.norm = RMSNorm(d_model)
        self.q_proj = nn.Linear(d_model, head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, head_dim, bias=False)
        self.out_proj = nn.Linear(head_dim, d_model, bias=False)

        self.rotary = RotaryEmbedding(dim=head_dim, max_seq_len=max_seq_len, theta=rope_theta)
        self.causal_mask = CausalMask(max_seq_len=max_seq_len)

        self.gate = nn.Parameter(torch.zeros(1))  # Zero-initialized no-op

    def forward(
        self,
        x: torch.Tensor,
        disable_head: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: [B, T, d_model]
        Returns:
            x_out: [B, T, d_model] updated residual representation
            attn_probs: [B, T, T] relational attention matrix
        """
        if disable_head:
            B, T, _ = x.shape
            dummy_probs = torch.eye(T, device=x.device).unsqueeze(0).expand(B, -1, -1)
            return x, dummy_probs

        B, T, C = x.shape
        x_norm = self.norm(x)

        q = self.q_proj(x_norm).unsqueeze(1)  # [B, 1, T, head_dim]
        k = self.k_proj(x_norm).unsqueeze(1)  # [B, 1, T, head_dim]
        v = self.v_proj(x_norm).unsqueeze(1)  # [B, 1, T, head_dim]

        q = self.rotary(q, T)
        k = self.rotary(k, T)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, 1, T, T]
        mask = self.causal_mask(T)
        scores = scores + mask
        probs = torch.softmax(scores, dim=-1)  # [B, 1, T, T]

        context = torch.matmul(probs, v).squeeze(1)  # [B, T, head_dim]
        delta = self.out_proj(context)  # [B, T, d_model]

        g = torch.tanh(self.gate)
        x_out = x + g * delta
        return x_out, probs.squeeze(1)


class ChakrMicroWithDedicatedRelationalHead(nn.Module):
    """
    Candidate model integrating DedicatedRelationalAttentionHead into target layer.
    """

    def __init__(
        self,
        base_model: ChakrMicro,
        target_layer: int = 3,
        head_dim: int = 32,
    ):
        super().__init__()
        self.base_model = base_model
        self.config = base_model.config
        self.d_model = base_model.config.d_model
        self.target_layer = target_layer

        for p in self.base_model.parameters():
            p.requires_grad = False

        self.rel_head = DedicatedRelationalAttentionHead(
            d_model=self.d_model,
            head_dim=head_dim,
            max_seq_len=self.config.max_seq_len,
            rope_theta=self.config.rope_theta,
        )

    def forward_hidden_states(
        self,
        input_ids: torch.Tensor,
        disable_head: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns final_h and relational attention matrix from target layer."""
        B, T = input_ids.shape
        x = self.base_model.embedding(input_ids)
        rel_attn = None

        for i, layer in enumerate(self.base_model.layers):
            x = layer(x, layer_idx=i)
            if i == self.target_layer:
                x, rel_attn = self.rel_head(x, disable_head=disable_head)

        final_h = self.base_model.final_norm(x)
        return final_h, rel_attn

    def forward(
        self,
        input_ids: torch.Tensor,
        disable_head: bool = False,
    ) -> torch.Tensor:
        final_h, _ = self.forward_hidden_states(input_ids, disable_head=disable_head)
        return self.base_model.lm_head(final_h)
