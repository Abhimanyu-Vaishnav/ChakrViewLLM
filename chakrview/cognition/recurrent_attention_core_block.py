"""Step 354: Minimal Recurrent Attention Core Block.

Implements RecurrentAttentionCoreBlock:
A self-contained neural computation block integrating:
1. Attention Cycle 1:
   q1 = Q1(norm1(x))
   k1 = K1(norm1(x))
   v1 = V1(norm1(x))
   a1 = causal_attention(q1, k1, v1)
   r1 = value_transform(a1)

2. Gated Recurrent State Transition:
   s1 = RecurrentTransition(r1, s0)  (d_state <= 64)

3. Query Generation from State:
   q2 = QueryFromState(s1, norm1(x))

4. Attention Cycle 2:
   k2 = K2(norm1(x))
   v2 = V2(norm1(x))
   a2 = causal_attention(q2, k2, v2)

5. Output and Residual Reconnection:
   x' = x + out_proj_1(a1) + gamma_c2 * out_proj_2(a2)
   y  = x' + FFN(norm2(x'))

Features:
- Differentiable end-to-end
- Parameter budget: <= 250,000 parameters (preferred <= 180,000)
- Zero-drift initialization option (gamma_c2 = 0.0, state gates initialized cleanly)
- Provides diagnostic probe hooks for a1, r1, s1, q2, a2
"""

from __future__ import annotations

import dataclasses
import math
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from chakrview.brain.config import ModelConfig
from chakrview.brain.normalization import RMSNorm
from chakrview.brain.feedforward import SwiGLU
from chakrview.brain.rotary import RotaryEmbedding
from chakrview.brain.masking import CausalMask


@dataclasses.dataclass
class CoreBlockTrace:
    q1: torch.Tensor
    k1: torch.Tensor
    v1: torch.Tensor
    a1: torch.Tensor
    r1: torch.Tensor
    s1: torch.Tensor
    q2: torch.Tensor
    k2: torch.Tensor
    v2: torch.Tensor
    a2: torch.Tensor
    attn1_weights: Optional[torch.Tensor]
    attn2_weights: Optional[torch.Tensor]


class RecurrentAttentionCoreBlock(nn.Module):
    """
    Self-contained Recurrent Attention Core Block for multi-hop relational reasoning.
    """

    def __init__(
        self,
        config: ModelConfig,
        d_state: int = 48,
        use_shared_kv: bool = True,
        freeze_ffn: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.config = config
        self.d_model = config.d_model  # 192
        self.n_heads = config.n_heads  # 6
        self.head_dim = self.d_model // self.n_heads  # 32
        self.d_state = d_state
        self.use_shared_kv = use_shared_kv

        # Pre-Norms
        self.norm_1 = RMSNorm(self.d_model, eps=config.rms_norm_eps)
        self.norm_2 = RMSNorm(self.d_model, eps=config.rms_norm_eps)

        # Attention Cycle 1 Projections
        self.q1_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.k1_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.v1_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.out1_proj = nn.Linear(self.d_model, self.d_model, bias=False)

        # Value Transformation to State Space
        self.norm_val = nn.LayerNorm(self.d_model)
        self.val_to_state = nn.Linear(self.d_model, self.d_state, bias=False)

        # Gated Recurrent State Transition (GRU-style)
        self.w_z_h = nn.Linear(self.d_state, self.d_state)
        self.w_z_s = nn.Linear(self.d_state, self.d_state, bias=False)

        self.w_r_h = nn.Linear(self.d_state, self.d_state)
        self.w_r_s = nn.Linear(self.d_state, self.d_state, bias=False)

        self.w_n_h = nn.Linear(self.d_state, self.d_state)
        self.w_n_s = nn.Linear(self.d_state, self.d_state, bias=False)

        # Base learnable initial state s0
        self.s0 = nn.Parameter(torch.zeros(1, self.d_state))

        # Query Generator from State s1
        self.state_to_q2 = nn.Linear(self.d_state, self.d_model, bias=False)

        # Attention Cycle 2 Projections
        if not use_shared_kv:
            self.k2_proj = nn.Linear(self.d_model, self.d_model, bias=False)
            self.v2_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        else:
            self.k2_proj = None
            self.v2_proj = None

        self.out2_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.gamma_c2 = nn.Parameter(torch.zeros(1))  # Zero-initialized no-op at init

        # Feedforward (SwiGLU)
        self.ffn = SwiGLU(config)
        self.freeze_ffn = freeze_ffn
        if freeze_ffn:
            for p in self.ffn.parameters():
                p.requires_grad = False

        self.rotary = RotaryEmbedding(
            dim=self.head_dim,
            max_seq_len=config.max_seq_len,
            theta=config.rope_theta,
        )
        self.causal_mask = CausalMask(max_seq_len=config.max_seq_len)

        self._reset_parameters()

    @property
    def trainable_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _reset_parameters(self) -> None:
        """Initializes projections cleanly."""
        std = self.config.initializer_range
        nn.init.normal_(self.q1_proj.weight, mean=0.0, std=std)
        nn.init.normal_(self.k1_proj.weight, mean=0.0, std=std)
        nn.init.normal_(self.v1_proj.weight, mean=0.0, std=std)
        nn.init.normal_(self.out1_proj.weight, mean=0.0, std=std)

        nn.init.normal_(self.val_to_state.weight, mean=0.0, std=std)
        nn.init.normal_(self.state_to_q2.weight, mean=0.0, std=std)
        nn.init.normal_(self.out2_proj.weight, mean=0.0, std=std)

        if self.k2_proj is not None:
            nn.init.normal_(self.k2_proj.weight, mean=0.0, std=std)
            nn.init.normal_(self.v2_proj.weight, mean=0.0, std=std)

    @property
    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def _causal_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes multi-head causal attention matching canonical MultiHeadAttention.
        """
        B, T, _ = q.shape
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        q = self.rotary(q, T)
        k = self.rotary(k, T)

        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        mask = self.causal_mask(T)
        scores = scores + mask

        if attention_mask is not None:
            if attention_mask.dim() == 2:
                pad_mask = (attention_mask == 0).unsqueeze(1).unsqueeze(2)
                scores = scores.masked_fill(pad_mask, -1e9)
            else:
                scores = scores + attention_mask

        weights = torch.softmax(scores, dim=-1)
        weights = torch.nan_to_num(weights, nan=0.0)

        context = torch.matmul(weights, v)
        context = context.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return context, weights

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        query_pos: Optional[int] = None,
        prev_state: Optional[torch.Tensor] = None,
        override_s1: Optional[torch.Tensor] = None,
        override_q2: Optional[torch.Tensor] = None,
        disable_c1: bool = False,
        disable_c2: bool = False,
        bypass_state: bool = False,
        return_trace: bool = False,
    ) -> Tuple[torch.Tensor, Optional[CoreBlockTrace]]:
        """
        Args:
            x: Residual tensor [B, T, d_model]
            attention_mask: Optional [B, T]
            query_pos: Token position where relational state transition occurs (defaults to last token T-1)
            prev_state: Optional incoming recurrent state tensor [B, d_state] from previous block
            override_s1: Optional tensor to force state s1 (for causal intervention)
            override_q2: Optional tensor to force query q2
            disable_c1: If True, zero out Cycle 1 attention contribution
            disable_c2: If True, zero out Cycle 2 attention contribution
            bypass_state: If True, do not use state s1 in query generation
            return_trace: If True, return CoreBlockTrace
        """
        B, T, d_model = x.shape
        pos = query_pos if query_pos is not None else (T - 1)

        norm_x = self.norm_1(x)

        # 1. Cycle 1: Causal Attention
        q1 = self.q1_proj(norm_x)
        k1 = self.k1_proj(norm_x)
        v1 = self.v1_proj(norm_x)

        a1_ctx, w1 = self._causal_attention(q1, k1, v1, attention_mask=attention_mask)
        a1 = self.out1_proj(a1_ctx)

        if disable_c1:
            a1 = torch.zeros_like(a1)

        # 2. Extract Value Representation at Query Position and State Transition
        # Query token attends backwards to premise value tokens.
        # r1 is the retrieved contextual representation at query token pos.
        h_query = a1_ctx[:, pos : pos + 1, :]  # [B, 1, d_model]
        h_norm = self.norm_val(h_query).squeeze(1)  # [B, d_model]
        r1 = self.val_to_state(h_norm)  # [B, d_state]

        # Recurrent state transition (GRU step)
        s_prev = prev_state if prev_state is not None else self.s0.expand(B, -1)  # [B, d_state]
        z_t = torch.sigmoid(self.w_z_h(r1) + self.w_z_s(s_prev))
        r_t = torch.sigmoid(self.w_r_h(r1) + self.w_r_s(s_prev))
        n_t = torch.tanh(self.w_n_h(r1) + self.w_n_s(r_t * s_prev))
        s1 = (1.0 - z_t) * s_prev + z_t * n_t

        if override_s1 is not None:
            s1 = override_s1

        # 3. Query Generation from s1 for Cycle 2
        # Base query from q1 plus state-driven query delta
        q_base = q1
        if bypass_state:
            q2 = q_base
        else:
            q_state = self.state_to_q2(s1).unsqueeze(1)  # [B, 1, d_model]
            # Modulate query at position pos
            q2 = q_base.clone()
            q2[:, pos : pos + 1, :] = q2[:, pos : pos + 1, :] + q_state

        if override_q2 is not None:
            q2 = override_q2

        # 4. Cycle 2: Second Attention Cycle
        if self.use_shared_kv:
            k2 = k1
            v2 = v1
        else:
            k2 = self.k2_proj(norm_x)
            v2 = self.v2_proj(norm_x)

        a2_ctx, w2 = self._causal_attention(q2, k2, v2, attention_mask=attention_mask)
        a2 = self.out2_proj(a2_ctx)

        if disable_c2:
            a2 = torch.zeros_like(a2)

        # 5. Output and Residual Reconnection
        # Pre-norm residual for attention cycles
        x_post_attn = x + a1 + torch.tanh(self.gamma_c2) * a2

        # Pre-norm FFN residual
        y = x_post_attn + self.ffn(self.norm_2(x_post_attn))

        trace = None
        if return_trace:
            trace = CoreBlockTrace(
                q1=q1,
                k1=k1,
                v1=v1,
                a1=a1,
                r1=r1,
                s1=s1,
                q2=q2,
                k2=k2,
                v2=v2,
                a2=a2,
                attn1_weights=w1,
                attn2_weights=w2,
            )

        return y, trace
