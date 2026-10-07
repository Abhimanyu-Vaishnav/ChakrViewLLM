"""Steps 330, 331, 332: Trainable Compositional Attention Subsets (Q-only, K-only, Q+K).

Implements compact low-rank parameter adaptations directly on ChakrMicro's actual Q and K projections:
For selected layer l (default Layer 3):
q_proj(x) = W_q_frozen(x) + (alpha_q / r) * B_q(A_q(x))
k_proj(x) = W_k_frozen(x) + (alpha_k / r) * B_k(A_k(x))

Where:
- A ∈ R^(r × d_model) initialized with N(0, 0.02)
- B ∈ R^(d_model × r) initialized with 0.0 (exact identity / no-op at initialization!)
- Rank r = 16 or 32
- Parameter budget:
  - Q-only: 2 * 192 * 16 = 6,144 trainable parameters (< 25,000)
  - K-only: 2 * 192 * 16 = 6,144 trainable parameters (< 25,000)
  - Q+K: 12,288 trainable parameters (< 50,000)
- Canonical baseline weights remain strictly frozen (Delta W = 0)
"""

from __future__ import annotations

import dataclasses
import hashlib
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from chakrview.brain.model import ChakrMicro
from chakrview.brain.config import ModelConfig
from chakrview.runtime.interactive import compute_model_hash, EXPECTED_WEIGHT_HASH


def compute_module_sha256(module: nn.Module) -> str:
    """Computes deterministic SHA-256 digest of named module parameters."""
    hasher = hashlib.sha256()
    with torch.no_grad():
        for name, param in sorted(module.named_parameters()):
            hasher.update(name.encode("utf-8"))
            hasher.update(param.detach().cpu().numpy().tobytes())
    return hasher.hexdigest()


class LowRankProjectionDelta(nn.Module):
    """Low-rank delta B @ A with zero-initialized B for exact identity at init."""

    def __init__(self, in_features: int = 192, out_features: int = 192, rank: int = 16):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.scale = 1.0 / rank

        self.down = nn.Linear(in_features, rank, bias=False)
        self.up = nn.Linear(rank, out_features, bias=False)

        # Zero initialize up projection for exact identity
        nn.init.zeros_(self.up.weight)
        nn.init.normal_(self.down.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.down(x)) * self.scale


class CompositionalAttentionSubsetModel(nn.Module):
    """
    ChakrMicro model wrapper that directly modifies actual attention Q and/or K computation
    in target layer(s) while keeping canonical baseline weights completely frozen.
    """

    def __init__(
        self,
        base_model: ChakrMicro,
        target_layer: int = 3,
        mode: str = "q_only",  # "q_only", "k_only", or "qk"
        rank: int = 16,
    ):
        super().__init__()
        self.base_model = base_model
        self.config = base_model.config
        self.d_model = base_model.config.d_model
        self.target_layer = target_layer
        self.mode = mode
        self.rank = rank

        # Freeze canonical base model
        for p in self.base_model.parameters():
            p.requires_grad = False

        self.q_delta = None
        self.k_delta = None

        if mode in ("q_only", "qk"):
            self.q_delta = LowRankProjectionDelta(self.d_model, self.d_model, rank=rank)

        if mode in ("k_only", "qk"):
            self.k_delta = LowRankProjectionDelta(self.d_model, self.d_model, rank=rank)

    def forward_hidden_states(
        self,
        input_ids: torch.Tensor,
        disable_adaptation: bool = False,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Executes layer-by-layer forward pass, injecting low-rank delta directly into
        actual attention Q/K projections at the target layer.
        """
        B, T = input_ids.shape
        x = self.base_model.embedding(input_ids)
        states = [x]

        for i, layer in enumerate(self.base_model.layers):
            norm_x = layer.norm_1(x)
            attn = layer.attn

            if i == self.target_layer and not disable_adaptation:
                # Actual modified attention execution
                q = attn.q_proj(norm_x)
                if self.q_delta is not None:
                    q = q + self.q_delta(norm_x)

                k = attn.k_proj(norm_x)
                if self.k_delta is not None:
                    k = k + self.k_delta(norm_x)

                v = attn.v_proj(norm_x)

                # Reshape heads: [B, H, T, d_head]
                q = q.view(B, T, attn.n_heads, attn.head_dim).transpose(1, 2)
                k = k.view(B, T, attn.n_heads, attn.head_dim).transpose(1, 2)
                v = v.view(B, T, attn.n_heads, attn.head_dim).transpose(1, 2)

                q_rot = attn.rotary(q, T)
                k_rot = attn.rotary(k, T)

                scores = torch.matmul(q_rot, k_rot.transpose(-2, -1)) * attn.scale
                mask = attn.causal_mask(T)
                scores = scores + mask
                probs = torch.softmax(scores, dim=-1)
                context = torch.matmul(probs, v)
                context = context.transpose(1, 2).contiguous().view(B, T, self.d_model)
                attn_out = attn.out_proj(context)

                # Pre-norm residual step
                x = x + attn_out
                x = x + layer.ffn(layer.norm_2(x))

            else:
                x = layer(x, layer_idx=i)

            states.append(x)

        final_h = self.base_model.final_norm(x)
        return final_h, states

    def forward(
        self,
        input_ids: torch.Tensor,
        disable_adaptation: bool = False,
    ) -> torch.Tensor:
        final_h, _ = self.forward_hidden_states(input_ids, disable_adaptation=disable_adaptation)
        return self.base_model.lm_head(final_h)
