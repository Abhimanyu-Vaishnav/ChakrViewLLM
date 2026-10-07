"""Step 273: Dynamic Contextual Token Binding Module.

Implements neural dynamic token binding:
Given:
1. Retrieved contextual value representation [B, D] (from Wave 257-264 adapter / value routing)
2. Contextual candidate token hidden states [B, N, D] extracted from the prompt

Dynamic Token Binding Architecture:
    retrieved_value_rep [B, D]
              ↓
           W_query  (LayerNorm + Linear(D -> d_bind))
              ↓
       query_rep [B, d_bind]

    contextual_token_states [B, N, D]
              ↓
            W_key   (LayerNorm + Linear(D -> d_bind))
              ↓
       candidate_keys [B, N, d_bind]

Compatibility scores:
    score_i = (query_norm · key_i_norm) / temperature
    scores = [B, N]

Properties:
- Differentiable neural scoring
- Zero hard-coded token IDs, zero Python lookup
- Candidate masking for variable sequence lengths/candidate counts
- Deterministic module hashing
- Parameter counting
"""

from __future__ import annotations

import dataclasses
import hashlib
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from chakrview.brain.model import ChakrMicro
from chakrview.runtime.interactive import compute_model_hash, EXPECTED_WEIGHT_HASH


def compute_module_sha256(module: nn.Module) -> str:
    """Computes deterministic SHA-256 digest of named module parameters."""
    hasher = hashlib.sha256()
    with torch.no_grad():
        for name, param in sorted(module.named_parameters()):
            hasher.update(name.encode("utf-8"))
            hasher.update(param.detach().cpu().numpy().tobytes())
    return hasher.hexdigest()


class DynamicContextualTokenBinding(nn.Module):
    """Binds retrieved contextual value representation to episode candidate token states."""

    def __init__(
        self,
        d_model: int = 192,
        d_bind: int = 64,
        temperature: float = 0.25,
        learnable_scale: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_bind = d_bind
        self.temperature = temperature

        self.query_norm = nn.LayerNorm(d_model)
        self.query_proj = nn.Linear(d_model, d_bind, bias=False)

        self.key_norm = nn.LayerNorm(d_model)
        self.key_proj = nn.Linear(d_model, d_bind, bias=False)

        if learnable_scale:
            self.scale = nn.Parameter(torch.tensor([1.0 / temperature]))
        else:
            self.register_buffer("scale", torch.tensor([1.0 / temperature]))

    def forward(
        self,
        retrieved_value_rep: torch.Tensor,
        candidate_token_states: torch.Tensor,
        candidate_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        retrieved_value_rep: [B, D] or [B, 1, D]
        candidate_token_states: [B, N, D]
        candidate_mask: [B, N] boolean or float (1 for valid candidate, 0 for invalid/padding)

        Returns:
            compatibility_logits: [B, N]
            normalized_probs: [B, N]
        """
        if retrieved_value_rep.ndim == 3:
            retrieved_value_rep = retrieved_value_rep.squeeze(1)

        B, D = retrieved_value_rep.shape
        _, N, _ = candidate_token_states.shape

        # 1. Project query
        q_norm = self.query_norm(retrieved_value_rep)
        q = self.query_proj(q_norm)  # [B, d_bind]
        q = F.normalize(q, p=2, dim=-1)

        # 2. Project candidate keys
        k_norm = self.key_norm(candidate_token_states)
        k = self.key_proj(k_norm)  # [B, N, d_bind]
        k = F.normalize(k, p=2, dim=-1)

        # 3. Bilinear / dot-product compatibility: [B, 1, d_bind] x [B, d_bind, N] -> [B, N]
        scores = torch.bmm(k, q.unsqueeze(-1)).squeeze(-1)  # [B, N]
        scaled_scores = scores * self.scale

        if candidate_mask is not None:
            # Mask out invalid positions with large negative value
            if candidate_mask.dtype == torch.bool:
                scaled_scores = scaled_scores.masked_fill(~candidate_mask, -1e9)
            else:
                scaled_scores = scaled_scores.masked_fill(candidate_mask == 0, -1e9)

        probs = F.softmax(scaled_scores, dim=-1)
        return scaled_scores, probs


class ChakrMicroWithDynamicBinding(nn.Module):
    """Wraps ChakrMicro with representation adapter and dynamic token binding module."""

    def __init__(
        self,
        base_model: ChakrMicro,
        adapter: Optional[nn.Module] = None,
        rank: int = 16,
        d_bind: int = 64,
        temperature: float = 0.25,
    ):
        super().__init__()
        self.base_model = base_model
        d_model = base_model.config.d_model

        from chakrview.cognition.neural_representation_adapter import GatedResidualAdapter
        self.adapter = adapter or GatedResidualAdapter(d_model=d_model, rank=rank)
        self.binding = DynamicContextualTokenBinding(
            d_model=d_model,
            d_bind=d_bind,
            temperature=temperature,
        )

    @property
    def config(self):
        return self.base_model.config

    def forward_backbone(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Runs frozen backbone and representation adapter."""
        x = self.base_model.embedding(input_ids)
        for i, layer in enumerate(self.base_model.layers):
            x = layer(x, attention_mask=None, kv_cache=None, layer_idx=i)
        raw_hidden = self.base_model.final_norm(x)
        adapted_hidden = self.adapter(raw_hidden)
        return adapted_hidden, raw_hidden

    def compute_binding_scores(
        self,
        retrieved_value_rep: torch.Tensor,
        candidate_states: torch.Tensor,
        candidate_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes dynamic candidate token selection scores."""
        return self.binding(retrieved_value_rep, candidate_states, candidate_mask)
