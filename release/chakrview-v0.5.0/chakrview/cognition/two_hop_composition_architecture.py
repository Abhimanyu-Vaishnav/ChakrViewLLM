"""Step 283: Neural Two-Hop Composition Architecture.

Implements the minimal isolated candidate architecture:
Frozen ChakrMicro Baseline
+
Gated Representation Adapter (rank=16)
+
Compositional Bridge Projector (W_bridge: D -> D with residual gate)
+
Dynamic Contextual Token Binding Module

The Compositional Bridge Projector:
Allows the retrieved intermediate value representation h_v1 to be mapped into
the key-query matching space for the second hop:
    h_q2 = LayerNorm(h_v1) + alpha * W_bridge(GELU(LayerNorm(h_v1)))

Properties:
- Compact parameter footprint: W_bridge is Linear(192 -> 192) + LayerNorm = 37,249 params
- Differentiable end-to-end training
- Isolated from baseline weights (Delta W = 0)
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
from chakrview.cognition.neural_representation_adapter import (
    GatedResidualAdapter,
    compute_module_sha256,
)
from chakrview.cognition.dynamic_contextual_token_binding import (
    DynamicContextualTokenBinding,
)


class CompositionalBridgeProjector(nn.Module):
    """Bridges first-hop retrieved value representation into second-hop query space."""

    def __init__(self, d_model: int = 192):
        super().__init__()
        self.d_model = d_model
        self.norm = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.act = nn.GELU()
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, intermediate_val_rep: torch.Tensor) -> torch.Tensor:
        """
        intermediate_val_rep: [B, D] or [B, 1, D]
        returns: [B, D] second-hop query representation
        """
        if intermediate_val_rep.ndim == 3:
            intermediate_val_rep = intermediate_val_rep.squeeze(1)

        x = self.norm(intermediate_val_rep)
        delta = self.proj(self.act(x))
        alpha = torch.tanh(self.gate)
        return intermediate_val_rep + alpha * delta


class ChakrMicroCompositionalReasoningModel(nn.Module):
    """
    Compositional Candidate Architecture:
    Frozen ChakrMicro + Adapter + Compositional Bridge + Dynamic Token Binding.
    """

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

        self.adapter = adapter or GatedResidualAdapter(d_model=d_model, rank=rank)
        self.bridge = CompositionalBridgeProjector(d_model=d_model)
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

    def bridge_intermediate_state(self, intermediate_val_rep: torch.Tensor) -> torch.Tensor:
        """Maps intermediate value state into second-hop query space."""
        return self.bridge(intermediate_val_rep)

    def compute_binding_scores(
        self,
        retrieved_value_rep: torch.Tensor,
        candidate_states: torch.Tensor,
        candidate_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes dynamic candidate token selection scores."""
        return self.binding(retrieved_value_rep, candidate_states, candidate_mask)
