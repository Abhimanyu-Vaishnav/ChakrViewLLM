"""Step 257: Neural Representation Adapter Architecture.

Implements compact, trainable representation adaptation layers that receive
frozen ChakrMicro hidden states and transform the representation manifold
to facilitate identity-invariant role matching:

    frozen ChakrMicro hidden states
               ↓
    trainable representation adapter
               ↓
    adapted contextual states

Candidates implemented:
A. CompactResidualAdapter:
   LayerNorm(d_model) -> Linear(d_model, rank) -> GELU -> Linear(rank, d_model) -> residual scaling

B. GatedResidualAdapter:
   LayerNorm(d_model) -> Linear(d_model, rank) -> GELU -> Linear(rank, d_model)
   gated by learned scalar / vector sigmoid gate:
   h_adapted = h + sigmoid(gate) * adapter(h)

STRICT INVARIANTS:
- Canonical ChakrMicro remains completely frozen (Delta W = 0)
- Low-rank bottleneck (e.g. rank r = 16 or 32)
- Zero external dependencies, pure PyTorch CPU
"""

from __future__ import annotations

import dataclasses
import hashlib
from typing import Dict, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from chakrview.brain.model import ChakrMicro


def compute_module_sha256(module: nn.Module) -> str:
    """Computes deterministic SHA-256 digest of named module parameters."""
    hasher = hashlib.sha256()
    with torch.no_grad():
        for name, param in sorted(module.named_parameters()):
            hasher.update(name.encode("utf-8"))
            hasher.update(param.detach().cpu().numpy().tobytes())
    return hasher.hexdigest()


class CompactResidualAdapter(nn.Module):
    """Candidate A: Minimal Bottleneck Residual Adapter."""

    def __init__(
        self,
        d_model: int = 192,
        rank: int = 32,
        scaling: float = 0.5,
    ):
        super().__init__()
        self.d_model = d_model
        self.rank = rank
        self.scaling = scaling

        self.norm = nn.LayerNorm(d_model)
        self.down_proj = nn.Linear(d_model, rank, bias=False)
        self.act = nn.GELU()
        self.up_proj = nn.Linear(rank, d_model, bias=False)

        # Initialize up_proj with small weights so initial delta is close to 0
        nn.init.zeros_(self.up_proj.weight)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        hidden_states: [B, T, d_model]
        returns: [B, T, d_model] adapted hidden states
        """
        h_norm = self.norm(hidden_states)
        h_bottleneck = self.down_proj(h_norm)
        h_act = self.act(h_bottleneck)
        delta = self.up_proj(h_act)
        return hidden_states + self.scaling * delta


class GatedResidualAdapter(nn.Module):
    """Candidate B: Bottleneck Adapter with Learned Sigmoid Gate."""

    def __init__(
        self,
        d_model: int = 192,
        rank: int = 32,
    ):
        super().__init__()
        self.d_model = d_model
        self.rank = rank

        self.norm = nn.LayerNorm(d_model)
        self.down_proj = nn.Linear(d_model, rank, bias=False)
        self.act = nn.GELU()
        self.up_proj = nn.Linear(rank, d_model, bias=False)

        # Learned gate scalar, initialized to negative so gate starts near 0.1
        self.gate = nn.Parameter(torch.tensor([-2.0]))

        nn.init.zeros_(self.up_proj.weight)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        hidden_states: [B, T, d_model]
        returns: [B, T, d_model] adapted hidden states
        """
        h_norm = self.norm(hidden_states)
        h_bottleneck = self.down_proj(h_norm)
        h_act = self.act(h_bottleneck)
        delta = self.up_proj(h_act)
        g = torch.sigmoid(self.gate)
        return hidden_states + g * delta


class ChakrMicroWithAdaptedRepresentations(nn.Module):
    """Wraps ChakrMicro with an isolated trainable representation adapter."""

    def __init__(
        self,
        base_model: ChakrMicro,
        adapter_type: str = "gated",  # "residual" or "gated"
        rank: int = 32,
    ):
        super().__init__()
        self.base_model = base_model
        self.adapter_type = adapter_type
        d_model = base_model.config.d_model

        if adapter_type == "residual":
            self.adapter = CompactResidualAdapter(d_model=d_model, rank=rank)
        elif adapter_type == "gated":
            self.adapter = GatedResidualAdapter(d_model=d_model, rank=rank)
        else:
            raise ValueError(f"Unknown adapter type: {adapter_type}")

    @property
    def config(self):
        return self.base_model.config

    def forward_adapted_backbone(
        self,
        input_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Runs frozen ChakrMicro backbone, passes hidden states through adapter,
        and projects to next-token logits via frozen lm_head.

        Returns:
            adapted_hidden: [B, T, d_model]
            raw_hidden: [B, T, d_model]
            logits: [B, T, vocab_size]
        """
        # 1. Frozen backbone pass
        x = self.base_model.embedding(input_ids)
        for i, layer in enumerate(self.base_model.layers):
            x = layer(x, attention_mask=None, kv_cache=None, layer_idx=i)
        raw_hidden = self.base_model.final_norm(x)

        # 2. Trainable representation adaptation
        adapted_hidden = self.adapter(raw_hidden)

        # 3. Logits from adapted representation
        logits = self.base_model.lm_head(adapted_hidden)

        return adapted_hidden, raw_hidden, logits

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        _, _, logits = self.forward_adapted_backbone(input_ids)
        return logits
