"""Step 265: Contextual Vocabulary Readout.

Implements a compact learned contextual vocabulary readout module that receives
a selected value contextual representation [B, d_model] and maps it to next-token
vocabulary logits over the tokenizer vocabulary:

    selected value representation [B, d_model]
                     ↓
              LayerNorm(d_model)
                     ↓
          Linear(d_model -> d_model)
                     ↓
                   GELU()
                     ↓
       Linear(d_model -> vocab_size)
                     ↓
               vocabulary logits

STRICT INVARIANTS:
- Canonical ChakrMicro remains completely frozen (Delta W = 0)
- Canonical embedding & tied lm_head remain completely frozen
- Zero hard-coded token-ID mappings, zero Python lookups
- Differentiable gradient-descent trainable
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


class ContextualVocabularyReadout(nn.Module):
    """Compact learned contextual vocabulary readout head."""

    def __init__(
        self,
        d_model: int = 192,
        vocab_size: int = 4096,
        hidden_dim: Optional[int] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        h_dim = hidden_dim or d_model

        self.norm = nn.LayerNorm(d_model)
        self.intermediate = nn.Linear(d_model, h_dim, bias=True)
        self.act = nn.GELU()
        self.emission = nn.Linear(h_dim, vocab_size, bias=False)

    def forward(self, value_representation: torch.Tensor) -> torch.Tensor:
        """
        value_representation: [B, d_model] or [B, 1, d_model]
        returns: [B, vocab_size] logits
        """
        if value_representation.ndim == 3:
            value_representation = value_representation.squeeze(1)

        x = self.norm(value_representation)
        x = self.act(self.intermediate(x))
        logits = self.emission(x)
        return logits


class ChakrMicroWithContextualReadout(nn.Module):
    """Wraps ChakrMicro with an adapter and ContextualVocabularyReadout."""

    def __init__(
        self,
        base_model: ChakrMicro,
        adapter: Optional[nn.Module] = None,
        rank: int = 16,
    ):
        super().__init__()
        self.base_model = base_model
        d_model = base_model.config.d_model
        vocab_size = base_model.config.vocab_size

        from chakrview.cognition.neural_representation_adapter import GatedResidualAdapter
        self.adapter = adapter or GatedResidualAdapter(d_model=d_model, rank=rank)
        self.readout = ContextualVocabularyReadout(d_model=d_model, vocab_size=vocab_size)

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

    def forward_from_value_rep(self, value_rep: torch.Tensor) -> torch.Tensor:
        """Emits vocabulary distribution directly from selected value representation."""
        return self.readout(value_rep)
