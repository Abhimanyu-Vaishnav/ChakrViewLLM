"""Step 306: Minimal Internal Trainable Pathway Architecture.

Implements an isolated, parameter-bounded trainable bottleneck adapter that operates
INSIDE the transformer layer computation, while keeping the canonical ChakrMicro backbone frozen.

Architecture:
Inside Transformer Block:
x = x + Attention(...)
x = x + SwiGLU(...)
x = x + tanh(alpha) * CompositionalPathway(LayerNorm(x))

Where:
CompositionalPathway(h) = DownLinear(192 -> bottleneck_dim) -> GELU -> UpLinear(bottleneck_dim -> 192)
Where:
- alpha is initialized to 0.0 (exact no-op / identity at initialization)
- bottleneck_dim = 32 or 48 (< 20,000 parameters per pathway)
- Canonical weights remain strictly frozen (Delta W = 0)
- Preserves token length and sequence context
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


class InternalBottleneckAdapter(nn.Module):
    """Compact bottleneck adapter with zero-initialized tanh blend gate."""

    def __init__(self, d_model: int = 192, bottleneck_dim: int = 32):
        super().__init__()
        self.d_model = d_model
        self.bottleneck_dim = bottleneck_dim

        self.norm = nn.LayerNorm(d_model)
        self.down = nn.Linear(d_model, bottleneck_dim, bias=False)
        self.act = nn.GELU()
        self.up = nn.Linear(bottleneck_dim, d_model, bias=False)
        self.gate = nn.Parameter(torch.zeros(1))  # Exact no-op initialization

    def forward(self, x: torch.Tensor, disabled: bool = False) -> torch.Tensor:
        if disabled:
            return x
        delta = self.up(self.act(self.down(self.norm(x))))
        alpha = torch.tanh(self.gate)
        return x + alpha * delta


class ChakrMicroWithInternalPathway(nn.Module):
    """
    Candidate model that routes computation through the frozen ChakrMicro backbone,
    with selected internal trainable bottleneck pathways inserted at specified layer indices.
    """

    def __init__(
        self,
        base_model: ChakrMicro,
        target_layers: Optional[List[int]] = None,
        bottleneck_dim: int = 32,
        insertion_side: str = "residual",  # "residual", "attention", or "ffn"
    ):
        super().__init__()
        self.base_model = base_model
        self.config = base_model.config

        # Strictly freeze canonical base model
        for p in self.base_model.parameters():
            p.requires_grad = False

        if target_layers is None:
            # Default to mid & late transformer blocks (e.g. layers 3, 4, 5)
            target_layers = [3, 4, 5]
        self.target_layers = sorted(target_layers)
        self.insertion_side = insertion_side
        self.bottleneck_dim = bottleneck_dim

        # Create one InternalBottleneckAdapter per target layer
        self.adapters = nn.ModuleDict({
            f"layer_{l}": InternalBottleneckAdapter(d_model=self.config.d_model, bottleneck_dim=bottleneck_dim)
            for l in self.target_layers
        })

    def forward_hidden_states(
        self,
        input_ids: torch.Tensor,
        disable_adapters: bool = False,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Executes causal forward pass layer-by-layer, inserting internal adapters.
        Returns final hidden state and list of intermediate hidden states per layer.
        """
        x = self.base_model.embedding(input_ids)
        states = [x]

        for i, layer in enumerate(self.base_model.layers):
            if self.insertion_side == "attention" and i in self.target_layers:
                # Insert right after attention residual
                attn_out = layer.attn(layer.norm_1(x), layer_idx=i)
                x = x + attn_out
                adapter = self.adapters[f"layer_{i}"]
                x = adapter(x, disabled=disable_adapters)
                x = x + layer.ffn(layer.norm_2(x))

            elif self.insertion_side == "ffn" and i in self.target_layers:
                # Insert right after FFN residual
                x = x + layer.attn(layer.norm_1(x), layer_idx=i)
                ffn_out = layer.ffn(layer.norm_2(x))
                x = x + ffn_out
                adapter = self.adapters[f"layer_{i}"]
                x = adapter(x, disabled=disable_adapters)

            else:
                # Default "residual": standard layer pass, then adapter on full block residual
                x = layer(x, layer_idx=i)
                if i in self.target_layers:
                    adapter = self.adapters[f"layer_{i}"]
                    x = adapter(x, disabled=disable_adapters)

            states.append(x)

        final_h = self.base_model.final_norm(x)
        return final_h, states

    def forward(
        self,
        input_ids: torch.Tensor,
        disable_adapters: bool = False,
    ) -> torch.Tensor:
        """Forward pass to vocabulary logits."""
        final_h, _ = self.forward_hidden_states(input_ids, disable_adapters=disable_adapters)
        logits = self.base_model.lm_head(final_h)
        return logits
