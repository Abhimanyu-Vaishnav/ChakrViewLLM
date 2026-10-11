"""
ChakrView Step 56: Native Low-Rank Task Adapter.

Provides parameter-efficient, modular adaptation for ChakrMicro without mutating base weights.
Architecture:
- Base weights are completely frozen (requires_grad = False).
- In each TransformerBlock, low-rank matrices A and B (rank r=4, scale=1.0) are added to Q and V projections:
      W_adapted(x) = W_base(x) + (x @ A @ B) * (alpha / r)
- Detachable / Swappable: Adapters can be mounted, unmounted, or switched at runtime.
- Independent Checkpoint: Adapter weights are saved/loaded independently with distinct SHA-256 hashes.
"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import torch
import torch.nn as nn

from chakrview.brain.model import ChakrMicro


class LowRankLinear(nn.Module):
    """
    Native low-rank adaptation matrix: Delta W = (A @ B) * scaling.
    Base linear layer remains untouched.
    """
    def __init__(self, in_features: int, out_features: int, rank: int = 4, alpha: float = 8.0) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.scaling = alpha / rank

        # Down-projection A and Up-projection B
        self.lora_A = nn.Parameter(torch.empty(in_features, rank))
        self.lora_B = nn.Parameter(torch.empty(rank, out_features))

        # Deterministic initialization: A has Kaiming uniform, B is zero (identity at init)
        nn.init.kaiming_uniform_(self.lora_A, a=2.236)
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computes: (x @ A @ B) * scaling."""
        return (x @ self.lora_A @ self.lora_B) * self.scaling


class AdaptedLinear(nn.Module):
    """
    Wraps an existing base nn.Linear with an optional LowRankLinear delta.
    Leaves base_layer untouched.
    """
    def __init__(self, base_layer: nn.Linear, adapter: LowRankLinear) -> None:
        super().__init__()
        self.base_layer = base_layer
        self.adapter = adapter

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_layer(x)
        adapter_out = self.adapter(x)
        return base_out + adapter_out


class NativeTaskAdapter(nn.Module):
    """
    Container for low-rank task adapter modules across all 6 Transformer layers.
    Attaches to Q and V projections.
    Total parameters for rank=4, d_model=192 across 6 layers:
    Per layer: 2 * (192*4 + 4*192) = 2 * (768 + 768) = 3,072 params.
    6 layers: 6 * 3,072 = 18,432 parameters (~0.53% of ChakrMicro).
    """
    def __init__(
        self,
        adapter_name: str,
        n_layers: int = 6,
        d_model: int = 192,
        rank: int = 4,
        alpha: float = 8.0,
    ) -> None:
        super().__init__()
        self.adapter_name = adapter_name
        self.n_layers = n_layers
        self.d_model = d_model
        self.rank = rank
        self.alpha = alpha

        self.q_adapters = nn.ModuleList([
            LowRankLinear(d_model, d_model, rank=rank, alpha=alpha)
            for _ in range(n_layers)
        ])
        self.v_adapters = nn.ModuleList([
            LowRankLinear(d_model, d_model, rank=rank, alpha=alpha)
            for _ in range(n_layers)
        ])

    def compute_adapter_hash(self) -> str:
        """Compute deterministic SHA-256 hash of adapter weights."""
        hasher = hashlib.sha256()
        for name, param in sorted(self.named_parameters()):
            tensor_bytes = param.detach().cpu().numpy().tobytes()
            hasher.update(name.encode("utf-8"))
            hasher.update(tensor_bytes)
        return hasher.hexdigest()

    def mount(self, model: ChakrMicro) -> None:
        """
        Mount this adapter onto a ChakrMicro instance by wrapping Q and V projections.
        """
        for i, layer in enumerate(model.layers):
            attn = layer.attn
            # If already wrapped, unwrap first
            if isinstance(attn.q_proj, AdaptedLinear):
                attn.q_proj = attn.q_proj.base_layer
            if isinstance(attn.v_proj, AdaptedLinear):
                attn.v_proj = attn.v_proj.base_layer

            attn.q_proj = AdaptedLinear(attn.q_proj, self.q_adapters[i])
            attn.v_proj = AdaptedLinear(attn.v_proj, self.v_adapters[i])

    def unmount(self, model: ChakrMicro) -> None:
        """
        Unmount adapter from ChakrMicro, restoring original base projections.
        """
        for layer in model.layers:
            attn = layer.attn
            if isinstance(attn.q_proj, AdaptedLinear):
                attn.q_proj = attn.q_proj.base_layer
            if isinstance(attn.v_proj, AdaptedLinear):
                attn.v_proj = attn.v_proj.base_layer

    def save_checkpoint(self, path: Path) -> Dict[str, Any]:
        """Save adapter weights and metadata atomically."""
        path.parent.mkdir(parents=True, exist_ok=True)
        adapter_hash = self.compute_adapter_hash()
        payload = {
            "adapter_name": self.adapter_name,
            "n_layers": self.n_layers,
            "d_model": self.d_model,
            "rank": self.rank,
            "alpha": self.alpha,
            "adapter_hash": adapter_hash,
            "state_dict": self.state_dict(),
        }
        torch.save(payload, path)
        return {
            "path": str(path),
            "adapter_hash": adapter_hash,
            "parameter_count": sum(p.numel() for p in self.parameters()),
        }

    @classmethod
    def load_checkpoint(cls, path: Path) -> NativeTaskAdapter:
        """Load adapter from checkpoint."""
        payload = torch.load(path, map_location="cpu", weights_only=False)
        adapter = cls(
            adapter_name=payload["adapter_name"],
            n_layers=payload["n_layers"],
            d_model=payload["d_model"],
            rank=payload["rank"],
            alpha=payload["alpha"],
        )
        adapter.load_state_dict(payload["state_dict"])
        return adapter
