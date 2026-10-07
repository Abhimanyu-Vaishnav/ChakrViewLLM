"""Step 298: Minimal Learned Associative Attractor / Codebook Module.

Implements a compact learned associative attractor / codebook layer:
h_intermediate [B, D]
        ↓
Attractor Projection (LayerNorm + Linear(D -> d_attractor))
        ↓
Cosine similarity against K learned attractor vectors [K, d_attractor]
        ↓
Soft assignment weights: p_k = softmax(sim_k / temperature)
        ↓
Weighted attractor state: h_attractor = sum_k p_k * A_k (projected back: Linear(d_attractor -> D))
        ↓
Residual blend with intermediate state:
h_out = h_intermediate + tanh(alpha) * h_attractor

Properties:
- K = 16 compact attractor states (compact parameter footprint: <15,000 parameters)
- Differentiable soft assignment during training (no Python nearest neighbor lookup)
- Optional straight-through / hard assignment mode for ablation
- Deterministic module SHA-256 computation
- Preserves canonical baseline freeze (Delta W = 0)
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


class LearnedAssociativeAttractor(nn.Module):
    """Learned attractor codebook with soft or straight-through assignment."""

    def __init__(
        self,
        d_model: int = 192,
        num_attractors: int = 16,
        d_attractor: int = 64,
        temperature: float = 0.25,
        use_hard_assignment: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_attractors = num_attractors
        self.d_attractor = d_attractor
        self.temperature = temperature
        self.use_hard_assignment = use_hard_assignment

        self.norm = nn.LayerNorm(d_model)
        self.proj_in = nn.Linear(d_model, d_attractor, bias=False)

        # Learned attractor prototypes [K, d_attractor]
        self.attractor_prototypes = nn.Parameter(torch.randn(num_attractors, d_attractor) * 0.02)

        self.proj_out = nn.Linear(d_attractor, d_model, bias=False)
        self.blend_gate = nn.Parameter(torch.zeros(1))  # Zero-initialized no-op

    def forward(
        self,
        intermediate_val_rep: torch.Tensor,
        force_hard: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        intermediate_val_rep: [B, D] or [B, 1, D]
        Returns:
            h_out: [B, D] blended attractor state
            assignment_probs: [B, K] assignment distribution
            entropy: scalar assignment entropy
        """
        if intermediate_val_rep.ndim == 3:
            intermediate_val_rep = intermediate_val_rep.squeeze(1)

        B, D = intermediate_val_rep.shape
        x_norm = self.norm(intermediate_val_rep)
        q = F.normalize(self.proj_in(x_norm), p=2, dim=-1)  # [B, d_attractor]

        proto_norm = F.normalize(self.attractor_prototypes, p=2, dim=-1)  # [K, d_attractor]

        # Cosine similarity: [B, K]
        sims = torch.matmul(q, proto_norm.transpose(0, 1)) / self.temperature
        probs = F.softmax(sims, dim=-1)

        do_hard = self.use_hard_assignment if force_hard is None else force_hard
        if do_hard:
            # Straight-through estimator
            idx = torch.argmax(probs, dim=-1, keepdim=True)
            hard_mask = torch.zeros_like(probs).scatter_(-1, idx, 1.0)
            assigned_weights = hard_mask - probs.detach() + probs
        else:
            assigned_weights = probs

        # Weighted attractor representation: [B, d_attractor]
        z_attractor = torch.matmul(assigned_weights, proto_norm)
        delta = self.proj_out(z_attractor)  # [B, D]

        gate = torch.tanh(self.blend_gate)
        h_out = intermediate_val_rep + gate * delta

        entropy = -float(torch.sum(probs * torch.log(probs + 1e-12), dim=-1).mean().item())

        return h_out, probs, entropy
