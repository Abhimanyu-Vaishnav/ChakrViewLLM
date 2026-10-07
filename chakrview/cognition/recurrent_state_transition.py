"""Step 322: Minimal Recurrent State-Transition Mechanism.

Implements an integrated GRU-like gated recurrent state transition:
Given neural representation h_t ∈ R^(d_model) and previous reasoning state s_t ∈ R^(d_state):
z_t = sigmoid(W_z_h(h_t) + W_z_s(s_t) + b_z)    # Update gate
r_t = sigmoid(W_r_h(h_t) + W_r_s(s_t) + b_r)    # Reset gate
n_t = tanh(W_n_h(h_t) + W_n_s(r_t * s_t) + b_n) # Candidate state
s_(t+1) = (1 - z_t) * s_t + z_t * n_t            # State update

State Injection / Transition:
h' = h_t + tanh(gamma) * Proj_out(s_(t+1))

Where:
- d_state = 64 (compact parameter bound: < 40,000 parameters)
- gamma initialized to 0.0 (exact no-op / identity at initialization)
- Fully differentiable (no discrete dictionaries or symbolic lookup)
- Canonical backbone remains frozen (Delta W = 0)
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


class GatedRecurrentStateTransition(nn.Module):
    """
    Compact GRU-style recurrent state transition operating inside neural computation.
    """

    def __init__(
        self,
        d_model: int = 192,
        d_state: int = 64,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        self.norm = nn.LayerNorm(d_model)

        # Update gate
        self.w_z_h = nn.Linear(d_model, d_state)
        self.w_z_s = nn.Linear(d_state, d_state, bias=False)

        # Reset gate
        self.w_r_h = nn.Linear(d_model, d_state)
        self.w_r_s = nn.Linear(d_state, d_state, bias=False)

        # Candidate state
        self.w_n_h = nn.Linear(d_model, d_state)
        self.w_n_s = nn.Linear(d_state, d_state, bias=False)

        # Projection back to residual stream
        self.proj_out = nn.Linear(d_state, d_model, bias=False)
        self.gamma = nn.Parameter(torch.zeros(1))  # Zero-initialized no-op

        # Initial state parameter
        self.s0 = nn.Parameter(torch.zeros(1, d_state))

    def get_initial_state(self, batch_size: int = 1) -> torch.Tensor:
        """Returns initial state s_0 [B, d_state]."""
        return self.s0.expand(batch_size, -1)

    def transition(
        self,
        h_t: torch.Tensor,
        s_t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Executes one recurrent step: (h_t, s_t) -> s_(t+1).
        h_t: [B, d_model]
        s_t: [B, d_state]
        Returns:
            s_next: [B, d_state]
            h_out: [B, d_model] updated neural representation
        """
        if h_t.ndim == 3:
            h_t = h_t.squeeze(1)

        x = self.norm(h_t)
        z_t = torch.sigmoid(self.w_z_h(x) + self.w_z_s(s_t))
        r_t = torch.sigmoid(self.w_r_h(x) + self.w_r_s(s_t))
        n_t = torch.tanh(self.w_n_h(x) + self.w_n_s(r_t * s_t))

        s_next = (1.0 - z_t) * s_t + z_t * n_t

        g = torch.tanh(self.gamma)
        h_out = h_t + g * self.proj_out(s_next)
        return s_next, h_out
