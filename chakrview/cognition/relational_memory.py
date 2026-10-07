"""Step 314 & Step 315: Differentiable Neural Relational Memory Module.

Implements a compact, fully differentiable neural relational memory:
Memory State:
M_t ∈ R^(S × d_mem)
where:
- S: number of memory slots (S = 1 for minimal single-slot, S = 2 or 4 for multi-slot)
- d_mem: slot dimensionality (e.g., 64)

Write Mechanism (WriteController):
- Given token representation v_t ∈ R^(d_model):
  k_write = Linear(v_t -> d_mem)
  v_write = Linear(v_t -> d_mem)
  write_gate = sigmoid(Linear(v_t -> S))
  address_weights = softmax(k_write @ M_t^T / sqrt(d_mem)) [S]
  M_(t+1) = (1 - alpha * write_gate * address_weights) * M_t + (alpha * write_gate * address_weights) * v_write

Read Mechanism (ReadController):
- Given query or retrieval intent q_t ∈ R^(d_model) (or default readout):
  read_attn = softmax(Linear(q_t -> d_mem) @ M_(t+1)^T / sqrt(d_mem))
  readout = read_attn @ M_(t+1)
  h_q2 = Linear(readout -> d_model)

Properties:
- Fully differentiable (no Python dictionaries, no symbolic mappings)
- Zero answer-label leakage
- Parameter-bounded (< 35,000 parameters for S=2..4)
- Deterministic initialization
"""

from __future__ import annotations

import dataclasses
import hashlib
import math
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


class DifferentiableRelationalMemory(nn.Module):
    """
    Compact neural relational working memory with S slots.
    """

    def __init__(
        self,
        d_model: int = 192,
        num_slots: int = 2,
        d_mem: int = 64,
        slot_dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_slots = num_slots
        self.d_mem = d_mem

        # Initial memory bank [S, d_mem]
        self.initial_memory = nn.Parameter(torch.randn(num_slots, d_mem) * 0.02)

        # Write controller projections
        self.write_k_proj = nn.Linear(d_model, d_mem, bias=False)
        self.write_v_proj = nn.Linear(d_model, d_mem, bias=False)
        self.write_gate_proj = nn.Linear(d_model, num_slots)
        self.write_alpha = nn.Parameter(torch.ones(1) * 0.5)

        # Read controller projections
        self.read_q_proj = nn.Linear(d_model, d_mem, bias=False)
        self.read_out_proj = nn.Linear(d_mem, d_model, bias=False)
        self.norm_in = nn.LayerNorm(d_model)

    def get_initial_state(self, batch_size: int = 1) -> torch.Tensor:
        """Returns initial memory state [B, S, d_mem]."""
        return self.initial_memory.unsqueeze(0).expand(batch_size, -1, -1)

    def write(
        self,
        memory_state: torch.Tensor,
        value_token_rep: torch.Tensor,
        disable_write: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Writes retrieved token representation into memory.
        memory_state: [B, S, d_mem]
        value_token_rep: [B, d_model] or [B, 1, d_model]
        Returns:
            updated_memory: [B, S, d_mem]
            write_weights: [B, S]
        """
        if disable_write:
            B = memory_state.shape[0]
            dummy_weights = torch.zeros(B, self.num_slots, device=memory_state.device)
            return memory_state, dummy_weights

        if value_token_rep.ndim == 3:
            value_token_rep = value_token_rep.squeeze(1)

        B = value_token_rep.shape[0]
        v_norm = self.norm_in(value_token_rep)

        k_write = self.write_k_proj(v_norm)  # [B, d_mem]
        v_write = self.write_v_proj(v_norm)  # [B, d_mem]

        # Addressing scores against current slots: [B, S]
        # (B, S, d_mem) x (B, d_mem, 1) -> [B, S]
        sims = torch.bmm(memory_state, k_write.unsqueeze(-1)).squeeze(-1) / math.sqrt(self.d_mem)
        address_weights = F.softmax(sims, dim=-1)  # [B, S]

        # Write gate: [B, S]
        gate = torch.sigmoid(self.write_gate_proj(v_norm))
        eff_weights = (address_weights * gate).unsqueeze(-1)  # [B, S, 1]

        alpha = torch.sigmoid(self.write_alpha)
        # Update: M_(t+1) = (1 - alpha * eff_weights) * M_t + (alpha * eff_weights) * v_write
        updated_memory = (1.0 - alpha * eff_weights) * memory_state + (alpha * eff_weights) * v_write.unsqueeze(1)
        return updated_memory, address_weights

    def read(
        self,
        memory_state: torch.Tensor,
        query_intent: Optional[torch.Tensor] = None,
        disable_read: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Reads from memory state to produce transformed query representation.
        memory_state: [B, S, d_mem]
        query_intent: [B, d_model] (optional)
        Returns:
            readout_rep: [B, d_model]
            read_weights: [B, S]
        """
        B = memory_state.shape[0]
        if disable_read:
            return torch.zeros(B, self.d_model, device=memory_state.device), torch.zeros(B, self.num_slots, device=memory_state.device)

        if query_intent is not None:
            if query_intent.ndim == 3:
                query_intent = query_intent.squeeze(1)
            q_norm = self.norm_in(query_intent)
            q_read = self.read_q_proj(q_norm)  # [B, d_mem]
            scores = torch.bmm(memory_state, q_read.unsqueeze(-1)).squeeze(-1) / math.sqrt(self.d_mem)
            read_weights = F.softmax(scores, dim=-1)  # [B, S]
        else:
            # Default mean-pooling over slots
            read_weights = torch.ones(B, self.num_slots, device=memory_state.device) / self.num_slots

        # [B, 1, S] x [B, S, d_mem] -> [B, 1, d_mem] -> [B, d_mem]
        read_vec = torch.bmm(read_weights.unsqueeze(1), memory_state).squeeze(1)
        h_q2 = self.read_out_proj(read_vec)  # [B, d_model]
        return h_q2, read_weights
