"""Step 249: Associative Pointer Architecture.

Implements an isolated neural circuit that composes:
1. Query representation (from prompt hidden state)
2. Neural query-key matching (soft key distribution alpha_k)
3. Selected key position distribution
4. Learned key->value association routing (relational association matrix A)
5. Contextual pointer distribution over input positions (alpha_v = alpha_k @ A)
6. Generation / pointer output combination:
   P(output_token) = g * P_gen(token) + (1 - g) * sum_{i: x_i = token} P_pointer(i)

STRICT INVARIANTS:
- Isolated candidate module (zero drift to canonical baseline ChakrMicro)
- Purely neural: learned linear projections, bidirectional contextualization, softmax
- No hard-coded token offsets, no dictionary lookups, no Python heuristics
"""

from __future__ import annotations

import dataclasses
import hashlib
from typing import Dict, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from chakrview.brain.model import ChakrMicro


@dataclasses.dataclass
class AssociativePointerOutput:
    combined_logits: torch.Tensor           # [B, vocab_size]
    gen_logits: torch.Tensor                # [B, vocab_size]
    key_distribution: torch.Tensor          # [B, T-1] (alpha_k)
    value_distribution: torch.Tensor        # [B, T-1] (alpha_v)
    query_key_distribution: torch.Tensor    # [B, T-1] (alpha_q)
    copy_prob: torch.Tensor                 # [B, 1] (1 - g)
    gen_prob: torch.Tensor                  # [B, 1] (g)
    selected_key_pos: torch.Tensor          # [B]
    selected_val_pos: torch.Tensor          # [B]
    selected_val_token: torch.Tensor        # [B]


class AssociativePointerHead(nn.Module):
    """Modular neural circuit composing Query-Key matching, Key->Value routing, and contextual pointer."""

    def __init__(
        self,
        d_model: int = 192,
        d_slot: int = 64,
        vocab_size: int = 4096,
        num_ctx_layers: int = 2,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_slot = d_slot
        self.vocab_size = vocab_size

        # 1. Bidirectional contextual relation encoder:
        # Enriches hidden states with clause-level syntactic/semantic associations
        self.ctx_layers = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=4,
                dim_feedforward=256,
                batch_first=True,
                dropout=0.0,
            ),
            num_layers=num_ctx_layers,
        )

        # 2. Query representation extractor: attends from prompt end to query token in context
        self.q_ext = nn.Linear(d_model, d_slot, bias=False)
        self.k_ext = nn.Linear(d_model, d_slot, bias=False)

        # 3. Neural Query-Key Matcher: aligns query representation with candidate key positions
        self.q_match = nn.Linear(d_model, d_slot, bias=False)
        self.k_match = nn.Linear(d_model, d_slot, bias=False)

        # 4. Learned Key->Value Association Router: routes from matched key state to associated value position
        self.k_route = nn.Linear(d_model, d_slot, bias=False)
        self.v_route = nn.Linear(d_model, d_slot, bias=False)

        # 5. Learned copy/generation gate: g = sigmoid(W_g * h + b_g)
        self.gate_proj = nn.Linear(d_model, 1)

    def forward(
        self,
        hidden_states: torch.Tensor,       # [B, T, d_model]
        gen_logits: torch.Tensor,          # [B, T, vocab_size]
        input_ids: torch.Tensor,           # [B, T]
        force_mode: Optional[str] = None,  # None, "pointer_only", "gen_only"
    ) -> AssociativePointerOutput:
        B, T, D = hidden_states.shape

        # 1. Contextual relation encoding
        h_ctx = self.ctx_layers(hidden_states)  # [B, T, D]
        h_last = h_ctx[:, -1:, :]               # [B, 1, D]
        ctx_reps = h_ctx[:, :-1, :]             # [B, T-1, D] candidate positions

        # 2. Query representation extraction
        s_q = torch.bmm(self.q_ext(h_last), self.k_ext(ctx_reps).transpose(1, 2)) / (self.d_slot ** 0.5)
        alpha_q = F.softmax(s_q, dim=-1)        # [B, 1, T-1]
        h_q = torch.bmm(alpha_q, ctx_reps)      # [B, 1, D]

        # 3. Neural Query-Key Matching
        s_m = torch.bmm(self.q_match(h_q), self.k_match(ctx_reps).transpose(1, 2)) / (self.d_slot ** 0.5)
        alpha_k = F.softmax(s_m, dim=-1)        # [B, 1, T-1] selected key distribution
        h_k = torch.bmm(alpha_k, ctx_reps)      # [B, 1, D] selected key contextual state

        # 4. Learned Key->Value Association Routing
        s_v = torch.bmm(self.k_route(h_k), self.v_route(ctx_reps).transpose(1, 2)) / (self.d_slot ** 0.5)
        alpha_v = F.softmax(s_v, dim=-1)        # [B, 1, T-1] selected value position distribution

        # 5. Build full sequence pointer distribution
        alpha_v_full = F.pad(alpha_v, (0, 1), value=0.0)  # [B, 1, T]

        # 6. Scatter pointer probabilities over vocabulary space
        ptr_dist = torch.zeros(B, 1, self.vocab_size, device=hidden_states.device, dtype=hidden_states.dtype)
        expanded_ids = input_ids.unsqueeze(1)             # [B, 1, T]
        ptr_dist.scatter_add_(dim=2, index=expanded_ids, src=alpha_v_full)

        # 7. Generation distribution & gate
        gate = torch.sigmoid(self.gate_proj(hidden_states[:, -1:, :]))  # [B, 1, 1]
        if force_mode == "pointer_only":
            p_gen = torch.zeros_like(gate)
            p_copy = torch.ones_like(gate)
        elif force_mode == "gen_only":
            p_gen = torch.ones_like(gate)
            p_copy = torch.zeros_like(gate)
        else:
            p_gen = gate
            p_copy = 1.0 - gate

        gen_prob = F.softmax(gen_logits[:, -1:, :], dim=-1)  # [B, 1, vocab_size]
        p_final = p_gen * gen_prob + p_copy * ptr_dist
        p_final = torch.clamp(p_final, min=1e-12)
        combined_logits = torch.log(p_final.squeeze(1))

        # Selected positions (argmax)
        sel_k_pos = torch.argmax(alpha_k.squeeze(1), dim=-1)
        sel_v_pos = torch.argmax(alpha_v.squeeze(1), dim=-1)
        sel_v_tok = torch.gather(input_ids, dim=1, index=sel_v_pos.unsqueeze(1)).squeeze(1)

        return AssociativePointerOutput(
            combined_logits=combined_logits,
            gen_logits=gen_logits[:, -1, :],
            key_distribution=alpha_k.squeeze(1),
            value_distribution=alpha_v.squeeze(1),
            query_key_distribution=alpha_q.squeeze(1),
            copy_prob=p_copy.squeeze(1),
            gen_prob=p_gen.squeeze(1),
            selected_key_pos=sel_k_pos,
            selected_val_pos=sel_v_pos,
            selected_val_token=sel_v_tok,
        )


class ChakrMicroWithAssociativePointer(nn.Module):
    """Isolated candidate model wrapping ChakrMicro with AssociativePointerHead."""

    def __init__(
        self,
        base_model: ChakrMicro,
        force_mode: Optional[str] = None,
        d_slot: int = 64,
        num_ctx_layers: int = 2,
    ):
        super().__init__()
        self.base_model = base_model
        self.force_mode = force_mode
        self.pointer_head = AssociativePointerHead(
            d_model=base_model.config.d_model,
            d_slot=d_slot,
            vocab_size=base_model.config.vocab_size,
            num_ctx_layers=num_ctx_layers,
        )

    @property
    def config(self):
        return self.base_model.config

    def forward(
        self,
        input_ids: torch.Tensor,
        force_mode: Optional[str] = None,
    ) -> Tuple[torch.Tensor, AssociativePointerOutput]:
        mode = force_mode or self.force_mode

        # Run base transformer backbone forward pass
        x = self.base_model.embedding(input_ids)
        for i, layer in enumerate(self.base_model.layers):
            x = layer(x, attention_mask=None, kv_cache=None, layer_idx=i)
        hidden_states = self.base_model.final_norm(x)
        gen_logits = self.base_model.lm_head(hidden_states)

        ptr_out = self.pointer_head(
            hidden_states=hidden_states,
            gen_logits=gen_logits,
            input_ids=input_ids,
            force_mode=mode,
        )
        return ptr_out.combined_logits, ptr_out


def compute_module_parameter_hash(module: nn.Module) -> str:
    """Computes deterministic SHA-256 hash of a module's parameters."""
    hasher = hashlib.sha256()
    with torch.no_grad():
        for name, param in sorted(module.named_parameters()):
            hasher.update(name.encode("utf-8"))
            hasher.update(param.detach().cpu().numpy().tobytes())
    return hasher.hexdigest()
