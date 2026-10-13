"""
Relation-Aware Pointer Mechanism.
Location: chakrview/runtime/relation_pointer.py

Extends ChakrPointerHead with relation-aware representation projection:
1. Projects query hidden state into relation feature space.
2. Context tokens are scored jointly by entity match and relation context.
3. Preserves identical tensor contract:
   forward(hidden_states, input_ids, vocab_logits, context_mask, query_mask, ...)
   -> (final_probs, pointer_attn, gate)
"""
from __future__ import annotations

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class ChakrRelationPointerHead(nn.Module):
    """
    Relation-Aware Neural Pointer Head.
    Computes contextual query-key matching augmented by a relation-projection pathway
    that modulates context position affinities using pooled query representation.
    """

    def __init__(
        self,
        d_model: int = 192,
        vocab_size: int = 4096,
        d_proj: int = 64,
        d_rel: int = 32,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.d_proj = d_proj
        self.d_rel = d_rel

        # Standard projections
        self.query_proj = nn.Linear(d_model, d_proj, bias=False)
        self.key_proj = nn.Linear(d_model, d_proj, bias=False)

        # Relation-aware pathway: maps pooled query hidden representation to relation vector
        self.relation_proj = nn.Linear(d_model, d_rel, bias=False)
        self.context_rel_proj = nn.Linear(d_model, d_rel, bias=False)

        # Gate network takes current hidden state and attended context vector
        self.gate_proj = nn.Linear(d_model + d_model, 1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        vocab_logits: torch.Tensor,
        context_mask: torch.Tensor,
        query_mask: Optional[torch.Tensor] = None,
        disentangle_query_overlap: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, D = hidden_states.shape

        q = self.query_proj(hidden_states)  # [B, T, d_proj]
        k = self.key_proj(hidden_states)    # [B, T, d_proj]

        # Base dot-product scores: [B, T, T]
        scores = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(self.d_proj)

        # Relation modulation if query_mask is available
        if query_mask is not None:
            # Pool query representation: mean over query positions
            # query_mask: [B, T]
            q_mask_float = query_mask.unsqueeze(-1).float()  # [B, T, 1]
            q_sum = (hidden_states * q_mask_float).sum(dim=1)  # [B, D]
            q_count = q_mask_float.sum(dim=1).clamp(min=1.0)   # [B, 1]
            pooled_q = q_sum / q_count                         # [B, D]

            r_query = self.relation_proj(pooled_q)             # [B, d_rel]
            r_context = self.context_rel_proj(hidden_states)   # [B, T, d_rel]

            # Dot product between relation vector and context token representations: [B, T]
            rel_scores = torch.bmm(r_context, r_query.unsqueeze(-1)).squeeze(-1)  # [B, T]
            rel_scores = rel_scores / math.sqrt(self.d_rel)

            # Broadcast across decoding steps: [B, 1, T] -> [B, T, T]
            scores = scores + 0.5 * rel_scores.unsqueeze(1)

        # Mask positions where context_mask is False
        mask_expanded = context_mask.unsqueeze(1).expand(B, T, T)
        scores = scores.masked_fill(~mask_expanded, -1e9)

        # Disentangled Query Overlap Adjustment
        if disentangle_query_overlap and query_mask is not None:
            overlap_mask = torch.zeros(B, T, dtype=torch.bool, device=hidden_states.device)
            for b_idx in range(B):
                q_ids_set = set(input_ids[b_idx, query_mask[b_idx]].tolist())
                for pos_idx in range(T):
                    if context_mask[b_idx, pos_idx] and input_ids[b_idx, pos_idx].item() in q_ids_set:
                        overlap_mask[b_idx, pos_idx] = True

            overlap_expanded = overlap_mask.unsqueeze(1).expand(B, T, T)
            scores = scores - 3.5 * overlap_expanded.float()

        pointer_attn = F.softmax(scores, dim=-1)
        pointer_attn = torch.nan_to_num(pointer_attn, nan=0.0)

        context_summary = torch.bmm(pointer_attn, hidden_states)

        gate_input = torch.cat([hidden_states, context_summary], dim=-1)
        gate = torch.sigmoid(self.gate_proj(gate_input))

        vocab_probs = F.softmax(vocab_logits, dim=-1)
        copy_probs = torch.zeros(B, T, self.vocab_size, device=hidden_states.device, dtype=vocab_probs.dtype)
        expanded_input_ids = input_ids.unsqueeze(1).expand(B, T, T)
        copy_probs.scatter_add_(dim=2, index=expanded_input_ids, src=pointer_attn)

        final_probs = (1.0 - gate) * vocab_probs + gate * copy_probs
        return final_probs, pointer_attn, gate
