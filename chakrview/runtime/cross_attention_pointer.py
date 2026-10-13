"""
Multi-Hop Query-to-Candidate Cross-Attention Pointer Mechanism.
Location: chakrview/runtime/cross_attention_pointer.py

Introduces a lightweight multi-head cross-attention layer between query token representations
and candidate context token representations:
1. Query Tokens Q: extracted via query_mask.
2. Context Tokens K, V: extracted via context_mask.
3. Cross-Attention: Context representations are enriched by query cross-attention,
   resolving intermediate entity bridges and suppressing spurious direct distractors.
4. Preserves identical tensor contract:
   forward(hidden_states, input_ids, vocab_logits, context_mask, query_mask, ...)
   -> (final_probs, pointer_attn, gate)
"""
from __future__ import annotations

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class ChakrCrossAttentionPointerHead(nn.Module):
    """
    Cross-Attention Augmented Neural Pointer Head for Relational Multi-Hop Grounding.
    Allows query tokens to attend directly to context tokens via multi-head cross-attention,
    enabling two-hop link resolution before pointer probability computation.
    """

    def __init__(
        self,
        d_model: int = 192,
        vocab_size: int = 4096,
        d_proj: int = 64,
        n_heads: int = 2,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.d_proj = d_proj
        self.n_heads = n_heads

        # Pointer Query / Key Projections
        self.query_proj = nn.Linear(d_model, d_proj, bias=False)
        self.key_proj = nn.Linear(d_model, d_proj, bias=False)

        # Cross-Attention Block: Context attends to Query
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            batch_first=True,
        )
        self.layer_norm = nn.LayerNorm(d_model)

        # Copy gate projection
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

        # If query_mask is present, perform query-to-context cross attention
        context_enriched = hidden_states
        if query_mask is not None and query_mask.any():
            # Query tokens as Key/Value for cross-attention
            # For each batch element, run cross-attention over query positions
            # We can use key_padding_mask on inverted query_mask
            key_padding_mask = ~query_mask  # True where padding (non-query)
            # Only apply if at least one query token exists in each batch
            if not key_padding_mask.all(dim=1).any():
                cross_out, _ = self.cross_attn(
                    query=hidden_states,
                    key=hidden_states,
                    value=hidden_states,
                    key_padding_mask=key_padding_mask,
                    need_weights=False,
                )
                context_enriched = self.layer_norm(hidden_states + cross_out)

        # Project enriched representations to pointer query and key
        q = self.query_proj(context_enriched)  # [B, T, d_proj]
        k = self.key_proj(context_enriched)    # [B, T, d_proj]

        scores = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(self.d_proj)

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

        context_summary = torch.bmm(pointer_attn, context_enriched)
        gate_input = torch.cat([hidden_states, context_summary], dim=-1)
        gate = torch.sigmoid(self.gate_proj(gate_input))

        vocab_probs = F.softmax(vocab_logits, dim=-1)
        copy_probs = torch.zeros(B, T, self.vocab_size, device=hidden_states.device, dtype=vocab_probs.dtype)
        expanded_input_ids = input_ids.unsqueeze(1).expand(B, T, T)
        copy_probs.scatter_add_(dim=2, index=expanded_input_ids, src=pointer_attn)

        final_probs = (1.0 - gate) * vocab_probs + gate * copy_probs
        return final_probs, pointer_attn, gate
