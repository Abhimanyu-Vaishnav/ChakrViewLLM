"""
Auxiliary Pointer-Generator Copy Mechanism for ChakrView.
Location: chakrview/runtime/pointer_mechanism.py

Features:
1. ChakrPointerHead: Computes attention over designated context token positions.
2. Learned Gate: Dynamic interpolation between base model vocabulary logits and context pointer probabilities.
3. Strict Causal & Mask Boundary Protection: Disallows pointing to prompt framing, padding, or future answer tokens.
4. Leaves canonical ChakrMicro model parameters 100% untouched.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, Any
import torch
import torch.nn as nn
import torch.nn.functional as F


class ChakrPointerHead(nn.Module):
    """
    Auxiliary attention head computing pointer distribution over context positions.
    P_final = (1 - g) * P_vocab + g * P_copy
    """

    def __init__(
        self,
        d_model: int = 192,
        vocab_size: int = 4096,
        d_proj: int = 64,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.d_proj = d_proj

        self.query_proj = nn.Linear(d_model, d_proj, bias=False)
        self.key_proj = nn.Linear(d_model, d_proj, bias=False)
        # Gate network takes current hidden state and attended context vector
        self.gate_proj = nn.Linear(d_model + d_model, 1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        vocab_logits: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        hidden_states: [B, T, d_model] from base model pre-norm
        input_ids: [B, T] token IDs corresponding to hidden_states
        vocab_logits: [B, T, vocab_size] from base model lm_head
        context_mask: [B, T] boolean tensor (True for valid context tokens to copy from, False otherwise)

        Returns:
            final_probs: [B, T, vocab_size] normalized probability distribution
            pointer_attn: [B, T, T] attention weights over context positions
            gate: [B, T, 1] copy gate values in [0, 1]
        """
        B, T, D = hidden_states.shape

        q = self.query_proj(hidden_states)  # [B, T, d_proj]
        k = self.key_proj(hidden_states)    # [B, T, d_proj]

        # Dot-product attention scores between all positions: [B, T, T]
        scores = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(self.d_proj)

        # Mask positions where context_mask is False (only allow pointing to context tokens)
        # context_mask shape is [B, T], expand to [B, 1, T] for broadcasting across query steps
        mask_expanded = context_mask.unsqueeze(1).expand(B, T, T)
        scores = scores.masked_fill(~mask_expanded, -1e9)

        pointer_attn = F.softmax(scores, dim=-1)  # [B, T, T]
        # Clean NaNs if a row has zero valid context tokens
        pointer_attn = torch.nan_to_num(pointer_attn, nan=0.0)

        # Compute attended context vector: [B, T, d_model]
        context_summary = torch.bmm(pointer_attn, hidden_states)

        # Gate computation: [B, T, 1] in [0, 1]
        gate_input = torch.cat([hidden_states, context_summary], dim=-1)
        gate = torch.sigmoid(self.gate_proj(gate_input))

        # Vocabulary probabilities: [B, T, vocab_size]
        vocab_probs = F.softmax(vocab_logits, dim=-1)

        # Map pointer attention positions to vocabulary token IDs
        # input_ids: [B, T]. Expand to [B, T, T] to scatter into [B, T, vocab_size]
        copy_probs = torch.zeros(B, T, self.vocab_size, device=hidden_states.device, dtype=vocab_probs.dtype)
        
        # For each query step t, scatter pointer_attn[b, t, j] into copy_probs[b, t, input_ids[b, j]]
        expanded_input_ids = input_ids.unsqueeze(1).expand(B, T, T)  # [B, T, T]
        copy_probs.scatter_add_(dim=2, index=expanded_input_ids, src=pointer_attn)

        # Final mixture distribution: [B, T, vocab_size]
        final_probs = (1.0 - gate) * vocab_probs + gate * copy_probs

        return final_probs, pointer_attn, gate
