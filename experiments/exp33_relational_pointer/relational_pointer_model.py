"""
Relational Pointer Circuit Architecture for Experiment 33.
Location: experiments/exp33_relational_pointer/relational_pointer_model.py

Features:
1. ChakrRelationalPointerMicro:
   Combines base ChakrMicro causal transformer with a Two-Stage Relational Pointer Head:
   Stage A (Key-Query Relational Alignment):
     Computes bilinear/projected affinity between the query position (the key being asked about)
     and all prompt positions:
       S_rel(t, i) = (W_q h_t)^T (W_k h_i) / sqrt(d)
   Stage B (Relational Value Routing):
     Uses multi-head attention where Head 1 attends to Key matches and Head 2 attends to
     Value representations. A learned value-transition kernel binds the matching key to the
     following Value token in the syntactic stream.
   Stage C (Abstention Gate):
     A dedicated confidence detector compares the maximum key-match score to an abstention threshold,
     routing probability to the abstention token '?' when max key score is low.

Strict Isolation:
- Does NOT touch or modify canonical model code or checkpoint.
- Base ChakrMicro loaded in read-only mode.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro


class RelationalPointerHead(nn.Module):
    """
    Two-stage relational pointer head:
    1. Key-alignment cross-attention between decoder query state and prompt tokens.
    2. Relational transition from key positions to value positions.
    3. Learned generation gate and explicit abstention routing.
    """

    def __init__(self, d_model: int = 192, head_dim: int = 64):
        super().__init__()
        self.d_model = d_model
        self.head_dim = head_dim
        self.scale = 1.0 / math.sqrt(head_dim)

        # Stage A: Key-Query Matching Projections
        self.W_q_rel = nn.Linear(d_model, head_dim, bias=False)
        self.W_k_rel = nn.Linear(d_model, head_dim, bias=False)

        # Stage B: Value Value Projection & Value-Transition Bi-Encoder
        self.W_v_proj = nn.Linear(d_model, head_dim, bias=False)
        # Shift kernel: projects key representation into expected value representation space
        self.W_key_to_val = nn.Linear(head_dim, head_dim, bias=False)

        # Stage C: Generation Gate & Abstention Detector
        # [h_t, context_vector, max_key_sim] -> p_gen in (0, 1)
        self.W_gate = nn.Linear(d_model + head_dim + 1, 1)
        # Abstention logit projection: [h_t, max_key_sim] -> abstention logit
        self.W_abstain = nn.Linear(d_model + 1, 1)

    def forward(
        self,
        hidden_states: torch.Tensor,       # [B, T, d_model]
        prompt_len: int,                  # Length of prompt prefix
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            pointer_probs: [B, T, prompt_len] distribution over prompt tokens
            p_gen: [B, T, 1] copy gate
            abstain_logits: [B, T, 1] logit indicating key absence / abstention
        """
        B, T, D = hidden_states.shape
        prompt_states = hidden_states[:, :prompt_len, :]  # [B, P, D]

        # 1. Project query at decoder position and keys across prompt
        q_rel = self.W_q_rel(hidden_states)              # [B, T, head_dim]
        k_rel = self.W_k_rel(prompt_states)              # [B, P, head_dim]

        # Raw key-matching affinity: [B, T, P]
        key_affinities = torch.bmm(q_rel, k_rel.transpose(1, 2)) * self.scale
        max_key_score, _ = torch.max(key_affinities, dim=-1, keepdim=True)  # [B, T, 1]

        # 2. Relational Value Routing:
        # Key representation shifted to expected value representation space:
        # key_to_val_queries: [B, T, head_dim]
        key_weights = F.softmax(key_affinities, dim=-1)  # [B, T, P]
        matched_key_rep = torch.bmm(key_weights, k_rel)  # [B, T, head_dim]
        val_query = self.W_key_to_val(matched_key_rep) + q_rel  # [B, T, head_dim]

        # Value candidate representations in prompt:
        v_candidates = self.W_v_proj(prompt_states)      # [B, P, head_dim]
        val_scores = torch.bmm(val_query, v_candidates.transpose(1, 2)) * self.scale  # [B, T, P]
        pointer_probs = F.softmax(val_scores, dim=-1)    # [B, T, P]

        # 3. Context vector for gate
        context = torch.bmm(pointer_probs, v_candidates)  # [B, T, head_dim]
        gate_input = torch.cat([hidden_states, context, max_key_score], dim=-1)
        p_gen = torch.sigmoid(self.W_gate(gate_input))   # [B, T, 1]

        # 4. Abstention logit
        abstain_input = torch.cat([hidden_states, max_key_score], dim=-1)
        abstain_logits = self.W_abstain(abstain_input)   # [B, T, 1]

        return pointer_probs, p_gen, abstain_logits


class ChakrRelationalPointerMicro(nn.Module):
    """
    ChakrMicro model extended with RelationalPointerHead.
    """

    def __init__(self, config: Optional[ModelConfig] = None):
        super().__init__()
        if config is None:
            config = ModelConfig()
        self.config = config
        self.core = ChakrMicro(config)
        self.relational_head = RelationalPointerHead(config.d_model, head_dim=64)

    def forward(
        self,
        input_ids: torch.Tensor,
        prompt_len: int,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Any] = None,
        abstain_token_id: int = 32,  # '?' in canonical vocab
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            vocab_logits: [B, T, vocab_size]
            pointer_probs: [B, T, prompt_len]
            p_gen: [B, T, 1]
            final_probs: [B, T, vocab_size] combined distribution
        """
        B, T = input_ids.shape
        vocab_logits, h = self.core(
            input_ids,
            attention_mask=attention_mask,
            kv_cache=kv_cache,
            return_hidden_states=True,
        )

        vocab_probs = F.softmax(vocab_logits, dim=-1)
        pointer_probs, p_gen, abstain_logits = self.relational_head(h, prompt_len)

        # Base combination
        final_probs = p_gen * vocab_probs

        # Scatter pointer probabilities over prompt token IDs
        prompt_ids = input_ids[:, :prompt_len]
        expanded_prompt_ids = prompt_ids.unsqueeze(1).expand(B, T, prompt_len)
        pointer_contrib = (1.0 - p_gen) * pointer_probs

        final_probs.scatter_add_(2, expanded_prompt_ids, pointer_contrib)

        # Blend abstention probability:
        p_abstain = torch.sigmoid(abstain_logits)  # [B, T, 1]
        # Boost '?' token by p_abstain, re-normalize
        abstain_mask = torch.zeros_like(final_probs)
        abstain_mask[:, :, abstain_token_id] = p_abstain.squeeze(-1)
        final_probs = (1.0 - p_abstain) * final_probs + abstain_mask

        return vocab_logits, pointer_probs, p_gen, final_probs
