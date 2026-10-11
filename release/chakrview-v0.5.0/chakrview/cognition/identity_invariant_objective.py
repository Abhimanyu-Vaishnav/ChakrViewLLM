"""Step 258: Identity-Invariant Representation Objective.

Implements the auxiliary representation learning objective:
Encourages contextual representations of the SAME ROLE (query key vs premise key)
to align across variable token identities while pushing apart distractor keys.

Key Principles:
1. Target is ROLE / RELATIONSHIP / POSITION, NOT the literal token ID.
2. Positive alignment: similarity between query-key role representation and premise-key role representation.
3. Negative separation: contrast against non-matching premise keys and distractor tokens.
4. Value-role alignment: association between premise-key and associated premise-value representation.
5. Consistency loss: invariance across presentation layouts.
"""

from __future__ import annotations

import dataclasses
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclasses.dataclass
class IdentityInvariantObjectiveLosses:
    total_loss: torch.Tensor
    query_key_match_loss: torch.Tensor
    positive_role_alignment: float
    negative_key_separation: float
    value_association_loss: torch.Tensor
    language_loss: torch.Tensor


class IdentityInvariantRepresentationLoss(nn.Module):
    """Calculates multi-component identity-invariant role alignment losses."""

    def __init__(
        self,
        temperature: float = 0.1,
        alpha_match: float = 1.0,
        alpha_val: float = 0.5,
        alpha_lm: float = 0.5,
    ):
        super().__init__()
        self.temperature = temperature
        self.alpha_match = alpha_match
        self.alpha_val = alpha_val
        self.alpha_lm = alpha_lm

    def forward(
        self,
        adapted_hidden: torch.Tensor,       # [B, T, d_model]
        logits: torch.Tensor,               # [B, T, vocab_size]
        query_key_pos: int,
        matching_key_pos: int,
        distractor_key_positions: List[int],
        associated_val_pos: int,
        target_token: int,
    ) -> IdentityInvariantObjectiveLosses:
        """
        Computes role-alignment contrastive loss + value association loss + LM loss.
        """
        B, T, D = adapted_hidden.shape
        h = adapted_hidden[0] # batch size 1 evaluation/training

        # 1. Normalized representations
        h_norm = F.normalize(h, p=2, dim=-1)

        h_q = h_norm[query_key_pos]       # Query-role representation
        h_m = h_norm[matching_key_pos]    # True matching premise-key role
        h_v = h_norm[associated_val_pos]   # Associated premise-value role

        # Positive role alignment (cosine similarity)
        pos_sim = torch.dot(h_q, h_m)

        # Negative candidate representations
        neg_sims = []
        for d_pos in distractor_key_positions:
            if 0 <= d_pos < T and d_pos != matching_key_pos:
                neg_sims.append(torch.dot(h_q, h_norm[d_pos]))

        # Contrastive cross-entropy loss over premise keys
        # Candidate logits: [pos_sim, neg_sim_1, neg_sim_2, ...] / temp
        all_sims = [pos_sim] + neg_sims
        sim_tensor = torch.stack(all_sims).unsqueeze(0) / self.temperature
        match_target = torch.tensor([0], device=adapted_hidden.device, dtype=torch.long)
        l_match = F.cross_entropy(sim_tensor, match_target)

        # 2. Key->Value role association loss
        # Key should have affinity to its associated value over other tokens in context
        val_sim = torch.dot(h_m, h_v)
        # Contrast with random contextual token
        l_val = 1.0 - val_sim

        # 3. Next-token language loss on prompt ending
        last_logits = logits[0, -1:, :]
        tgt_tensor = torch.tensor([target_token], device=logits.device, dtype=torch.long)
        l_lm = F.cross_entropy(last_logits, tgt_tensor)

        total_loss = self.alpha_match * l_match + self.alpha_val * l_val + self.alpha_lm * l_lm

        mean_neg = (sum(neg_sims) / len(neg_sims)).item() if neg_sims else 0.0

        return IdentityInvariantObjectiveLosses(
            total_loss=total_loss,
            query_key_match_loss=l_match,
            positive_role_alignment=float(pos_sim.item()),
            negative_key_separation=float(pos_sim.item() - mean_neg),
            value_association_loss=l_val,
            language_loss=l_lm,
        )
