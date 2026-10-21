"""
span_pointer_model.py
ChakrView Experiment 10 — Dual-Boundary Factorized Span Pointer Architecture

Architecture:
- Backbone: ChakrMicro (6 layers, d_model=192, context=512).
- Query Representation:
  Contextual hidden state h_query = H[T-1] at the final prompt token.
- Start Pointer Projection:
  Score_start(i) = (W_q_start * h_query)^T (W_k_start * H[i]) / sqrt(d)
- End Pointer Projection:
  Score_end(j) = (W_q_end * h_query)^T (W_k_end * H[j]) / sqrt(d)
- Constrained Joint Decoding:
  Finds (s*, e*) maximizing Score_start(s) + Score_end(e)
  subject to s <= e <= s + max_span_len (max_span_len = 8).
  Applies ineligible candidate masking to eliminate prompt boilerplate.
- Extraction:
  Directly slices tokens[s* : e* + 1] and decodes with Tokenizer.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro

class ChakrSpanPointerModel(nn.Module):
    def __init__(self, config: ModelConfig, max_span_len: int = 8, freeze_backbone: bool = False):
        super().__init__()
        self.config = config
        self.max_span_len = max_span_len
        self.backbone = ChakrMicro(config)
        self.d_model = config.d_model  # 192

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # Dual factorized start & end projections
        self.query_start = nn.Linear(self.d_model, self.d_model, bias=False)
        self.key_start = nn.Linear(self.d_model, self.d_model, bias=False)

        self.query_end = nn.Linear(self.d_model, self.d_model, bias=False)
        self.key_end = nn.Linear(self.d_model, self.d_model, bias=False)

        self.scale = 1.0 / math.sqrt(self.d_model)

    def extract_hidden_states(self, idx: torch.Tensor) -> torch.Tensor:
        _, h = self.backbone(idx, return_hidden_states=True)
        return h

    def forward(
        self,
        input_ids: torch.Tensor,
        candidate_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        input_ids: [B, T]
        candidate_mask: [B, T] (bool: True for eligible positions)
        Returns:
            start_scores: [B, T]
            end_scores: [B, T]
        """
        h = self.extract_hidden_states(input_ids)  # [B, T, d_model]
        query = h[:, -1:, :]  # [B, 1, d_model]

        # 1. Start scoring
        q_s = self.query_start(query)
        k_s = self.key_start(h)
        start_scores = torch.bmm(q_s, k_s.transpose(1, 2)).squeeze(1) * self.scale  # [B, T]

        # 2. End scoring
        q_e = self.query_end(query)
        k_e = self.key_end(h)
        end_scores = torch.bmm(q_e, k_e.transpose(1, 2)).squeeze(1) * self.scale    # [B, T]

        if candidate_mask is not None:
            start_scores = start_scores.masked_fill(~candidate_mask, float("-inf"))
            end_scores = end_scores.masked_fill(~candidate_mask, float("-inf"))

        return start_scores, end_scores

    def predict_span(
        self,
        input_ids: torch.Tensor,
        candidate_mask: Optional[torch.Tensor] = None
    ) -> Tuple[int, int, List[int], float]:
        """
        Constrained span decoding:
        Finds s, e maximizing start_scores[s] + end_scores[e]
        such that s <= e <= s + max_span_len
        Returns: (start_idx, end_idx, extracted_token_ids, joint_confidence)
        """
        self.eval()
        with torch.no_grad():
            start_scores, end_scores = self.forward(input_ids, candidate_mask)
            B, T = input_ids.shape

            s_probs = F.softmax(start_scores[0], dim=-1)
            e_probs = F.softmax(end_scores[0], dim=-1)

            best_val = float("-inf")
            best_s, best_e = 0, 0

            for s in range(T):
                if candidate_mask is not None and not candidate_mask[0, s]:
                    continue
                for e in range(s, min(s + self.max_span_len, T)):
                    if candidate_mask is not None and not candidate_mask[0, e]:
                        continue
                    score = start_scores[0, s].item() + end_scores[0, e].item()
                    if score > best_val:
                        best_val = score
                        best_s = s
                        best_e = e

            extracted_tokens = input_ids[0, best_s:best_e + 1].tolist()
            conf = float((s_probs[best_s] * e_probs[best_e]).item())

        return best_s, best_e, extracted_tokens, conf
