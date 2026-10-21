"""
start_conditioned_span_model.py
ChakrView Experiment 10.2 — Start-Conditioned Span Pointer Model

Implements the candidate correction:
- Start scoring: q_start = W_q_start * h_query
  Score_start(s) = q_start^T (W_k_start * h_s) / sqrt(d)
- Start-conditioned End scoring:
  For candidate start position s, end query is:
  q_end(s) = W_q_end * (h_query + h_s)
  Score_end(e | s) = q_end(s)^T (W_k_end * h_e) / sqrt(d)
- Joint Inference:
  (s*, e*) = argmax_{s <= e <= s + max_span_len} [ Score_start(s) + Score_end(e | s) ]
- Oracle Mode Support:
  predict_span(input_ids, mask, oracle_start=s_gold) computes Score_end(e | s_gold).
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro

class ChakrStartConditionedSpanPointer(nn.Module):
    def __init__(self, config: ModelConfig, max_span_len: int = 8, freeze_backbone: bool = False):
        super().__init__()
        self.config = config
        self.max_span_len = max_span_len
        self.backbone = ChakrMicro(config)
        self.d_model = config.d_model  # 192

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # Start pointer projection
        self.query_start = nn.Linear(self.d_model, self.d_model, bias=False)
        self.key_start = nn.Linear(self.d_model, self.d_model, bias=False)

        # Start-conditioned End pointer projection
        # Takes (h_query + h_start) -> q_end
        self.query_end = nn.Linear(self.d_model, self.d_model, bias=False)
        self.key_end = nn.Linear(self.d_model, self.d_model, bias=False)

        self.scale = 1.0 / math.sqrt(self.d_model)

    def extract_hidden_states(self, idx: torch.Tensor) -> torch.Tensor:
        _, h = self.backbone(idx, return_hidden_states=True)
        return h

    def forward(
        self,
        input_ids: torch.Tensor,
        candidate_mask: Optional[torch.Tensor] = None,
        gold_start: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Training forward pass:
        input_ids: [B, T]
        candidate_mask: [B, T]
        gold_start: [B] (ground truth start position for teacher-forced end query during training)
        Returns:
            start_scores: [B, T]
            end_scores: [B, T] (conditioned on gold_start if provided)
        """
        h = self.extract_hidden_states(input_ids)  # [B, T, d_model]
        B, T, _ = h.shape
        query = h[:, -1:, :]  # [B, 1, d_model]

        # 1. Start scoring
        q_s = self.query_start(query)
        k_s = self.key_start(h)
        start_scores = torch.bmm(q_s, k_s.transpose(1, 2)).squeeze(1) * self.scale  # [B, T]

        # 2. End scoring conditioned on start
        if gold_start is not None:
            # Gather h at gold_start
            batch_idx = torch.arange(B, device=input_ids.device)
            h_start = h[batch_idx, gold_start, :].unsqueeze(1)  # [B, 1, d_model]
            cond_query = query + h_start
        else:
            # During evaluation without gold_start, fallback to prompt query
            cond_query = query

        q_e = self.query_end(cond_query)
        k_e = self.key_end(h)
        end_scores = torch.bmm(q_e, k_e.transpose(1, 2)).squeeze(1) * self.scale  # [B, T]

        if candidate_mask is not None:
            start_scores = start_scores.masked_fill(~candidate_mask, float("-inf"))
            end_scores = end_scores.masked_fill(~candidate_mask, float("-inf"))

        return start_scores, end_scores

    def predict_span(
        self,
        input_ids: torch.Tensor,
        candidate_mask: Optional[torch.Tensor] = None,
        oracle_start: Optional[int] = None
    ) -> Tuple[int, int, List[int], float]:
        """
        Inference with start-conditioned end search.
        If oracle_start is provided, evaluates conditional end prediction directly.
        """
        self.eval()
        with torch.no_grad():
            h = self.extract_hidden_states(input_ids)  # [1, T, d_model]
            T = input_ids.shape[1]
            query = h[:, -1:, :]  # [1, 1, d_model]

            # Start scores
            q_s = self.query_start(query)
            k_s = self.key_start(h)
            start_scores = torch.bmm(q_s, k_s.transpose(1, 2)).squeeze(1)[0] * self.scale  # [T]
            if candidate_mask is not None:
                start_scores = start_scores.masked_fill(~candidate_mask[0], float("-inf"))

            k_e = self.key_end(h)  # [1, T, d_model]

            if oracle_start is not None:
                # Oracle mode: start is fixed to oracle_start
                best_s = oracle_start
                h_s = h[:, best_s:best_s+1, :]
                q_e = self.query_end(query + h_s)
                end_scores = torch.bmm(q_e, k_e.transpose(1, 2)).squeeze(1)[0] * self.scale  # [T]
                if candidate_mask is not None:
                    end_scores = end_scores.masked_fill(~candidate_mask[0], float("-inf"))

                best_val = float("-inf")
                best_e = best_s
                for e in range(best_s, min(best_s + self.max_span_len, T)):
                    if candidate_mask is not None and not candidate_mask[0, e]:
                        continue
                    sc = end_scores[e].item()
                    if sc > best_val:
                        best_val = sc
                        best_e = e

                extracted_tokens = input_ids[0, best_s:best_e + 1].tolist()
                return best_s, best_e, extracted_tokens, float(best_val)

            # Normal Joint Search:
            # Top candidate starts
            top_k_starts = torch.topk(start_scores, k=min(5, T)).indices.tolist()
            best_joint_score = float("-inf")
            best_s, best_e = 0, 0

            for s in top_k_starts:
                if candidate_mask is not None and not candidate_mask[0, s]:
                    continue
                s_score = start_scores[s].item()
                if math.isinf(s_score):
                    continue

                # Condition end query on this start representation
                h_s = h[:, s:s+1, :]
                q_e = self.query_end(query + h_s)
                e_scores = torch.bmm(q_e, k_e.transpose(1, 2)).squeeze(1)[0] * self.scale  # [T]
                if candidate_mask is not None:
                    e_scores = e_scores.masked_fill(~candidate_mask[0], float("-inf"))

                for e in range(s, min(s + self.max_span_len, T)):
                    if candidate_mask is not None and not candidate_mask[0, e]:
                        continue
                    joint_score = s_score + e_scores[e].item()
                    if joint_score > best_joint_score:
                        best_joint_score = joint_score
                        best_s = s
                        best_e = e

            extracted_tokens = input_ids[0, best_s:best_e + 1].tolist()
            return best_s, best_e, extracted_tokens, float(best_joint_score)
