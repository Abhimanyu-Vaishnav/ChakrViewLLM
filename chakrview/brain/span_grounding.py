"""
Batches 861-871: Span-Level Argument Grounding Architecture for ChakrMicro.
Location: chakrview/brain/span_grounding.py

Features:
1. SpanGroundingHead:
   Predicts start index, end index, span type, and confidence over prompt tokens.
2. Contiguous Span Preservation:
   Multi-token BPE sequences (e.g. '600' -> ['6', '00']) are treated as single atomic spans.
3. Positional Disambiguation:
   Distinguishes repeated values by explicit token index coordinates.
4. Missing Argument Detection:
   Outputs an ungrounded flag or low confidence when no valid source span is found.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro


SPAN_TYPES = ["NONE", "NUMBER", "STRING", "IDENTIFIER"]
SPAN_TYPE_TO_ID = {t: i for i, t in enumerate(SPAN_TYPES)}


@dataclass
class ExtractedSpan:
    start_idx: int
    end_idx: int
    span_type: str
    confidence: float
    span_text: str = ""


class SpanGroundingHead(nn.Module):
    """
    Predicts start and end boundaries, semantic type, and confidence
    for argument spans located in the prompt tokens.
    """

    def __init__(self, d_model: int = 192, max_spans: int = 4):
        super().__init__()
        self.d_model = d_model
        self.max_spans = max_spans

        # Query embeddings representing argument slots (e.g., slot 0, slot 1, etc.)
        self.slot_queries = nn.Embedding(max_spans, d_model)

        # Boundary predictors
        self.W_start = nn.Linear(d_model * 2, 1)
        self.W_end = nn.Linear(d_model * 3, 1)

        # Type classification and confidence
        self.type_classifier = nn.Linear(d_model * 2, len(SPAN_TYPES))
        self.conf_head = nn.Linear(d_model * 2, 1)

    def forward(
        self,
        prompt_hidden_states: torch.Tensor,  # [B, P, d_model]
        slot_idx: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            prompt_hidden_states: Context representations for prompt tokens [B, P, D]
            slot_idx: Which argument slot is being grounded (0 to max_spans - 1)
            
        Returns:
            start_logits: [B, P]
            end_logits: [B, P]
            type_logits: [B, num_types]
            confidence: [B, 1] in (0, 1)
        """
        B, P, D = prompt_hidden_states.shape
        slot_emb = self.slot_queries.weight[slot_idx].view(1, 1, D).expand(B, P, D)

        # 1. Start Logits
        start_feat = torch.cat([prompt_hidden_states, slot_emb], dim=-1)  # [B, P, 2*D]
        start_logits = self.W_start(start_feat).squeeze(-1)              # [B, P]

        # Softmax start weights to condition end prediction
        start_probs = F.softmax(start_logits, dim=-1).unsqueeze(-1)      # [B, P, 1]
        start_ctx = torch.sum(start_probs * prompt_hidden_states, dim=1, keepdim=True).expand(B, P, D)

        # 2. End Logits conditioned on predicted start representation
        end_feat = torch.cat([prompt_hidden_states, slot_emb, start_ctx], dim=-1)  # [B, P, 3*D]
        end_logits = self.W_end(end_feat).squeeze(-1)                              # [B, P]

        # 3. Overall Span Representation for Type and Confidence
        end_probs = F.softmax(end_logits, dim=-1).unsqueeze(-1)                    # [B, P, 1]
        end_ctx = torch.sum(end_probs * prompt_hidden_states, dim=1)              # [B, D]
        start_summary = torch.sum(start_probs * prompt_hidden_states, dim=1)       # [B, D]

        span_rep = torch.cat([start_summary, end_ctx], dim=-1)                     # [B, 2*D]
        type_logits = self.type_classifier(span_rep)                               # [B, num_types]
        confidence = torch.sigmoid(self.conf_head(span_rep))                       # [B, 1]

        return start_logits, end_logits, type_logits, confidence


class ChakrSpanGroundingModel(nn.Module):
    """
    ChakrMicro model extended with a neural Span Grounding Head.
    """

    def __init__(self, config: Optional[ModelConfig] = None, max_slots: int = 4):
        super().__init__()
        if config is None:
            config = ModelConfig()
        self.config = config
        self.core = ChakrMicro(config)
        self.span_head = SpanGroundingHead(config.d_model, max_spans=max_slots)

    def forward(
        self,
        input_ids: torch.Tensor,
        prompt_len: int,
        slot_idx: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass returning language modeling logits + span grounding predictions for slot_idx.
        """
        B, T = input_ids.shape
        vocab_logits, h = self.core(input_ids, return_hidden_states=True)
        prompt_states = h[:, :prompt_len, :]

        start_logits, end_logits, type_logits, conf = self.span_head(prompt_states, slot_idx=slot_idx)
        return vocab_logits, start_logits, end_logits, type_logits, conf

    def predict_span(
        self,
        input_ids: torch.Tensor,
        prompt_len: int,
        slot_idx: int = 0,
        tokenizer: Optional[Any] = None,
        raw_prompt_text: Optional[str] = None,
    ) -> ExtractedSpan:
        """
        Inference method to extract the span for a specific slot.
        """
        self.eval()
        with torch.no_grad():
            _, start_logits, end_logits, type_logits, conf = self.forward(input_ids, prompt_len, slot_idx)

            start_p = F.softmax(start_logits[0], dim=-1)
            end_p = F.softmax(end_logits[0], dim=-1)

            # Mask out invalid spans where end < start
            P = prompt_len
            span_scores = torch.outer(start_p, end_p)
            valid_mask = torch.triu(torch.ones(P, P, device=input_ids.device), diagonal=0)
            masked_scores = span_scores * valid_mask

            best_idx = torch.argmax(masked_scores).item()
            s_idx = best_idx // P
            e_idx = best_idx % P

            pred_type_id = torch.argmax(type_logits[0]).item()
            pred_type = SPAN_TYPES[pred_type_id]
            confidence_val = float(conf[0, 0].item())

            span_text = ""
            if tokenizer is not None:
                token_slice = input_ids[0, s_idx : e_idx + 1].tolist()
                try:
                    span_text = tokenizer.decode(token_slice, errors="replace").strip()
                except TypeError:
                    span_text = tokenizer.decode(token_slice).strip()
                except Exception:
                    span_text = ""

            return ExtractedSpan(
                start_idx=s_idx,
                end_idx=e_idx,
                span_type=pred_type,
                confidence=confidence_val,
                span_text=span_text,
            )
