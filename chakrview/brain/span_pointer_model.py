"""
Batches 596-609: Neural Span Extraction & Pointer Head Architecture.
Location: chakrview/brain/span_pointer_model.py

Features:
1. ChakrSpanPointerModel:
   - Takes ChakrMicro neural core representations h: [B, T, 192].
   - Multi-Task Prediction:
     a) Intent Classifier Head: Linear(d_model, num_intents=7). Classifies user intent over sequence representation.
     b) Span Boundary Head: Linear(d_model, 2). Predicts start_logit and end_logit for each token position T.
     c) Has-Span Gate: Linear(d_model, 1). Predicts whether an argument span exists in this turn.
2. Zero Disturbance to Base Model:
   - Base weights can be loaded directly from canonical or research checkpoints.
   - Preserves causal masking and sequence safety.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from chakrview.brain.model import ChakrMicro
from chakrview.brain.config import ModelConfig


INTENT_LIST = [
    "calculator",
    "text_utility",
    "concept_definition",
    "knowledge_lookup",
    "direct_chat",
    "clarification_needed",
    "unsupported",
]
INTENT_TO_ID = {name: idx for idx, name in enumerate(INTENT_LIST)}
ID_TO_INTENT = {idx: name for idx, name in enumerate(INTENT_LIST)}


class ChakrSpanPointerModel(nn.Module):
    """
    Joint Neural Intent and Token-Span Pointer Architecture for ChakrView.
    """

    def __init__(
        self,
        base_model: ChakrMicro,
        num_intents: int = len(INTENT_LIST),
    ) -> None:
        super().__init__()
        self.base_model = base_model
        d_model = base_model.config.d_model  # 192

        # 1. Intent Classification Head
        self.intent_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, num_intents),
        )

        # 2. Span Boundary Heads (Start and End logits per token)
        self.span_start_head = nn.Linear(d_model, 1)
        self.span_end_head = nn.Linear(d_model, 1)

        # 3. Has-Span Gate (distinguishes span-requiring queries from direct chat / unsupported)
        self.has_span_gate = nn.Linear(d_model, 1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            input_ids: [B, T]
            attention_mask: [B, T] (optional)
        Returns:
            dict containing:
                intent_logits: [B, num_intents]
                start_logits: [B, T]
                end_logits: [B, T]
                has_span_logits: [B, 1]
        """
        # Forward pass through base model up to final pre-norm hidden states
        # Shape: [B, T, d_model]
        _, h = self.base_model(input_ids, attention_mask=attention_mask, return_hidden_states=True)

        # Sequence-level representation: use representation at last token position (before padding)
        # In our prompts 'User: ...\nIntent:', the last prompt token contains causal summary
        seq_rep = h[:, -1, :]  # [B, d_model]

        # 1. Intent logits
        intent_logits = self.intent_head(seq_rep)  # [B, num_intents]

        # 2. Span boundary logits over sequence
        start_logits = self.span_start_head(h).squeeze(-1)  # [B, T]
        end_logits = self.span_end_head(h).squeeze(-1)      # [B, T]

        # 3. Has-span gate
        has_span_logits = self.has_span_gate(seq_rep)  # [B, 1]

        return {
            "intent_logits": intent_logits,
            "start_logits": start_logits,
            "end_logits": end_logits,
            "has_span_logits": has_span_logits,
            "hidden_states": h,
        }

    def predict_span(
        self,
        input_ids: torch.Tensor,
        min_start_tok: int = 1,
    ) -> Tuple[str, Optional[Tuple[int, int]], float]:
        """
        Inference utility: returns (predicted_intent, (start_tok, end_tok), confidence)
        """
        self.eval()
        with torch.no_grad():
            out = self.forward(input_ids)
            intent_id = int(torch.argmax(out["intent_logits"][0]).item())
            intent_name = ID_TO_INTENT.get(intent_id, "unsupported")

            has_span = torch.sigmoid(out["has_span_logits"][0, 0]).item() > 0.5
            span = None
            if has_span and intent_name in ("calculator", "text_utility", "concept_definition", "knowledge_lookup"):
                T = input_ids.shape[1]
                st_logits = out["start_logits"][0].clone()
                end_logits = out["end_logits"][0].clone()

                # Mask out tokens before query begins or past end
                if min_start_tok > 0:
                    st_logits[:min_start_tok] = -1e9
                    end_logits[:min_start_tok] = -1e9

                best_st = int(torch.argmax(st_logits).item())
                # Constrain end >= start
                end_logits[:best_st] = -1e9
                best_end = int(torch.argmax(end_logits).item())
                span = (best_st, max(best_st + 1, best_end + 1))

            conf = float(torch.softmax(out["intent_logits"][0], dim=-1)[intent_id].item())
            return intent_name, span, conf
