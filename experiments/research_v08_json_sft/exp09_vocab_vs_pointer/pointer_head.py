"""
pointer_head.py
ChakrView Experiment 09 — Arm B: Isolated Pointer-Head Prototype

Architecture:
- ChakrMicro backbone: processes prompt input sequence tokens [0] + prompt_toks.
- Pointer Scoring Head:
  Takes hidden state representation at the final prompt position (query)
  and hidden states across all candidate source positions (keys).
  Computes score: Score_i = (W_q * h_last)^T (W_k * h_i) / sqrt(d)
  Masks out ineligible non-candidate positions with -infinity.
  Applies Softmax over eligible candidate source positions.
- Execution / Extraction:
  Directly copies the token ID at argmax_i(Score_i) from the input tensor.
  Bypasses the 4,096-vocabulary projection matrix entirely.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro

class ChakrPointerModel(nn.Module):
    def __init__(self, config: ModelConfig, freeze_backbone: bool = False):
        super().__init__()
        self.config = config
        self.backbone = ChakrMicro(config)
        self.d_model = config.d_model  # 192 in ChakrMicro

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # Small 2-layer MLP projection for query and key pointer scoring
        self.query_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.key_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.scale = 1.0 / math.sqrt(self.d_model)

    def extract_hidden_states(self, idx: torch.Tensor) -> torch.Tensor:
        """
        Runs input through ChakrMicro, returning the final normalized hidden states: [B, T, d_model].
        """
        _, h = self.backbone(idx, return_hidden_states=True)
        return h

    def forward(
        self,
        input_ids: torch.Tensor,
        candidate_masks: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        input_ids: [B, T]
        candidate_masks: [B, T] (bool: True for eligible positions, False for ineligible)
        Returns: logits over positions [B, T]
        """
        h = self.extract_hidden_states(input_ids)  # [B, T, d_model]
        
        # Query is the hidden state at the final prompt position (-1)
        q = self.query_proj(h[:, -1:, :])  # [B, 1, d_model]
        # Keys are the hidden states at all positions
        k = self.key_proj(h)               # [B, T, d_model]

        # Dot-product attention scores
        scores = torch.bmm(q, k.transpose(1, 2)).squeeze(1) * self.scale  # [B, T]

        if candidate_masks is not None:
            scores = scores.masked_fill(~candidate_masks, float("-inf"))

        return scores

    def point_and_copy(
        self,
        input_ids: torch.Tensor,
        candidate_masks: Optional[torch.Tensor] = None
    ) -> Tuple[int, int, float]:
        """
        Inference method:
        Returns: (selected_position, copied_token_id, confidence)
        """
        self.eval()
        with torch.no_grad():
            scores = self.forward(input_ids, candidate_masks)
            probs = F.softmax(scores, dim=-1)
            best_pos = int(torch.argmax(scores, dim=-1)[0].item())
            copied_token_id = int(input_ids[0, best_pos].item())
            conf = float(probs[0, best_pos].item())
        return best_pos, copied_token_id, conf
