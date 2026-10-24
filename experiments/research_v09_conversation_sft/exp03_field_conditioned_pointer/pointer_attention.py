"""
pointer_attention.py
ChakrView Experiment 15 — Field-Conditioned Pointer Attention Architecture.

Decouples the queried semantic slot from the general causal prefix by injecting an explicit
field-query embedding into dual factorized start and end pointer projections.
Operates on top of the frozen ChakrMicro 3.44M causal LM backbone.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro

class FieldConditionedSpanPointer(nn.Module):
    """
    Field-Conditioned Dual-Boundary Span Pointer Model.
    """
    def __init__(
        self,
        config: ModelConfig,
        num_fields: int = 10,
        max_span_len: int = 12,
        freeze_backbone: bool = True
    ):
        super().__init__()
        self.config = config
        self.d_model = config.d_model  # 192
        self.num_fields = num_fields
        self.max_span_len = max_span_len
        
        # 1. Native Causal LM Backbone (Frozen)
        self.backbone = ChakrMicro(config)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
                
        # 2. Field Query Embeddings
        self.field_embedding = nn.Embedding(num_fields, self.d_model)
        
        # 3. Factorized Pointer Query and Key Projections (Trainable)
        self.query_start = nn.Linear(self.d_model, self.d_model, bias=False)
        self.key_start = nn.Linear(self.d_model, self.d_model, bias=False)
        
        self.query_end = nn.Linear(self.d_model, self.d_model, bias=False)
        self.key_end = nn.Linear(self.d_model, self.d_model, bias=False)
        
        self.scale = 1.0 / math.sqrt(self.d_model)

    def extract_hidden_states(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Runs the frozen backbone to obtain contextual representations."""
        # Backbone returns logits, hidden_states if return_hidden_states=True
        _, h = self.backbone(input_ids, return_hidden_states=True)
        return h

    def forward(
        self,
        input_ids: torch.Tensor,
        field_ids: torch.Tensor,
        candidate_mask: Optional[torch.Tensor] = None,
        use_field_conditioning: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        input_ids: [B, T]
        field_ids: [B] (integer field slot ID)
        candidate_mask: [B, T] (bool: True for eligible positions)
        use_field_conditioning: bool (False for query-agnostic ablation)
        Returns:
            start_logits: [B, T]
            end_logits: [B, T]
        """
        B, T = input_ids.shape
        h = self.extract_hidden_states(input_ids)  # [B, T, d_model]
        
        if use_field_conditioning:
            # Query vector is the explicit field slot embedding
            query = self.field_embedding(field_ids).unsqueeze(1)  # [B, 1, d_model]
        else:
            # Query-agnostic ablation: uses final prefix token hidden state
            query = h[:, -1:, :]  # [B, 1, d_model]
            
        # 1. Start Pointer Scoring
        q_s = self.query_start(query)
        k_s = self.key_start(h)
        start_logits = torch.bmm(q_s, k_s.transpose(1, 2)).squeeze(1) * self.scale  # [B, T]
        
        # 2. End Pointer Scoring
        q_e = self.query_end(query)
        k_e = self.key_end(h)
        end_logits = torch.bmm(q_e, k_e.transpose(1, 2)).squeeze(1) * self.scale      # [B, T]
        
        if candidate_mask is not None:
            # Mask out forbidden positions with -1e9
            start_logits = start_logits.masked_fill(~candidate_mask, -1e9)
            end_logits = end_logits.masked_fill(~candidate_mask, -1e9)
            
        return start_logits, end_logits

    def predict_span(
        self,
        input_ids: torch.Tensor,
        field_id: int,
        candidate_mask: Optional[torch.Tensor] = None,
        use_field_conditioning: bool = True
    ) -> Tuple[int, int, List[int], float]:
        """
        Constrained joint span decoding:
        Finds s, e maximizing start_logits[s] + end_logits[e]
        subject to:
          - s <= e <= s + max_span_len (or s == e == 0 for ABSTAIN)
          - s, e are candidate_mask eligible
        Returns: (start_idx, end_idx, extracted_token_ids, joint_confidence)
        """
        self.eval()
        with torch.no_grad():
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)
            if candidate_mask is not None and candidate_mask.dim() == 1:
                candidate_mask = candidate_mask.unsqueeze(0)
            f_tensor = torch.tensor([field_id], dtype=torch.long, device=input_ids.device)
            start_logits, end_logits = self.forward(
                input_ids, f_tensor, candidate_mask, use_field_conditioning=use_field_conditioning
            )
            
            s_probs = F.softmax(start_logits[0], dim=-1)
            e_probs = F.softmax(end_logits[0], dim=-1)
            
            T = input_ids.shape[1]
            best_score = float("-inf")
            best_s, best_e = 0, 0
            
            # Position 0 is the dedicated ABSTAIN position
            if candidate_mask is None or candidate_mask[0, 0]:
                abstain_score = start_logits[0, 0].item() + end_logits[0, 0].item()
                best_score = abstain_score
                best_s, best_e = 0, 0
                
            for s in range(1, T):
                if candidate_mask is not None and not candidate_mask[0, s]:
                    continue
                for e in range(s, min(s + self.max_span_len, T)):
                    if candidate_mask is not None and not candidate_mask[0, e]:
                        continue
                    score = start_logits[0, s].item() + end_logits[0, e].item()
                    if score > best_score:
                        best_score = score
                        best_s = s
                        best_e = e
                        
            if best_s == 0 and best_e == 0:
                extracted_tokens = []  # ABSTAIN / MISSING
            else:
                extracted_tokens = input_ids[0, best_s:best_e + 1].tolist()
                
            conf = float((s_probs[best_s] * e_probs[best_e]).item())
            
        return best_s, best_e, extracted_tokens, conf


def build_field_conditioned_pointer(
    checkpoint_path: Optional[str | Path] = None,
    num_fields: int = 10,
    device: str = "cpu"
) -> FieldConditionedSpanPointer:
    """Helper to build FieldConditionedSpanPointer with canonical checkpoint weights."""
    from pathlib import Path
    REPO_ROOT = Path(__file__).resolve().parents[3]
    CANONICAL_CKPT = REPO_ROOT / "experiments" / "batch16_21_combined" / "main_training_10000" / "training_checkpoint_step_10000.pt"
    
    config = ModelConfig()
    model = FieldConditionedSpanPointer(config, num_fields=num_fields, freeze_backbone=True)
    
    # Load canonical backbone weights
    ckpt_file = checkpoint_path or CANONICAL_CKPT
    ckpt = torch.load(ckpt_file, map_location="cpu", weights_only=False)
    if "model_state_dict" in ckpt:
        model.backbone.load_state_dict(ckpt["model_state_dict"], strict=True)
    else:
        model.backbone.load_state_dict(ckpt, strict=True)
        
    model.to(device)
    return model
