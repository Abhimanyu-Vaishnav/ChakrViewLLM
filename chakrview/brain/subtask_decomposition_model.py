"""
Batches 717-722: Neural Subtask Decomposition & Dependency Prediction Head.
Location: chakrview/brain/subtask_decomposition_model.py

Features:
1. ChakrSubtaskDecompositionModel:
   - Takes ChakrMicro representations h: [B, T, 192].
   - Multi-Task Plan Prediction Heads:
     a) Goal Head: Predicts goal type (compound_arithmetic, text_transform_chain, etc.).
     b) Subtask Action Heads (Step 1, Step 2): Predicts action type for each step.
     c) Subtask Tool Heads (Step 1, Step 2): Predicts target tool for each step.
     d) Dependency Edge Head: Predicts whether Step 2 depends on Step 1 (binary classification).
     e) Clarification Head: Predicts if goal is missing information.
2. Lightweight, stable, zero leakage:
   - Does not overwrite base weights.
   - Preserves causal masking and sequence safety.
"""
from __future__ import annotations

from typing import Any, Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from chakrview.brain.model import ChakrMicro


GOAL_TYPES = [
    "compound_arithmetic",
    "text_transform_chain",
    "evidence_reasoning",
    "clarification_needed",
    "unsupported",
]
GOAL_TO_ID = {name: idx for idx, name in enumerate(GOAL_TYPES)}
ID_TO_GOAL = {idx: name for idx, name in enumerate(GOAL_TYPES)}

ACTION_TYPES = ["tool_call", "text_transform", "clarification", "none"]
ACTION_TO_ID = {name: idx for idx, name in enumerate(ACTION_TYPES)}
ID_TO_ACTION = {idx: name for idx, name in enumerate(ACTION_TYPES)}

TOOL_TYPES = ["calculator", "text_utility", "none"]
TOOL_TO_ID = {name: idx for idx, name in enumerate(TOOL_TYPES)}
ID_TO_TOOL = {idx: name for idx, name in enumerate(TOOL_TYPES)}


class ChakrSubtaskDecompositionModel(nn.Module):
    """
    Joint Neural Goal, Subtask Sequence, and Dependency Edge Prediction Head.
    """

    def __init__(
        self,
        base_model: ChakrMicro,
        num_goals: int = len(GOAL_TYPES),
        num_actions: int = len(ACTION_TYPES),
        num_tools: int = len(TOOL_TYPES),
    ) -> None:
        super().__init__()
        self.base_model = base_model
        d_model = base_model.config.d_model  # 192

        # 1. Goal Classification Head
        self.goal_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, num_goals),
        )

        # 2. Subtask 1 Action & Tool Heads
        self.st1_action_head = nn.Linear(d_model, num_actions)
        self.st1_tool_head = nn.Linear(d_model, num_tools)

        # 3. Subtask 2 Action & Tool Heads
        self.st2_action_head = nn.Linear(d_model, num_actions)
        self.st2_tool_head = nn.Linear(d_model, num_tools)

        # 4. Dependency Head (Does Subtask 2 depend on Subtask 1?)
        self.dependency_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

        # 5. Clarification Need Head
        self.clarification_head = nn.Linear(d_model, 1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        # Forward pass through base model with return_hidden_states=True
        _, h = self.base_model(input_ids, attention_mask=attention_mask, return_hidden_states=True)

        # Sequence pooling: use final token representation
        seq_rep = h[:, -1, :]  # [B, d_model]

        goal_logits = self.goal_head(seq_rep)
        st1_act_logits = self.st1_action_head(seq_rep)
        st1_tool_logits = self.st1_tool_head(seq_rep)

        st2_act_logits = self.st2_action_head(seq_rep)
        st2_tool_logits = self.st2_tool_head(seq_rep)

        dep_logits = self.dependency_head(seq_rep).squeeze(-1)  # [B]
        clar_logits = self.clarification_head(seq_rep).squeeze(-1)  # [B]

        return {
            "goal_logits": goal_logits,
            "st1_act_logits": st1_act_logits,
            "st1_tool_logits": st1_tool_logits,
            "st2_act_logits": st2_act_logits,
            "st2_tool_logits": st2_tool_logits,
            "dependency_logits": dep_logits,
            "clarification_logits": clar_logits,
        }
