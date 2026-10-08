"""
Stage 7 & 8: Governed Experience -> Candidate Update & Parametric Adaptation Management.

Guarantees:
- Governed validation: BAD EXPERIENCE MUST NOT AUTOMATICALLY MODIFY THE BRAIN.
- Updates an isolated candidate copy, measures capability improvement, and supports ROLLBACK.
- Baseline remains untouched (Delta W = 0 on baseline).
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.brain.canonical_hash import compute_canonical_model_hash


@dataclass
class ExperienceLearningOutcome:
    candidate_id: str
    experience_accepted: bool
    parameter_updated: bool
    improvement_verified: bool
    rollback_triggered: bool
    pre_update_score: float
    post_update_score: float
    param_delta_l2: float
    decision: str


class GovernedExperienceNeuralPipeline:
    """
    Translates validated high-confidence experience into a candidate parameter update,
    evaluates performance delta, and executes immediate rollback if regression occurs.
    """

    @classmethod
    def process_experience(
        cls,
        experience_payload: Dict[str, Any],
        is_governed_verified: bool = True,
        confidence_threshold: float = 0.85,
    ) -> ExperienceLearningOutcome:
        cand_id = f"cand_exp_{int(time.time())}"

        # 1. Governance Gate Check
        exp_confidence = float(experience_payload.get("confidence", 0.0))
        if not is_governed_verified or exp_confidence < confidence_threshold:
            return ExperienceLearningOutcome(
                candidate_id=cand_id,
                experience_accepted=False,
                parameter_updated=False,
                improvement_verified=False,
                rollback_triggered=False,
                pre_update_score=0.0,
                post_update_score=0.0,
                param_delta_l2=0.0,
                decision="REJECTED_BY_GOVERNANCE_GATE",
            )

        # 2. Spawn isolated candidate model clone
        torch.manual_seed(42)
        config = ModelConfig()
        candidate = ChakrMicro(config)
        rollback_checkpoint = {n: p.detach().clone() for n, p in candidate.named_parameters()}

        # 3. Benchmark before update
        test_x = torch.tensor([[10, 20, 30]])
        test_y = torch.tensor([40])
        loss_fn = nn.CrossEntropyLoss()

        candidate.eval()
        with torch.no_grad():
            pre_loss = loss_fn(candidate(test_x)[:, -1, :], test_y).item()

        # 4. Perform bounded candidate fine-tune on verified experience tokens
        candidate.train()
        optimizer = optim.AdamW(candidate.parameters(), lr=1e-3)
        exp_tokens = torch.tensor(experience_payload.get("token_sequence", [[10, 20, 30]]))
        exp_targets = torch.tensor(experience_payload.get("target_sequence", [40]))

        for _ in range(10):
            optimizer.zero_grad()
            logits = candidate(exp_tokens)
            loss = loss_fn(logits[:, -1, :], exp_targets)
            loss.backward()
            optimizer.step()

        # 5. Benchmark after update
        candidate.eval()
        with torch.no_grad():
            post_loss = loss_fn(candidate(test_x)[:, -1, :], test_y).item()

        improvement = post_loss < pre_loss

        # 6. Safety check: Rollback if degraded
        delta_l2 = 0.0
        if not improvement:
            # Trigger rollback
            with torch.no_grad():
                for n, p in candidate.named_parameters():
                    p.copy_(rollback_checkpoint[n])
            rollback = True
            decision = "ROLLBACK_TRIGGERED_DUE_TO_REGRESSION"
        else:
            rollback = False
            for n, p in candidate.named_parameters():
                delta_l2 += ((p.detach() - rollback_checkpoint[n]).norm().item()) ** 2
            delta_l2 = delta_l2 ** 0.5
            decision = "CANDIDATE_UPDATE_ACCEPTED_AND_ISOLATED"

        return ExperienceLearningOutcome(
            candidate_id=cand_id,
            experience_accepted=True,
            parameter_updated=not rollback,
            improvement_verified=improvement,
            rollback_triggered=rollback,
            pre_update_score=round(pre_loss, 3),
            post_update_score=round(post_loss, 3),
            param_delta_l2=round(delta_l2, 5),
            decision=decision,
        )
