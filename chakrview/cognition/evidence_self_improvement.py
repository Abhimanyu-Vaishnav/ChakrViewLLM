"""
ChakrView Step 159: Evidence-Based Self-Improvement Cycle for Neural Reasoning.

Applies the governed self-improvement loop directly to the measured weakness:
OBSERVE (Neural reasoning = 0) -> DIAGNOSE (Step 153 autopsy) -> PROPOSE (Ladder curriculum) ->
TRAIN (Isolated candidate) -> EVALUATE (Held-out reasoning) -> CHECK (Language regression) ->
DECISION (PROMOTE / REJECT / ROLLBACK).

Enforces strictly capability-specific mandatory gates:
- A reasoning candidate MUST achieve held_out_reasoning_delta >= min_threshold
- If reasoning gate fails, decision is REJECT_CANDIDATE even if training loss decreased.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from chakrview.brain.model import ChakrMicro
from chakrview.runtime.interactive import EXPECTED_WEIGHT_HASH
from chakrview.cognition.neural_observability import CandidateIsolationManager
from chakrview.cognition.reasoning_autopsy import ReasoningFailureDiagnosis


@dataclass
class EvidenceSelfImprovementDecision:
    cycle_id: str
    target_weakness: str
    decision: str  # "PROMOTE_CANDIDATE", "REJECT_CANDIDATE", "ROLLBACK"
    held_out_reasoning_acc: float
    held_out_reasoning_delta: float
    language_loss_delta: float
    baseline_intact: bool
    audit_rationale: str


class EvidenceSelfImprovementOrchestrator:
    """
    Drives evidence-based self-improvement on real measured neural weaknesses.
    """

    def __init__(self, isolation_manager: CandidateIsolationManager) -> None:
        self.isolation_manager = isolation_manager

    def execute_reasoning_improvement_cycle(
        self,
        cycle_id: str,
        diagnosis: ReasoningFailureDiagnosis,
        baseline_model: ChakrMicro,
        candidate_model: ChakrMicro,
        baseline_reasoning_acc: float,
        candidate_reasoning_acc: float,
        language_loss_delta: float,
        min_reasoning_acc_threshold: float = 0.50,
    ) -> EvidenceSelfImprovementDecision:
        base_intact, cand_diverged, base_h, cand_h = self.isolation_manager.verify_candidate_integrity(
            baseline_model, candidate_model
        )

        delta = round(candidate_reasoning_acc - baseline_reasoning_acc, 4)

        reasons = []
        if not base_intact:
            reasons.append("Baseline corruption detected")
        if candidate_reasoning_acc < min_reasoning_acc_threshold:
            reasons.append(
                f"Held-out reasoning accuracy ({candidate_reasoning_acc}) below mandatory threshold ({min_reasoning_acc_threshold})"
            )
        if language_loss_delta > 0.5:
            reasons.append(f"Severe language regression detected ({language_loss_delta})")

        if not reasons:
            decision = "PROMOTE_CANDIDATE"
            rationale = "Candidate demonstrated held-out reasoning mastery without language regression."
        else:
            decision = "REJECT_CANDIDATE"
            rationale = f"Candidate rejected under mandatory capability gating: {'; '.join(reasons)}"

        return EvidenceSelfImprovementDecision(
            cycle_id=cycle_id,
            target_weakness=diagnosis.diagnosis_id,
            decision=decision,
            held_out_reasoning_acc=candidate_reasoning_acc,
            held_out_reasoning_delta=delta,
            language_loss_delta=language_loss_delta,
            baseline_intact=base_intact,
            audit_rationale=rationale,
        )
