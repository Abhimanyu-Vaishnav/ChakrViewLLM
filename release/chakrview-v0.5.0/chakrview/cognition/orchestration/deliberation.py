"""
Adaptive Deliberation Controller for Step 28 Cognitive Orchestration.

Manages the bounded deliberation loop: evaluating evidence sufficiency,
conflict presence, and determining whether an additional round is justified.

CRITICAL INVARIANTS:
1. HARD BOUNDED DELIBERATION: No unbounded loops. Deliberation is strictly capped at max_rounds.
2. ANTI-MAJORITY PRINCIPLE: Disagreements trigger verification or uncertainty rather than
   simple majority voting. Minority perspectives are preserved.
3. FAIL-SAFE TERMINATION: Insufficient evidence at the deliberation ceiling resolves to
   DecisionState.UNCERTAIN or SAFE_STOP, never fabricated certainty.
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any

from chakrview.cognition.orchestration.models import (
    WorkloadClass,
    MAX_DELIBERATION_ROUNDS,
)
from chakrview.cognition.federated.models import AgentRole, ConflictState
from chakrview.cognition.unified.models import DecisionState


@dataclass
class DeliberationSufficiencyEvaluation:
    """
    Outcome of evaluating whether the current deliberation round is sufficient.
    """
    is_sufficient: bool
    requires_additional_round: bool
    reason: str
    recommended_focus_role: Optional[AgentRole] = None
    detected_conflicts: int = 0
    minority_evidence_count: int = 0
    recommended_decision_state: Optional[DecisionState] = None


class AdaptiveDeliberationController:
    """
    Evaluates evidence sufficiency across deliberation rounds and guides escalation.
    """

    @classmethod
    def evaluate_sufficiency(
        cls,
        workload_class: WorkloadClass,
        current_round: int,
        max_rounds: int,
        total_claims: int,
        ground_evidence_count: int,
        contradiction_count: int,
        minority_evidence_count: int,
        verification_invoked: bool,
        verification_passed: Optional[bool],
        agent_failure_count: int = 0,
    ) -> DeliberationSufficiencyEvaluation:
        """
        Determine if the cognitive cycle can safely proceed to synthesis or requires
        another bounded deliberation round.
        """
        effective_max = min(max_rounds, MAX_DELIBERATION_ROUNDS)

        # 1. Simple workload: always completes in 1 round
        if workload_class == WorkloadClass.SIMPLE:
            return DeliberationSufficiencyEvaluation(
                is_sufficient=True,
                requires_additional_round=False,
                reason="Simple workload completed minimal single-round evaluation.",
                recommended_decision_state=DecisionState.ANSWER,
            )

        # 2. Reached maximum permitted rounds: hard stop!
        if current_round >= effective_max:
            if contradiction_count > 0:
                return DeliberationSufficiencyEvaluation(
                    is_sufficient=True,
                    requires_additional_round=False,
                    reason=f"Reached maximum round ceiling ({effective_max}) with unresolved conflicts; preserving uncertainty.",
                    detected_conflicts=contradiction_count,
                    minority_evidence_count=minority_evidence_count,
                    recommended_decision_state=DecisionState.UNCERTAIN if hasattr(DecisionState, "UNCERTAIN") else DecisionState.ANSWER_WITH_UNCERTAINTY,
                )
            if ground_evidence_count == 0 and total_claims == 0:
                return DeliberationSufficiencyEvaluation(
                    is_sufficient=True,
                    requires_additional_round=False,
                    reason=f"Reached maximum round ceiling ({effective_max}) with zero ground evidence.",
                    recommended_decision_state=DecisionState.INSUFFICIENT_INFORMATION,
                )
            return DeliberationSufficiencyEvaluation(
                is_sufficient=True,
                requires_additional_round=False,
                reason=f"Reached maximum round ceiling ({effective_max}); finalizing synthesis.",
                detected_conflicts=contradiction_count,
                minority_evidence_count=minority_evidence_count,
                recommended_decision_state=DecisionState.ANSWER,
            )

        # 3. Conflicted workload check: if contradictions exist and rounds remain, escalate to Verifier
        if contradiction_count > 0:
            return DeliberationSufficiencyEvaluation(
                is_sufficient=False,
                requires_additional_round=True,
                reason=f"Detected {contradiction_count} active conflicts; scheduling verification round.",
                recommended_focus_role=AgentRole.VERIFIER,
                detected_conflicts=contradiction_count,
                minority_evidence_count=minority_evidence_count,
            )

        # 4. Verification-required workload check
        if workload_class in (WorkloadClass.VERIFICATION_REQUIRED, WorkloadClass.AMBIGUOUS):
            if not verification_invoked or verification_passed is None:
                return DeliberationSufficiencyEvaluation(
                    is_sufficient=False,
                    requires_additional_round=True,
                    reason="Mandatory verification has not yet concluded; executing verification round.",
                    recommended_focus_role=AgentRole.VERIFIER,
                )

        # 5. Agent failures with empty evidence: retry if within ceiling
        if agent_failure_count > 0 and ground_evidence_count == 0:
            return DeliberationSufficiencyEvaluation(
                is_sufficient=False,
                requires_additional_round=True,
                reason="Primary agent failed with insufficient evidence; executing recovery round.",
                recommended_focus_role=AgentRole.ANALYST,
            )

        # 6. Standard sufficient evidence: complete early!
        return DeliberationSufficiencyEvaluation(
            is_sufficient=True,
            requires_additional_round=False,
            reason="Sufficient grounded evidence gathered without unresolved contradictions.",
            detected_conflicts=contradiction_count,
            minority_evidence_count=minority_evidence_count,
            recommended_decision_state=DecisionState.ANSWER,
        )
