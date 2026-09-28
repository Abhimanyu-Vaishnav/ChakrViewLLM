"""
Cognitive Decision Layer for ChakrView (Step 25).

Evaluates reasoning traces, critical thinking challenges, evidence quality,
and contradiction states to deterministically decide the cognitive outcome state:
    ANSWER
    ANSWER_WITH_UNCERTAINTY
    NEED_CLARIFICATION
    INSUFFICIENT_INFORMATION
    REQUIRE_VERIFICATION
    REVISION_REQUIRED
    CAPABILITY_REQUIRED
    SAFE_STOP

CRITICAL AXIOMS:
1. No Fabricated Certainty: If evidence is absent or contradictions remain unresolved,
   explicitly declares INSUFFICIENT_INFORMATION or ANSWER_WITH_UNCERTAINTY.
2. Capability Boundary: If external environment execution is required, marks CAPABILITY_REQUIRED
   for formal CapabilityGate authorization.
"""

from typing import Dict, List, Optional, Any, Tuple
from chakrview.cognition.unified.models import DecisionState, UnifiedCognitiveState


class CognitiveDecisionLayer:
    """
    Deterministic rule-based decision arbiter evaluating multi-stage cognitive outputs.
    """

    def __init__(
        self,
        min_confidence_for_answer: float = 0.70,
        min_confidence_for_uncertainty: float = 0.40,
    ) -> None:
        self.min_confidence_for_answer = min_confidence_for_answer
        self.min_confidence_for_uncertainty = min_confidence_for_uncertainty

    def decide(
        self,
        state: UnifiedCognitiveState,
        candidate_response: str,
        reasoning_success: bool = True,
        critique_verdict: Optional[str] = None,
        capability_needed: bool = False,
        safety_violation: bool = False,
    ) -> Tuple[DecisionState, float, Optional[str]]:
        """
        Evaluate cognitive state and determine decision state, confidence, and notes.
        Returns: (DecisionState, confidence, uncertainty_notes)
        """
        # 1. Safety or Invariant Hazard Check
        if safety_violation:
            return (DecisionState.SAFE_STOP, 0.0, "Stopped due to safety constraint or invariant boundary violation.")

        # 2. Capability Requirement Check
        if capability_needed:
            return (DecisionState.CAPABILITY_REQUIRED, 0.85, "External capability execution required.")

        # 3. Revision Check
        if critique_verdict in ("REVISE", "FAIL") and state.revision_count < 2:
            return (DecisionState.REVISION_REQUIRED, 0.50, f"Critique verdict '{critique_verdict}' requires revision.")

        # 4. Check for Unresolved Contradictions
        unresolved_contradictions = [
            c for c in state.contradictions
            if c.get("resolution_state") in ("UNRESOLVED", "PERSISTENT_CONFLICT")
        ]
        if unresolved_contradictions:
            c_desc = f"Unresolved contradiction across {len(unresolved_contradictions)} conflicting records."
            return (DecisionState.ANSWER_WITH_UNCERTAINTY, 0.50, c_desc)

        # 5. Check Evidence Availability
        # Filter out user prompt assertion to check for actual corroborating evidence or memory
        corroborating_evidence = [
            e for e in state.evidence
            if e.get("evidence_type") != "USER_ASSERTION" and e.get("source_id") != "prompt"
        ]
        if not corroborating_evidence and not state.retrieved_memories:
            words = state.user_prompt.split()
            if len(words) > 4 and any(w.lower().rstrip("?:;,.") in ("what", "why", "how", "who", "when", "where", "which", "verify", "calculate", "prove") for w in words):
                return (
                    DecisionState.INSUFFICIENT_INFORMATION,
                    0.25,
                    "No verified evidence or relevant memory was retrieved to support an authoritative answer.",
                )

        # 6. Evaluate Confidence
        base_confidence = state.confidence
        if not reasoning_success:
            base_confidence = min(base_confidence, 0.45)

        if base_confidence >= self.min_confidence_for_answer:
            return (DecisionState.ANSWER, base_confidence, None)
        elif base_confidence >= self.min_confidence_for_uncertainty:
            return (DecisionState.ANSWER_WITH_UNCERTAINTY, base_confidence, "Answer subject to epistemic uncertainty.")
        else:
            return (
                DecisionState.NEED_CLARIFICATION,
                base_confidence,
                "Confidence is below threshold; further input or clarification needed.",
            )
