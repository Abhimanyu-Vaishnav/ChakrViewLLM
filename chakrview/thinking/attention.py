"""
Deliberation Attention & Prioritization for ChakrView (Step 21).

Decides which item, question, hypothesis, or contradiction in the workspace
deserves attention next during deliberation.

Architectural Rule:
Honest heuristic scoring: explicitly flags is_heuristic=True. Never fabricates
calibrated neural probabilities when ranking cognitive priorities.
"""

from dataclasses import dataclass, asdict
from enum import Enum
from typing import Dict, List, Optional, Any

from chakrview.thinking.workspace import ThinkingWorkspace


class FocusType(str, Enum):
    """Categorization of deliberation focus targets."""
    UNRESOLVED_CONTRADICTION = "UNRESOLVED_CONTRADICTION"
    VERIFICATION_FAILURE = "VERIFICATION_FAILURE"
    DETECTED_WEAKNESS = "DETECTED_WEAKNESS"
    HIGH_UNCERTAINTY = "HIGH_UNCERTAINTY"
    MISSING_EVIDENCE = "MISSING_EVIDENCE"
    UNTESTED_HYPOTHESIS = "UNTESTED_HYPOTHESIS"
    PRIMARY_OBJECTIVE = "PRIMARY_OBJECTIVE"


@dataclass
class AttentionFocus:
    """The selected target of cognitive deliberation for the next step."""
    focus_type: FocusType
    item_reference: str
    priority_score: float         # 0.0 to 1.0
    rationale: str
    is_heuristic: bool = True     # Explicit scientific labeling

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["focus_type"] = self.focus_type.value
        return d


class ThinkingAttention:
    """
    Transparent attention mechanism prioritizing what the deliberation engine considers next.
    """

    def select_focus(self, workspace: ThinkingWorkspace) -> AttentionFocus:
        """
        Evaluate workspace state and select the highest priority focus item.
        """
        # 1. Unresolved contradictions in uncertainty state
        for key, unc in workspace.uncertainty_state.items():
            if "contradiction" in key.lower():
                return AttentionFocus(
                    focus_type=FocusType.UNRESOLVED_CONTRADICTION,
                    item_reference=key,
                    priority_score=0.95,
                    rationale=f"Active contradiction '{key}' requires epistemic reconciliation.",
                    is_heuristic=True,
                )

        # 2. Detected weaknesses from prior critiques
        if workspace.detected_weaknesses:
            weakest = workspace.detected_weaknesses[-1]
            return AttentionFocus(
                focus_type=FocusType.DETECTED_WEAKNESS,
                item_reference=weakest,
                priority_score=0.88,
                rationale=f"Addressing latest detected weakness: {weakest[:80]}",
                is_heuristic=True,
            )

        # 3. Verification failure in latest thought step
        for step in reversed(workspace.thought_steps):
            if step.verification_state == "FAIL":
                return AttentionFocus(
                    focus_type=FocusType.VERIFICATION_FAILURE,
                    item_reference=step.thought_id,
                    priority_score=0.85,
                    rationale=f"Revising outcome from failed step {step.thought_id} ({step.purpose.value}).",
                    is_heuristic=True,
                )

        # 4. Unanswered / unresolved questions
        if workspace.unresolved_questions:
            q = workspace.unresolved_questions[0]
            return AttentionFocus(
                focus_type=FocusType.MISSING_EVIDENCE,
                item_reference=q,
                priority_score=0.78,
                rationale=f"Gathering evidence for unanswered question: {q[:80]}",
                is_heuristic=True,
            )

        # 5. Untested active hypotheses
        for h_id, hyp in workspace.hypotheses.items():
            if not hyp.get("supporting_evidence"):
                return AttentionFocus(
                    focus_type=FocusType.UNTESTED_HYPOTHESIS,
                    item_reference=h_id,
                    priority_score=0.70,
                    rationale=f"Hypothesis {h_id} lacks supporting evidence and requires evaluation.",
                    is_heuristic=True,
                )

        # 6. Default to primary objective
        return AttentionFocus(
            focus_type=FocusType.PRIMARY_OBJECTIVE,
            item_reference=workspace.task_id,
            priority_score=0.50,
            rationale="Progressing primary task objective through standard deliberation.",
            is_heuristic=True,
        )
