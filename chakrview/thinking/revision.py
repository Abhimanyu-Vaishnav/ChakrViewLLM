"""
Revision Engine for ChakrView Deliberation (Step 21).

Coordinates corrective revision cycles when candidate approaches fail verification
or critique:
- Diagnoses specific weaknesses or contradictions.
- Preserves historical trace without destructive mutation.
- Bounded by max_revision_cycles to guarantee termination.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any

from chakrview.thinking.thought import ThoughtPurpose
from chakrview.thinking.workspace import ThinkingWorkspace
from chakrview.thinking.critique import CritiqueResult


@dataclass
class RevisionPlan:
    """Action plan for a corrective deliberation cycle."""
    revision_index: int
    trigger_reason: str
    focus_areas: List[str]
    context_delta: str
    timestamp: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class RevisionEngine:
    """
    Formulates structured corrective directions for subsequent deliberation cycles.
    """

    def plan_revision(
        self,
        workspace: ThinkingWorkspace,
        critique: CritiqueResult,
    ) -> RevisionPlan:
        """
        Formulate a RevisionPlan based on critique findings and workspace history.
        """
        focus_areas: List[str] = []
        delta_lines: List[str] = [f"[DELIBERATION_REVISION_DIRECTIVE]"]

        # 1. Address contradictions
        if critique.contradictions_detected:
            focus_areas.append("resolve_contradiction")
            delta_lines.append("- Resolve active contradiction:")
            for c in critique.contradictions_detected:
                delta_lines.append(f"  * {c}")

        # 2. Address missing evidence
        if critique.missing_evidence:
            focus_areas.append("gather_evidence")
            delta_lines.append("- Gather or ground evidence for:")
            for m in critique.missing_evidence:
                delta_lines.append(f"  * {m}")

        # 3. Address other findings
        if critique.findings:
            focus_areas.append("correct_weakness")
            delta_lines.append("- Correct identified weaknesses:")
            for f in critique.findings:
                delta_lines.append(f"  * {f}")

        if not focus_areas:
            focus_areas.append("refine_clarity")
            delta_lines.append("- Refine solution clarity and precision.")

        delta_text = "\n".join(delta_lines)
        reason = critique.recommended_action or "Critique threshold not met."

        # Increment workspace revision counter and record thought step
        rev_idx = workspace.record_revision(reason=reason, direction=delta_text)
        workspace.add_thought_step(
            purpose=ThoughtPurpose.REVISE,
            content=f"Revision cycle #{rev_idx}: {reason}\n{delta_text}",
            verification_state="REVISION_REQUESTED",
            candidate_action="rethink_with_feedback",
        )

        return RevisionPlan(
            revision_index=rev_idx,
            trigger_reason=reason,
            focus_areas=focus_areas,
            context_delta=delta_text,
        )
