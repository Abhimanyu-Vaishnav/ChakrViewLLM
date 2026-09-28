"""
Thinking Stopping Policy for ChakrView (Step 21).

Decides deterministically when enough deliberation has occurred and evaluates
terminal conditions:
- Distinguishes SOLVED from SOLVED_WITH_UNCERTAINTY.
- Prevents infinite thought accumulation (MAX_REASONING_LIMIT, MAX_REVISIONS_REACHED).
- Acknowledges epistemic boundaries honestly (INSUFFICIENT_INFORMATION, REQUIRES_EXTERNAL_INPUT).
- Never forces a false fabricated answer when thinking budget expires.
"""

from enum import Enum
from typing import Tuple, Optional

from chakrview.thinking.workspace import ThinkingWorkspace
from chakrview.thinking.critique import CritiqueResult, CritiqueVerdict


class StoppingCondition(str, Enum):
    """Formal terminal condition concluding the deliberation loop."""
    SOLVED = "SOLVED"                                         # Verified pass and critique passed
    SOLVED_WITH_UNCERTAINTY = "SOLVED_WITH_UNCERTAINTY"       # Adequate answer with documented uncertainty
    INSUFFICIENT_INFORMATION = "INSUFFICIENT_INFORMATION"     # Missing essential evidence
    MAX_REASONING_LIMIT = "MAX_REASONING_LIMIT"               # Reached step or timeout budget ceiling
    MAX_REVISIONS_REACHED = "MAX_REVISIONS_REACHED"           # Reached maximum allowed revision cycles
    REQUIRES_EXTERNAL_INPUT = "REQUIRES_EXTERNAL_INPUT"       # Requires user or sensor interaction
    FAILED_VERIFICATION = "FAILED_VERIFICATION"               # Failed formal invariant checks


class ThinkingStoppingPolicy:
    """
    Evaluates whether deliberation should terminate or continue to another cycle.
    """

    def evaluate_stopping(
        self,
        workspace: ThinkingWorkspace,
        latest_critique: Optional[CritiqueResult] = None,
        verification_passed: bool = False,
    ) -> Tuple[bool, StoppingCondition, str]:
        """
        Evaluate workspace state and returns (should_stop, condition, explanation).
        """
        policy = workspace.policy

        # 1. Deliberation timeout check
        if workspace.elapsed_time_ms >= policy.max_deliberation_time_ms:
            return (
                True,
                StoppingCondition.MAX_REASONING_LIMIT,
                f"Deliberation timeout ({policy.max_deliberation_time_ms:.1f} ms) exceeded.",
            )

        # 2. Maximum thought steps reached
        if workspace.step_count >= policy.max_thought_steps:
            return (
                True,
                StoppingCondition.MAX_REASONING_LIMIT,
                f"Maximum allowed thought steps ({policy.max_thought_steps}) reached.",
            )

        # 3. Maximum revision cycles reached
        if workspace.revision_count >= policy.max_revision_cycles:
            if verification_passed and latest_critique and latest_critique.is_passed(policy.min_critique_score):
                return (
                    True,
                    StoppingCondition.SOLVED,
                    "Deliberation satisfied after final revision cycle.",
                )
            return (
                True,
                StoppingCondition.MAX_REVISIONS_REACHED,
                f"Maximum revision cycles ({policy.max_revision_cycles}) exhausted without full verification pass.",
            )

        # 4. Explicit missing essential information check
        if latest_critique and latest_critique.verdict == CritiqueVerdict.INSUFFICIENT_EVIDENCE:
            if not workspace.evidence and len(workspace.unresolved_questions) > 0:
                return (
                    True,
                    StoppingCondition.INSUFFICIENT_INFORMATION,
                    "Essential premise evidence is missing and cannot be derived from local state.",
                )

        # 5. Success / Solved check
        critique_ok = latest_critique.is_passed(policy.min_critique_score) if latest_critique else False
        verif_ok = verification_passed or not policy.require_verification_to_stop

        if critique_ok and verif_ok and workspace.current_working_conclusion:
            # Check for active unresolved contradictions in uncertainty state
            has_contradiction = any("contradiction" in k.lower() for k in workspace.uncertainty_state)
            if has_contradiction:
                # Must not stop if high-priority contradiction is unresolved and revisions remain
                return (
                    False,
                    StoppingCondition.INSUFFICIENT_INFORMATION,
                    "Continuing deliberation: active contradiction requires resolution.",
                )

            # Check for residual acknowledged uncertainty
            if workspace.uncertainty_state:
                return (
                    True,
                    StoppingCondition.SOLVED_WITH_UNCERTAINTY,
                    "Solution verified with residual acknowledged uncertainty.",
                )

            return (
                True,
                StoppingCondition.SOLVED,
                "Solution verified and satisfied all critique constraints.",
            )

        # Deliberation continues
        return (
            False,
            StoppingCondition.INSUFFICIENT_INFORMATION,
            "Deliberation criteria not yet satisfied; continuing deliberation cycle.",
        )
