"""
Safe Public Cognitive Trace for ChakrView (Step 25).

Generates safe, auditable public traces for the Unified Cognitive Architecture.

SAFETY GUARANTEE:
Never exposes raw private thoughts, internal neural logits, or unredacted
intermediate chain-of-thought reasoning steps.
Only high-level audit summaries, counts, and decision metadata are made public.
"""

from typing import Dict, Any, Optional
from chakrview.cognition.unified.models import UnifiedCognitiveState, SafePublicCognitiveTrace


class PublicTraceBuilder:
    """
    Constructs clean, sanitized public cognitive traces without internal scratchpad leakage.
    """

    @staticmethod
    def build_trace(
        state: UnifiedCognitiveState,
        hardware_profile: str,
        execution_time_ms: float,
    ) -> SafePublicCognitiveTrace:
        """
        Build a sanitized public trace from a UnifiedCognitiveState.
        """
        is_uncertain = state.decision_state.value in (
            "ANSWER_WITH_UNCERTAINTY",
            "INSUFFICIENT_INFORMATION",
            "NEED_CLARIFICATION",
        )
        has_capability_req = len(state.capability_requests) > 0
        is_capability_auth = any(
            r.get("status") in ("AUTHORIZED", "SUCCESS", "COMPLETED")
            for r in state.capability_results
        )

        # Generate safe summary statement
        summary = (
            f"Cognitive task '{state.task_type.value}' concluded with decision '{state.decision_state.value}' "
            f"(confidence={state.confidence:.2f}) across {state.revision_count} revision cycles."
        )

        return SafePublicCognitiveTrace(
            cycle_id=state.cycle_id,
            tenant_id=state.tenant_id,
            session_id=state.session_id,
            task_type=state.task_type.value,
            decision_state=state.decision_state.value,
            confidence=state.confidence,
            retrieved_memory_count=len(state.retrieved_memories),
            evidence_count=len(state.evidence),
            counter_evidence_count=len(state.counter_evidence),
            contradiction_count=len(state.contradictions),
            revision_count=state.revision_count,
            uncertainty_acknowledged=is_uncertain,
            capability_requested=has_capability_req,
            capability_authorized=is_capability_auth,
            hardware_profile=hardware_profile,
            execution_time_ms=execution_time_ms,
            weights_modified=False,  # Hard invariant: permanently False
            summary=summary,
        )
