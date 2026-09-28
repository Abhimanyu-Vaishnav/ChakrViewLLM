"""
Governed Memory Bridge for Step 28 Cognitive Orchestration.

Sanitizes and captures orchestration outcomes into the Continual Cognition
and Memory Subsystems under strict governance rules.

CRITICAL INVARIANTS:
1. PRIVACY & SECURITY: NEVER stores private chain-of-thought, raw activations,
   logits, or cryptographic keys.
2. GOVERNED PROMOTION: Experiences are recorded through existing governed memory
   structures and remain subject to consolidation policies.
"""

from typing import Dict, Any, Optional
import time

from chakrview.cognition.orchestration.models import (
    WorkloadClass,
    ResourceAllocationDecision,
    SafePublicOrchestrationTrace,
)
from chakrview.cognition.unified.models import DecisionState
from chakrview.memory.engine import ContinualCognitionEngine


class GovernedOrchestrationMemoryBridge:
    """
    Sanitizes orchestration results and routes approved structured experiences
    into the governed memory engine.
    """

    def __init__(self, memory_engine: Optional[ContinualCognitionEngine] = None) -> None:
        self.memory_engine = memory_engine

    def record_orchestration_experience(
        self,
        trace: SafePublicOrchestrationTrace,
        allocation: ResourceAllocationDecision,
        objective: str,
        final_answer: str,
        decision_state: DecisionState,
    ) -> Optional[str]:
        """
        Record a sanitized, governed episode for the completed orchestration cycle.
        """
        if self.memory_engine is None:
            return None

        # Build sanitized experience payload
        sanitized_summary = (
            f"Orchestration completed for {trace.workload_class} task. "
            f"Allocated {allocation.allocated_agent_count} agents across {allocation.allocated_node_count} nodes. "
            f"Decision: {decision_state.value}. Conflicts: {trace.conflicts_detected}."
        )

        metadata: Dict[str, Any] = {
            "orchestration_trace_id": trace.trace_id,
            "workload_class": trace.workload_class,
            "resource_profile": trace.resource_profile,
            "rounds_executed": trace.rounds_executed,
            "conflicts_detected": trace.conflicts_detected,
            "minority_evidence_preserved": trace.minority_evidence_preserved,
            "verification_invoked": trace.verification_invoked,
            "verification_passed": trace.verification_passed,
            "latency_ms": trace.total_latency_ms,
            "termination_reason": trace.termination_reason,
            "weights_modified": False,
        }

        try:
            episode = self.memory_engine.record_experience(
                tenant_id=trace.tenant_id,
                session_id=trace.session_id,
                situation=objective[:256],
                action_or_response=final_answer[:512],
                outcome=sanitized_summary[:256],
                task_id=trace.task_id,
                metadata=metadata,
            )
            return getattr(episode, "episode_id", None)
        except Exception:
            return None
