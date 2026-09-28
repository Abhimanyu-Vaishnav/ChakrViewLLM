"""
Recovery & Rollback Foundation for ChakrView Cognitive Subsystem (Step 15).

Provides safe recovery primitives:
- Bounded retries
- Failure propagation prevention
- Task execution state rollback
- Audit history preservation
"""

from dataclasses import dataclass, field
import time
from typing import Dict, List, Optional, Any

from chakrview.cognition.task import CognitiveTask, TaskStatus
from chakrview.cognition.planner import PlanStep, PlanStepStatus
from chakrview.cognition.graph import ExecutionGraph
from chakrview.cognition.observation import StepObservation


@dataclass
class RollbackRecord:
    """
    Audit record of a cognitive state rollback event.
    """
    step_id: str
    previous_status: TaskStatus
    reverted_state_keys: List[str]
    timestamp: float = field(default_factory=time.time)
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_id": self.step_id,
            "previous_status": self.previous_status.value,
            "reverted_state_keys": self.reverted_state_keys,
            "timestamp": self.timestamp,
            "reason": self.reason,
        }


class RecoveryManager:
    """
    Manages bounded retries, failure isolation, and cognitive state rollback.
    """

    def __init__(self) -> None:
        self.rollback_history: List[RollbackRecord] = []

    def should_retry(self, step: PlanStep, observation: StepObservation) -> bool:
        """
        Determine whether a failed step is eligible for another attempt.
        """
        if observation.success and observation.verification_status:
            return False
        return step.retry_policy.can_retry()

    def handle_retry(self, step: PlanStep, graph: ExecutionGraph) -> int:
        """
        Record retry attempt and reset step to READY.
        """
        count = step.retry_policy.record_retry()
        step.status = PlanStepStatus.READY
        step.metadata["last_retry_at"] = time.time()
        step.metadata["retry_count"] = count
        return count

    def handle_step_failure(
        self,
        task: CognitiveTask,
        step: PlanStep,
        observation: StepObservation,
        graph: ExecutionGraph,
    ) -> List[str]:
        """
        Mark step as failed, isolate downstream dependencies by marking them BLOCKED.
        """
        error_msg = observation.error or observation.verification_notes or "Step execution failed."
        blocked = graph.mark_step_failed(step.step_id, error=error_msg)
        return blocked

    def rollback_task_state(
        self,
        task: CognitiveTask,
        step: PlanStep,
        reason: str = "",
    ) -> RollbackRecord:
        """
        Roll back transient execution state keys associated with the failed step,
        and transition the task status to ROLLED_BACK.
        """
        prev_status = task.status
        reverted_keys: List[str] = []

        # Remove keys produced by this step from task.execution_state
        for out_ref_name, state_key in step.output_references.items():
            if state_key in task.execution_state:
                del task.execution_state[state_key]
                reverted_keys.append(state_key)

        record = RollbackRecord(
            step_id=step.step_id,
            previous_status=prev_status,
            reverted_state_keys=reverted_keys,
            reason=reason or f"Rollback following failure of step '{step.step_id}'",
        )
        self.rollback_history.append(record)

        # Transition task state safely
        if task.status != TaskStatus.ROLLED_BACK:
            # Check if task is in FAILED or VERIFYING or RUNNING
            if task.status == TaskStatus.RUNNING:
                task.transition_to(TaskStatus.VERIFYING, reason="Moving to verifying before rollback")
            if task.status == TaskStatus.VERIFYING:
                task.transition_to(TaskStatus.ROLLED_BACK, reason=record.reason)
            elif task.status == TaskStatus.FAILED:
                task.transition_to(TaskStatus.ROLLED_BACK, reason=record.reason)

        return record
