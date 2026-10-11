"""
Task State Representation for ChakrView (Step 18).

Provides inspectable, auditable state tracking for active and completed
cognitive tasks, including phases, required capabilities, observations,
decisions, and failure telemetry.
"""

from dataclasses import dataclass, field, asdict
from enum import Enum
import time
from typing import Dict, List, Optional, Any


class TaskPhase(str, Enum):
    """Execution phase of an active cognitive task."""
    IDLE = "IDLE"
    PLANNING = "PLANNING"
    EXECUTING = "EXECUTING"
    OBSERVING = "OBSERVING"
    VERIFYING = "VERIFYING"
    RECOVERING = "RECOVERING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


@dataclass
class TaskState:
    """
    State record of a cognitive task within the state layer.
    """
    task_id: str
    goal: str
    status: str = "PENDING"
    phase: TaskPhase = TaskPhase.IDLE
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    parent_task_id: Optional[str] = None
    current_step_id: Optional[str] = None
    required_capabilities: List[str] = field(default_factory=list)
    execution_state: Dict[str, Any] = field(default_factory=dict)
    observations: List[Dict[str, Any]] = field(default_factory=list)
    decisions: List[Dict[str, Any]] = field(default_factory=list)
    failures: List[Dict[str, Any]] = field(default_factory=list)
    completion_state: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def transition_phase(self, new_phase: TaskPhase, note: str = "") -> None:
        """Update active phase with timestamp."""
        self.phase = new_phase
        self.updated_at = time.time()
        if note:
            self.metadata[f"phase_{new_phase.value}_note"] = note

    def record_observation(
        self,
        step_id: str,
        output: Any,
        success: bool,
        elapsed_ms: float = 0.0,
        provenance: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Append an observation event to task history."""
        self.observations.append({
            "step_id": step_id,
            "output": output,
            "success": success,
            "elapsed_ms": elapsed_ms,
            "timestamp": time.time(),
            "provenance": provenance or {},
        })
        self.updated_at = time.time()

    def record_decision(
        self,
        rationale: str,
        chosen_action: str,
        alternatives: Optional[List[str]] = None,
    ) -> None:
        """Append a decision event to task audit trail."""
        self.decisions.append({
            "timestamp": time.time(),
            "rationale": rationale,
            "chosen_action": chosen_action,
            "alternatives": alternatives or [],
        })
        self.updated_at = time.time()

    def record_failure(
        self,
        step_id: str,
        error: str,
        recovery_attempted: bool = False,
    ) -> None:
        """Append a failure event to task telemetry."""
        self.failures.append({
            "timestamp": time.time(),
            "step_id": step_id,
            "error": error,
            "recovery_attempted": recovery_attempted,
        })
        self.updated_at = time.time()

    def mark_completed(
        self,
        summary: str,
        output_references: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Mark task as COMPLETED and record completion payload."""
        self.status = "COMPLETED"
        self.phase = TaskPhase.COMPLETED
        self.updated_at = time.time()
        self.completion_state = {
            "summary": summary,
            "completed_at": self.updated_at,
            "outputs": output_references or {},
        }

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["phase"] = self.phase.value
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskState":
        d = dict(data)
        d["phase"] = TaskPhase(d["phase"])
        return cls(**d)
