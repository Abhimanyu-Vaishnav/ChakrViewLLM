"""
Cognitive Task Model and Lifecycle State Machine for ChakrView (Step 15).

Provides typed, serializable task representations with strict, validated
lifecycle transitions, bounded resource constraints, and state tracking.
"""

from dataclasses import dataclass, field, asdict
from enum import Enum
import time
from typing import Dict, List, Optional, Any, Set, Tuple


class TaskStatus(str, Enum):
    """Explicit lifecycle states for a CognitiveTask."""
    PENDING = "PENDING"
    PLANNING = "PLANNING"
    READY = "READY"
    RUNNING = "RUNNING"
    WAITING = "WAITING"
    VERIFYING = "VERIFYING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    ROLLED_BACK = "ROLLED_BACK"


# Valid explicit state transitions
VALID_TRANSITIONS: Dict[TaskStatus, Set[TaskStatus]] = {
    TaskStatus.PENDING: {
        TaskStatus.PLANNING,
        TaskStatus.CANCELLED,
    },
    TaskStatus.PLANNING: {
        TaskStatus.READY,
        TaskStatus.FAILED,
        TaskStatus.CANCELLED,
    },
    TaskStatus.READY: {
        TaskStatus.RUNNING,
        TaskStatus.CANCELLED,
    },
    TaskStatus.RUNNING: {
        TaskStatus.WAITING,
        TaskStatus.VERIFYING,
        TaskStatus.COMPLETED,
        TaskStatus.FAILED,
        TaskStatus.CANCELLED,
    },
    TaskStatus.WAITING: {
        TaskStatus.RUNNING,
        TaskStatus.FAILED,
        TaskStatus.CANCELLED,
    },
    TaskStatus.VERIFYING: {
        TaskStatus.RUNNING,
        TaskStatus.COMPLETED,
        TaskStatus.FAILED,
        TaskStatus.ROLLED_BACK,
        TaskStatus.CANCELLED,
    },
    TaskStatus.FAILED: {
        TaskStatus.ROLLED_BACK,
        TaskStatus.PLANNING,  # Re-planning
        TaskStatus.CANCELLED,
    },
    TaskStatus.ROLLED_BACK: {
        TaskStatus.READY,
        TaskStatus.PLANNING,
        TaskStatus.FAILED,
        TaskStatus.CANCELLED,
    },
    TaskStatus.COMPLETED: set(),  # Terminal state
    TaskStatus.CANCELLED: set(),  # Terminal state
}


class InvalidStateTransitionError(ValueError):
    """Raised when an illegal lifecycle transition is attempted."""
    pass


@dataclass
class TaskConstraints:
    """
    Bounded resource constraints for cognitive task execution.
    """
    max_depth: int = 5
    max_steps: int = 10
    max_retries: int = 3
    timeout_seconds: float = 60.0
    max_token_budget: int = 512

    def __post_init__(self) -> None:
        if self.max_depth <= 0:
            raise ValueError(f"max_depth must be positive, got {self.max_depth}")
        if self.max_steps <= 0:
            raise ValueError(f"max_steps must be positive, got {self.max_steps}")
        if self.max_retries < 0:
            raise ValueError(f"max_retries cannot be negative, got {self.max_retries}")
        if self.timeout_seconds <= 0.0:
            raise ValueError(f"timeout_seconds must be positive, got {self.timeout_seconds}")
        if self.max_token_budget <= 0 or self.max_token_budget > 512:
            raise ValueError(f"max_token_budget must be in [1, 512], got {self.max_token_budget}")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskConstraints":
        return cls(**data)


@dataclass
class StateTransitionRecord:
    """Audit entry recording a state change."""
    from_status: TaskStatus
    to_status: TaskStatus
    timestamp: float
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "from_status": self.from_status.value,
            "to_status": self.to_status.value,
            "timestamp": self.timestamp,
            "reason": self.reason,
        }


@dataclass
class CognitiveTask:
    """
    Structured, serializable cognitive task model.
    
    Attributes:
        task_id: Unique task identifier string.
        user_request: Original user goal or prompt.
        status: Current lifecycle state.
        priority: Execution priority (0 = normal, 1 = high, etc.).
        created_at: Epoch timestamp of creation.
        current_step_id: Identifier of active executing step.
        parent_task_id: Optional parent task ID for subtasks.
        constraints: Bounded resource configuration.
        metadata: Domain and session tracking metadata.
        execution_state: Mutable transient working memory and outputs for this task.
        transition_history: Chronological log of all validated state transitions.
    """
    task_id: str
    user_request: str
    status: TaskStatus = TaskStatus.PENDING
    priority: int = 0
    created_at: float = field(default_factory=time.time)
    current_step_id: Optional[str] = None
    parent_task_id: Optional[str] = None
    constraints: TaskConstraints = field(default_factory=TaskConstraints)
    metadata: Dict[str, Any] = field(default_factory=dict)
    execution_state: Dict[str, Any] = field(default_factory=dict)
    transition_history: List[StateTransitionRecord] = field(default_factory=list)

    def transition_to(self, new_status: TaskStatus, reason: str = "") -> None:
        """
        Validate and transition task to a new lifecycle state.
        
        Raises:
            InvalidStateTransitionError: If the transition is prohibited by VALID_TRANSITIONS.
        """
        if new_status == self.status:
            return  # No-op for idempotent re-assertion

        allowed = VALID_TRANSITIONS.get(self.status, set())
        if new_status not in allowed:
            raise InvalidStateTransitionError(
                f"Cannot transition task '{self.task_id}' from {self.status.value} to {new_status.value}. "
                f"Valid targets from {self.status.value}: {[s.value for s in allowed]}"
            )

        record = StateTransitionRecord(
            from_status=self.status,
            to_status=new_status,
            timestamp=time.time(),
            reason=reason,
        )
        self.transition_history.append(record)
        self.status = new_status

    def is_terminal(self) -> bool:
        """Check if task is in a terminal state."""
        return self.status in (TaskStatus.COMPLETED, TaskStatus.CANCELLED)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize task to structured dictionary."""
        return {
            "task_id": self.task_id,
            "user_request": self.user_request,
            "status": self.status.value,
            "priority": self.priority,
            "created_at": self.created_at,
            "current_step_id": self.current_step_id,
            "parent_task_id": self.parent_task_id,
            "constraints": self.constraints.to_dict(),
            "metadata": self.metadata,
            "execution_state": self.execution_state,
            "transition_history": [r.to_dict() for r in self.transition_history],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CognitiveTask":
        """Deserialize task from structured dictionary."""
        d = dict(data)
        d["status"] = TaskStatus(d["status"])
        if isinstance(d.get("constraints"), dict):
            d["constraints"] = TaskConstraints.from_dict(d["constraints"])
        if "transition_history" in d and isinstance(d["transition_history"], list):
            records = []
            for item in d["transition_history"]:
                records.append(StateTransitionRecord(
                    from_status=TaskStatus(item["from_status"]),
                    to_status=TaskStatus(item["to_status"]),
                    timestamp=item["timestamp"],
                    reason=item.get("reason", ""),
                ))
            d["transition_history"] = records
        return cls(**d)
