"""
Reasoning Task Model for ChakrView (Step 19).

Defines strongly typed representations of reasoning tasks, lifecycle phases,
and completion statuses with multi-tenant ownership metadata.
"""

from dataclasses import dataclass, field, asdict
from enum import Enum
import time
from typing import Dict, List, Optional, Any
import uuid


class ReasoningTaskType(str, Enum):
    """Categorization of reasoning problems."""
    ANALYTIC = "ANALYTIC"
    DEDUCTIVE = "DEDUCTIVE"
    MATHEMATICAL = "MATHEMATICAL"
    EXPLORATORY = "EXPLORATORY"
    DECISION_MAKING = "DECISION_MAKING"
    HYPOTHESIS_TESTING = "HYPOTHESIS_TESTING"
    GENERAL = "GENERAL"


class ReasoningPhase(str, Enum):
    """Explicit lifecycle phases of the governed reasoning loop."""
    UNDERSTAND = "UNDERSTAND"
    DECOMPOSE = "DECOMPOSE"
    EVIDENCE_GATHERING = "EVIDENCE_GATHERING"
    HYPOTHESIS = "HYPOTHESIS"
    INFERENCE = "INFERENCE"
    CONSISTENCY_CHECK = "CONSISTENCY_CHECK"
    UNCERTAINTY_ESTIMATION = "UNCERTAINTY_ESTIMATION"
    DECISION = "DECISION"
    ACTION = "ACTION"
    VERIFICATION = "VERIFICATION"
    REVISION = "REVISION"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"


class ReasoningStatus(str, Enum):
    """Execution status of a reasoning task."""
    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    REVISION_REQUIRED = "REVISION_REQUIRED"


@dataclass
class ReasoningTask:
    """
    Strongly typed, inspectable container for an active reasoning problem.
    """
    task_id: str = field(default_factory=lambda: f"rtask_{uuid.uuid4().hex[:8]}")
    owner_id: str = "default_owner"
    session_id: str = "default_session"
    original_objective: str = ""
    normalized_objective: str = ""
    task_type: ReasoningTaskType = ReasoningTaskType.GENERAL
    complexity_estimate: float = 0.5
    required_capabilities: List[str] = field(default_factory=list)
    constraints: Dict[str, Any] = field(default_factory=dict)
    deadline: Optional[float] = None
    current_phase: ReasoningPhase = ReasoningPhase.UNDERSTAND
    completion_status: ReasoningStatus = ReasoningStatus.PENDING
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def transition_to(
        self,
        phase: ReasoningPhase,
        status: Optional[ReasoningStatus] = None,
        reason: str = "",
    ) -> None:
        """Advance reasoning phase with validation and timestamp tracking."""
        self.current_phase = phase
        if status is not None:
            self.completion_status = status
        self.updated_at = time.time()
        if reason:
            if "phase_history" not in self.metadata:
                self.metadata["phase_history"] = []
            self.metadata["phase_history"].append({
                "phase": phase.value,
                "status": self.completion_status.value,
                "reason": reason,
                "timestamp": self.updated_at,
            })

    def to_dict(self) -> Dict[str, Any]:
        """Safe JSON-serializable dictionary representation."""
        return {
            "task_id": self.task_id,
            "owner_id": self.owner_id,
            "session_id": self.session_id,
            "original_objective": self.original_objective,
            "normalized_objective": self.normalized_objective,
            "task_type": self.task_type.value,
            "complexity_estimate": self.complexity_estimate,
            "required_capabilities": list(self.required_capabilities),
            "constraints": dict(self.constraints),
            "deadline": self.deadline,
            "current_phase": self.current_phase.value,
            "completion_status": self.completion_status.value,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReasoningTask":
        """Deserialize from dictionary safely."""
        d = dict(data)
        d["task_type"] = ReasoningTaskType(d["task_type"])
        d["current_phase"] = ReasoningPhase(d["current_phase"])
        d["completion_status"] = ReasoningStatus(d["completion_status"])
        return cls(**d)
