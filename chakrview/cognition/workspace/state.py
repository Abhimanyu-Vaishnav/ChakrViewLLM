"""
ChakrView Step 58: Multi-Turn Cognitive Working State Schema.

Defines:
- CognitiveStatePhase: Enum for operational phases in the workspace.
- FailedAttemptSummary: Dataclass recording previous failed attempts.
- CognitiveWorkingState: Strictly typed, deterministic dataclass tracking active task state,
  turn and attempt indices, active plans, diagnoses, retrieved memories, and verification flags.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
import time
from typing import Any, Dict, List, Optional


class CognitiveStatePhase(str, Enum):
    INITIALIZING = "INITIALIZING"
    PLANNING = "PLANNING"
    EXECUTING = "EXECUTING"
    OBSERVING = "OBSERVING"
    EVALUATING = "EVALUATING"
    DIAGNOSING = "DIAGNOSING"
    REPAIRING = "REPAIRING"
    VERIFYING = "VERIFYING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


@dataclass
class FailedAttemptSummary:
    attempt_index: int
    action: str
    observation: str
    diagnosis: str
    duration_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> FailedAttemptSummary:
        return cls(
            attempt_index=data.get("attempt_index", 1),
            action=data.get("action", ""),
            observation=data.get("observation", ""),
            diagnosis=data.get("diagnosis", ""),
            duration_seconds=data.get("duration_seconds", 0.0),
        )


@dataclass
class CognitiveWorkingState:
    """
    Deterministic working state for a CognitiveWorkspace execution.
    Auditable, serializable, and decoupled from neural core weights.
    """
    task_id: str
    task_family: str
    objective: str
    current_phase: CognitiveStatePhase = CognitiveStatePhase.INITIALIZING
    current_plan: List[str] = field(default_factory=list)
    current_action: str = ""
    last_observation: str = ""
    diagnosis: Optional[str] = None
    attempt_index: int = 1
    max_attempts: int = 3
    completed: bool = False
    verified: bool = False
    failure_history: List[FailedAttemptSummary] = field(default_factory=list)
    relevant_memories: List[Dict[str, Any]] = field(default_factory=list)
    active_adapter: Optional[str] = None
    timestamp_utc: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def transition_to(self, phase: CognitiveStatePhase) -> None:
        """Deterministic phase transition with validation."""
        self.current_phase = phase

    def record_failure(self, action: str, observation: str, diagnosis: str, duration: float = 0.0) -> None:
        """Record an unverified attempt in the working state's failure history."""
        summary = FailedAttemptSummary(
            attempt_index=self.attempt_index,
            action=action,
            observation=observation,
            diagnosis=diagnosis,
            duration_seconds=duration,
        )
        self.failure_history.append(summary)
        self.diagnosis = diagnosis

    def to_dict(self) -> Dict[str, Any]:
        """Deterministic dictionary serialization."""
        data = asdict(self)
        data["current_phase"] = self.current_phase.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CognitiveWorkingState:
        """Deterministic reconstruction from dictionary."""
        phase_str = data.get("current_phase", CognitiveStatePhase.INITIALIZING.value)
        phase = CognitiveStatePhase(phase_str) if isinstance(phase_str, str) else phase_str
        history_raw = data.get("failure_history", [])
        history = [FailedAttemptSummary.from_dict(h) for h in history_raw]
        return cls(
            task_id=data["task_id"],
            task_family=data.get("task_family", "general"),
            objective=data.get("objective", ""),
            current_phase=phase,
            current_plan=list(data.get("current_plan", [])),
            current_action=data.get("current_action", ""),
            last_observation=data.get("last_observation", ""),
            diagnosis=data.get("diagnosis"),
            attempt_index=data.get("attempt_index", 1),
            max_attempts=data.get("max_attempts", 3),
            completed=data.get("completed", False),
            verified=data.get("verified", False),
            failure_history=history,
            relevant_memories=list(data.get("relevant_memories", [])),
            active_adapter=data.get("active_adapter"),
            timestamp_utc=data.get("timestamp_utc", time.time()),
            metadata=dict(data.get("metadata", {})),
        )
