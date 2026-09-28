"""
Execution Trace & Machine-Readable Audit Log for ChakrView (Step 15).

Captures full lifecycle telemetry for cognitive tasks with safe sanitization
of sensitive credentials and complete serializability.
"""

from dataclasses import dataclass, field, asdict
import json
import time
from typing import Dict, List, Optional, Any, Set


SENSITIVE_KEYS: Set[str] = {
    "password", "secret", "token", "api_key", "auth", "credential", "private_key"
}


def sanitize_dict(d: Any) -> Any:
    """
    Recursively sanitize dictionaries by masking sensitive values.
    """
    if isinstance(d, dict):
        clean = {}
        for k, v in d.items():
            if any(s in k.lower() for s in SENSITIVE_KEYS):
                clean[k] = "[REDACTED]"
            else:
                clean[k] = sanitize_dict(v)
        return clean
    elif isinstance(d, list):
        return [sanitize_dict(item) for item in d]
    return d


@dataclass
class TraceEntry:
    """
    Individual event within an execution trace.
    """
    event_type: str
    timestamp: float = field(default_factory=time.time)
    step_id: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": self.event_type,
            "timestamp": self.timestamp,
            "step_id": self.step_id,
            "details": sanitize_dict(self.details),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TraceEntry":
        return cls(**data)


@dataclass
class ExecutionTrace:
    """
    Machine-readable serializable audit trace of a cognitive task run.
    """
    task_id: str
    plan_id: Optional[str] = None
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    entries: List[TraceEntry] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def record_event(
        self,
        event_type: str,
        step_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Append an event to the trace with automatic redaction."""
        entry = TraceEntry(
            event_type=event_type,
            timestamp=time.time(),
            step_id=step_id,
            details=details or {},
        )
        self.entries.append(entry)

    def finalize(self) -> None:
        """Mark completion timestamp."""
        self.end_time = time.time()

    @property
    def duration_ms(self) -> float:
        """Total execution duration in milliseconds."""
        if self.end_time is None:
            return (time.time() - self.start_time) * 1000.0
        return (self.end_time - self.start_time) * 1000.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "plan_id": self.plan_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "entries": [e.to_dict() for e in self.entries],
            "metadata": sanitize_dict(self.metadata),
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert trace to formatted JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExecutionTrace":
        d = dict(data)
        d.pop("duration_ms", None)
        if "entries" in d and isinstance(d["entries"], list):
            d["entries"] = [TraceEntry.from_dict(e) for e in d["entries"]]
        return cls(**d)
