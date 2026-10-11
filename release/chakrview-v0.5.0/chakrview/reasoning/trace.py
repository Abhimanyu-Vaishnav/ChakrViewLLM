"""
Auditable Reasoning Trace for ChakrView (Step 19).

Captures the structured execution progression across all reasoning phases:
Task -> Decomposition -> Evidence -> Hypotheses -> Inferences -> Decision -> Action
-> Observation -> Verification -> Revision -> Outcome.

Guarantees safe JSON serialization without pickle and provides structured,
non-anthropomorphic summaries for inspection, replay, and auditing.
"""

from dataclasses import dataclass, field
import json
import time
from typing import Dict, List, Optional, Any
import uuid


@dataclass
class ReasoningTrace:
    """
    Comprehensive, inspectable audit log of a governed reasoning execution.
    """
    trace_id: str = field(default_factory=lambda: f"rtrace_{uuid.uuid4().hex[:8]}")
    task_id: str = ""
    events: List[Dict[str, Any]] = field(default_factory=list)
    subproblems: List[Dict[str, Any]] = field(default_factory=list)
    evidence_items: List[Dict[str, Any]] = field(default_factory=list)
    hypotheses: List[Dict[str, Any]] = field(default_factory=list)
    inferences: List[Dict[str, Any]] = field(default_factory=list)
    contradictions: List[Dict[str, Any]] = field(default_factory=list)
    decisions: List[Dict[str, Any]] = field(default_factory=list)
    observations: List[Dict[str, Any]] = field(default_factory=list)
    verifications: List[Dict[str, Any]] = field(default_factory=list)
    revisions: List[Dict[str, Any]] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    outcome: Optional[str] = None
    success: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def record_event(self, event_type: str, phase: str, details: Optional[Dict[str, Any]] = None) -> None:
        """Append an auditable event timestamped to system time."""
        self.events.append({
            "event_id": f"evt_{uuid.uuid4().hex[:6]}",
            "event_type": event_type,
            "phase": phase,
            "timestamp": time.time(),
            "details": dict(details or {}),
        })

    def finalize(self, success: bool, outcome: str) -> None:
        """Mark trace completion."""
        self.completed_at = time.time()
        self.success = success
        self.outcome = outcome
        self.record_event(
            event_type="REASONING_COMPLETED",
            phase="COMPLETE" if success else "FAILED",
            details={"success": success, "outcome": outcome},
        )

    def get_safe_summary(self) -> Dict[str, Any]:
        """
        Produce a machine-readable summary suitable for user presentation or monitoring
        without exposing uncontrolled internal chain-of-thought text.
        """
        decisions_summary = []
        for d in self.decisions:
            decisions_summary.append({
                "decision_id": d.get("decision_id"),
                "selected_candidate": d.get("selected_candidate_id"),
                "rationale": d.get("rationale"),
                "confidence": d.get("confidence"),
            })

        verifications_summary = []
        for v in self.verifications:
            verifications_summary.append({
                "verification_id": v.get("verification_id"),
                "status": v.get("status"),
                "criteria_count": len(v.get("criteria_evaluated", [])),
                "failure_reason": v.get("failure_reason"),
            })

        return {
            "trace_id": self.trace_id,
            "task_id": self.task_id,
            "total_events": len(self.events),
            "subproblem_count": len(self.subproblems),
            "evidence_count": len(self.evidence_items),
            "hypothesis_count": len(self.hypotheses),
            "inference_count": len(self.inferences),
            "contradiction_count": len(self.contradictions),
            "revision_count": len(self.revisions),
            "decisions": decisions_summary,
            "verifications": verifications_summary,
            "duration_ms": round(((self.completed_at or time.time()) - self.created_at) * 1000, 2),
            "success": self.success,
            "outcome": self.outcome,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "task_id": self.task_id,
            "events": list(self.events),
            "subproblems": list(self.subproblems),
            "evidence_items": list(self.evidence_items),
            "hypotheses": list(self.hypotheses),
            "inferences": list(self.inferences),
            "contradictions": list(self.contradictions),
            "decisions": list(self.decisions),
            "observations": list(self.observations),
            "verifications": list(self.verifications),
            "revisions": list(self.revisions),
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "outcome": self.outcome,
            "success": self.success,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReasoningTrace":
        return cls(**data)

    def to_json(self, indent: Optional[int] = None) -> str:
        """Serialize trace to safe standard JSON."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    @classmethod
    def from_json(cls, json_str: str) -> "ReasoningTrace":
        """Deserialize trace from JSON safely without pickle."""
        data = json.loads(json_str)
        return cls.from_dict(data)
