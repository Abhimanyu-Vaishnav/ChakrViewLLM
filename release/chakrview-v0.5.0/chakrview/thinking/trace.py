"""
Thinking Trace and Auditable Telemetry for ChakrView (Step 21).

Provides structured, JSON-serializable records of the deliberation process:
- Tracks steps, hypotheses, evidence references, critiques, and revisions.
- Strictly guards against dumping unrestricted private chain-of-thought to users.
- Generates concise, safe public summaries for transparent inspection.
"""

from dataclasses import dataclass, field, asdict
import time
from typing import Dict, List, Optional, Any
import uuid


@dataclass
class ThinkingTrace:
    """
    Complete auditable execution trace of a deliberate thinking session.
    """
    deliberation_id: str = field(default_factory=lambda: f"delib_{uuid.uuid4().hex[:10]}")
    task_id: str = "task_default"
    owner_id: str = "default_user"
    session_id: str = "default_session"
    number_of_steps: int = 0
    number_of_revisions: int = 0
    hypotheses_considered: List[str] = field(default_factory=list)
    evidence_used: List[str] = field(default_factory=list)
    critiques: List[Dict[str, Any]] = field(default_factory=list)
    revisions: List[Dict[str, Any]] = field(default_factory=list)
    stopping_condition: str = "UNCONCLUDED"
    stopping_reason: str = ""
    final_status: str = "IN_PROGRESS"
    timing_metrics: Dict[str, float] = field(default_factory=dict)
    weights_modified: bool = False  # Invariant: ALWAYS False

    def finalize(
        self,
        status: str,
        condition: str,
        reason: str,
        elapsed_ms: float,
    ) -> None:
        """Finalize trace metadata."""
        self.final_status = status
        self.stopping_condition = condition
        self.stopping_reason = reason
        self.timing_metrics["total_deliberation_ms"] = round(elapsed_ms, 3)

    def get_safe_summary(self) -> Dict[str, Any]:
        """
        Produce a privacy-safe, structured public summary.
        
        Crucial Rule:
        Does NOT dump private internal thought text or raw chain-of-thought.
        """
        return {
            "deliberation_id": self.deliberation_id,
            "task_id": self.task_id,
            "status": self.final_status,
            "stopping_condition": self.stopping_condition,
            "stopping_reason": self.stopping_reason,
            "steps_executed": self.number_of_steps,
            "revisions_performed": self.number_of_revisions,
            "hypotheses_evaluated": len(self.hypotheses_considered),
            "evidence_items_referenced": len(self.evidence_used),
            "total_deliberation_ms": self.timing_metrics.get("total_deliberation_ms", 0.0),
            "weights_modified": self.weights_modified,
        }

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ThinkingTrace":
        return cls(**dict(data))
