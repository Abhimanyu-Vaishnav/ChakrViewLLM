"""
ChakrView Step 86: Dynamic Task Expansion & Replanning Models.

Defines:
- TaskProvenance: Explicit audit record detailing why a task was created (parent node, observation, trigger).
- DynamicSubtaskRequest: Request to dynamically expand the task graph with newly discovered work.
- GoalVerificationVerdict: Epistemic evaluation outcome for the root goal (ACCEPT, PARTIAL, REVISE, REJECT, ABSTAIN).
- GoalVerificationResult: Detailed breakdown of goal-level verification including unsatisfied constraints,
  unresolved unknowns, follow-up recommendations, and justification.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from chakrview.cognition.ppb.task_models import TaskResourceType, TaskNodeStatus


class GoalVerificationVerdict(str, Enum):
    """Authoritative verdicts for whole-goal completion evaluation."""
    ACCEPT = "ACCEPT"     # All objectives, constraints, tests, and verifications passed without unresolved blockers
    PARTIAL = "PARTIAL"   # Intermediate milestones accomplished, but critical follow-ups required
    REVISE = "REVISE"     # Outputs produced but fail one or more quality, safety, or constraint gates; revision needed
    REJECT = "REJECT"     # Fundamental failure, regression, or boundary violation
    ABSTAIN = "ABSTAIN"   # Insufficient evidence or conflicting signals prevent honest verification


@dataclass(frozen=True)
class TaskProvenance:
    """
    Explicit provenance record explaining the origin and justification for a task.
    """
    origin_type: str  # "DECOMPOSITION", "DYNAMIC_EXPANSION", "INVESTIGATION", "REPLANNING"
    trigger_node_id: Optional[str] = None
    trigger_observation: str = ""
    evidence_ids: Tuple[str, ...] = field(default_factory=tuple)
    rationale: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "origin_type": self.origin_type,
            "trigger_node_id": self.trigger_node_id,
            "trigger_observation": self.trigger_observation,
            "evidence_ids": list(self.evidence_ids),
            "rationale": self.rationale,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TaskProvenance:
        d = dict(data)
        d["evidence_ids"] = tuple(d.get("evidence_ids", []))
        return cls(**d)


@dataclass(frozen=True)
class DynamicSubtaskRequest:
    """
    Request submitted mid-flight to expand the active task graph with new subtasks.
    """
    title: str
    description: str
    resource_type: TaskResourceType
    parent_id: str
    dependencies: Tuple[str, ...] = field(default_factory=tuple)
    affected_files: Tuple[str, ...] = field(default_factory=tuple)
    required_symbols: Tuple[str, ...] = field(default_factory=tuple)
    priority: int = 50
    provenance: TaskProvenance = field(default_factory=lambda: TaskProvenance(origin_type="DYNAMIC_EXPANSION"))

    def compute_deterministic_id(self, graph_id: str) -> str:
        """Computes deterministic node ID based on graph, title, parent, and affected files."""
        hasher = hashlib.sha256()
        hasher.update(graph_id.encode("utf-8"))
        hasher.update(self.title.encode("utf-8"))
        hasher.update(self.parent_id.encode("utf-8"))
        hasher.update(",".join(sorted(self.affected_files)).encode("utf-8"))
        return f"{graph_id}_dyn_{hasher.hexdigest()[:10]}"


@dataclass
class GoalVerificationResult:
    """
    Structured outcome of the goal-level verification gate.
    Prevents assuming success merely because all scheduled subtasks terminated.
    """
    verdict: GoalVerificationVerdict
    score: float  # 0.0 to 1.0 confidence score
    satisfied_requirements: List[str] = field(default_factory=list)
    unsatisfied_requirements: List[str] = field(default_factory=list)
    unresolved_unknowns: List[str] = field(default_factory=list)
    suggested_followups: List[DynamicSubtaskRequest] = field(default_factory=list)
    justification: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "verdict": self.verdict.value,
            "score": round(self.score, 4),
            "satisfied_requirements": list(self.satisfied_requirements),
            "unsatisfied_requirements": list(self.unsatisfied_requirements),
            "unresolved_unknowns": list(self.unresolved_unknowns),
            "suggested_followups": [f.title for f in self.suggested_followups],
            "justification": self.justification,
        }
