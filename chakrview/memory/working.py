"""
Working Memory Subsystem for ChakrView Continual Cognition (Step 24).

Maintains bounded short-term cognitive state for active tasks:
- current objective
- active context
- current hypotheses
- relevant evidence
- intermediate decisions
- active constraints
- pending questions
- recent observations

CRITICAL BOUNDARY RULE:
Working memory is strictly bounded by the active execution policy.
Excess items are evicted deterministically without unboundedly consuming host RAM.
"""

from dataclasses import dataclass, field, asdict
import time
from typing import Dict, List, Optional, Any


@dataclass
class WorkingMemoryConfig:
    """Bounded capacity limits for working memory."""
    max_context_items: int = 15
    max_hypotheses: int = 5
    max_evidence_items: int = 10
    max_decisions: int = 10
    max_constraints: int = 10
    max_pending_questions: int = 5
    max_observations: int = 15


class WorkingMemory:
    """
    Session- and task-scoped bounded working memory workspace.
    """

    def __init__(
        self,
        tenant_id: str,
        session_id: str,
        config: Optional[WorkingMemoryConfig] = None,
    ) -> None:
        self.tenant_id = tenant_id
        self.session_id = session_id
        self.config = config or WorkingMemoryConfig()

        self.objective: str = ""
        self.active_context: List[str] = []
        self.current_hypotheses: List[Dict[str, Any]] = []
        self.relevant_evidence: List[Dict[str, Any]] = []
        self.intermediate_decisions: List[Dict[str, Any]] = []
        self.active_constraints: List[str] = []
        self.pending_questions: List[str] = []
        self.recent_observations: List[str] = []

        self.eviction_count: int = 0
        self.created_at: float = time.time()
        self.updated_at: float = time.time()

    def set_objective(self, objective: str) -> None:
        """Set the current primary task objective."""
        self.objective = objective.strip()
        self.updated_at = time.time()

    def add_context(self, text: str) -> None:
        """Add context item with bounded FIFO eviction."""
        if not text or not text.strip():
            return
        if len(self.active_context) >= self.config.max_context_items:
            self.active_context.pop(0)
            self.eviction_count += 1
        self.active_context.append(text.strip())
        self.updated_at = time.time()

    def add_hypothesis(self, hypothesis_data: Dict[str, Any]) -> None:
        """Add hypothesis with bounded FIFO eviction."""
        if len(self.current_hypotheses) >= self.config.max_hypotheses:
            self.current_hypotheses.pop(0)
            self.eviction_count += 1
        self.current_hypotheses.append(hypothesis_data)
        self.updated_at = time.time()

    def add_evidence(self, evidence_data: Dict[str, Any]) -> None:
        """Add evidence item with bounded FIFO eviction."""
        if len(self.relevant_evidence) >= self.config.max_evidence_items:
            self.relevant_evidence.pop(0)
            self.eviction_count += 1
        self.relevant_evidence.append(evidence_data)
        self.updated_at = time.time()

    def add_decision(self, decision_data: Dict[str, Any]) -> None:
        """Add decision record with bounded FIFO eviction."""
        if len(self.intermediate_decisions) >= self.config.max_decisions:
            self.intermediate_decisions.pop(0)
            self.eviction_count += 1
        self.intermediate_decisions.append(decision_data)
        self.updated_at = time.time()

    def add_constraint(self, constraint: str) -> None:
        """Add active constraint with deduplication and bounded FIFO eviction."""
        c = constraint.strip()
        if not c or c in self.active_constraints:
            return
        if len(self.active_constraints) >= self.config.max_constraints:
            self.active_constraints.pop(0)
            self.eviction_count += 1
        self.active_constraints.append(c)
        self.updated_at = time.time()

    def add_pending_question(self, question: str) -> None:
        """Add pending question with bounded FIFO eviction."""
        q = question.strip()
        if not q or q in self.pending_questions:
            return
        if len(self.pending_questions) >= self.config.max_pending_questions:
            self.pending_questions.pop(0)
            self.eviction_count += 1
        self.pending_questions.append(q)
        self.updated_at = time.time()

    def add_observation(self, observation: str) -> None:
        """Add recent observation with bounded FIFO eviction."""
        obs = observation.strip()
        if not obs:
            return
        if len(self.recent_observations) >= self.config.max_observations:
            self.recent_observations.pop(0)
            self.eviction_count += 1
        self.recent_observations.append(obs)
        self.updated_at = time.time()

    def clear(self) -> None:
        """Reset all working memory contents for a fresh task."""
        self.objective = ""
        self.active_context.clear()
        self.current_hypotheses.clear()
        self.relevant_evidence.clear()
        self.intermediate_decisions.clear()
        self.active_constraints.clear()
        self.pending_questions.clear()
        self.recent_observations.clear()
        self.updated_at = time.time()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize working memory snapshot."""
        return {
            "tenant_id": self.tenant_id,
            "session_id": self.session_id,
            "objective": self.objective,
            "active_context": list(self.active_context),
            "current_hypotheses": [dict(h) for h in self.current_hypotheses],
            "relevant_evidence": [dict(e) for e in self.relevant_evidence],
            "intermediate_decisions": [dict(d) for d in self.intermediate_decisions],
            "active_constraints": list(self.active_constraints),
            "pending_questions": list(self.pending_questions),
            "recent_observations": list(self.recent_observations),
            "eviction_count": self.eviction_count,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        config: Optional[WorkingMemoryConfig] = None,
    ) -> "WorkingMemory":
        """Deserialize working memory snapshot."""
        wm = cls(
            tenant_id=data.get("tenant_id", "default_tenant"),
            session_id=data.get("session_id", "default_session"),
            config=config,
        )
        wm.objective = data.get("objective", "")
        wm.active_context = list(data.get("active_context", []))
        wm.current_hypotheses = list(data.get("current_hypotheses", []))
        wm.relevant_evidence = list(data.get("relevant_evidence", []))
        wm.intermediate_decisions = list(data.get("intermediate_decisions", []))
        wm.active_constraints = list(data.get("active_constraints", []))
        wm.pending_questions = list(data.get("pending_questions", []))
        wm.recent_observations = list(data.get("recent_observations", []))
        wm.eviction_count = data.get("eviction_count", 0)
        wm.created_at = data.get("created_at", time.time())
        wm.updated_at = data.get("updated_at", time.time())
        return wm
