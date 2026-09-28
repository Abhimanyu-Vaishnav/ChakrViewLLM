"""
Thinking Workspace for ChakrView (Step 21).

Maintains the bounded, structured working state of active deliberation:
- Deterministic, inspectable, and JSON-serializable.
- Multi-tenant isolation enforced by owner_id and session_id.
- Explicit operational bounds (max steps, max revisions, max hypotheses, timeout).
- Audit-safe: records all thought steps, critiques, hypotheses, and revisions.
"""

from dataclasses import dataclass, field, asdict
import time
from typing import Dict, List, Optional, Any, Tuple
import uuid

from chakrview.thinking.thought import ThoughtStep, ThoughtPurpose
from chakrview.thinking.policy import ThinkingPolicy, get_standard_policy


class WorkspaceBudgetExceededError(RuntimeError):
    """Raised when an operation attempts to exceed configured workspace capacity."""
    pass


class TenantIsolationError(PermissionError):
    """Raised when cross-tenant workspace access or tampering is attempted."""
    pass


@dataclass
class ThinkingWorkspace:
    """
    State container for bounded, multi-cycle deliberation.
    """
    workspace_id: str = field(default_factory=lambda: f"ws_{uuid.uuid4().hex[:10]}")
    task_id: str = "task_default"
    owner_id: str = "default_user"
    session_id: str = "default_session"
    objective: str = ""
    policy: ThinkingPolicy = field(default_factory=get_standard_policy)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    # Active deliberation state
    thought_steps: List[ThoughtStep] = field(default_factory=list)
    hypotheses: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    unresolved_questions: List[str] = field(default_factory=list)
    candidate_approaches: List[Dict[str, Any]] = field(default_factory=list)
    current_working_conclusion: Optional[str] = None
    detected_weaknesses: List[str] = field(default_factory=list)
    critiques: List[Dict[str, Any]] = field(default_factory=list)
    revisions: List[Dict[str, Any]] = field(default_factory=list)
    uncertainty_state: Dict[str, Any] = field(default_factory=dict)
    stopping_state: Optional[str] = None
    is_concluded: bool = False

    def validate_tenant(self, owner_id: str, session_id: Optional[str] = None) -> None:
        """Enforce tenant isolation on workspace access."""
        if self.owner_id != owner_id:
            raise TenantIsolationError(
                f"Multi-tenant violation: Access denied to workspace owned by '{self.owner_id}' "
                f"for requester '{owner_id}'."
            )
        if session_id is not None and self.session_id != session_id:
            raise TenantIsolationError(
                f"Session isolation violation: Workspace session '{self.session_id}' "
                f"does not match request session '{session_id}'."
            )

    @property
    def elapsed_time_ms(self) -> float:
        """Elapsed time since workspace creation in milliseconds."""
        return (time.time() - self.created_at) * 1000.0

    @property
    def step_count(self) -> int:
        return len(self.thought_steps)

    @property
    def revision_count(self) -> int:
        return len(self.revisions)

    def can_revise(self) -> bool:
        """Check whether another revision cycle is permitted under active policy."""
        return not self.is_concluded and self.revision_count < self.policy.max_revision_cycles

    def can_add_thought(self) -> bool:
        """Check whether an additional thought step fits within budget."""
        return not self.is_concluded and self.step_count < self.policy.max_thought_steps

    def can_deliberate(self) -> Tuple[bool, Optional[str]]:
        """
        Check whether additional deliberation steps are permitted under active policy.
        """
        if self.is_concluded:
            return False, "Workspace already marked concluded."
        if self.step_count >= self.policy.max_thought_steps:
            return False, f"Maximum thought steps ({self.policy.max_thought_steps}) reached."
        if self.revision_count > self.policy.max_revision_cycles:
            return False, f"Maximum revision cycles ({self.policy.max_revision_cycles}) exceeded."
        if self.elapsed_time_ms >= self.policy.max_deliberation_time_ms:
            return False, f"Deliberation timeout ({self.policy.max_deliberation_time_ms} ms) exceeded."
        return True, None

    def add_thought_step(
        self,
        purpose: ThoughtPurpose,
        content: str,
        input_references: Optional[List[str]] = None,
        hypothesis_reference: Optional[str] = None,
        evidence_references: Optional[List[str]] = None,
        candidate_action: Optional[str] = None,
        result: Optional[Any] = None,
        uncertainty: Optional[float] = None,
        confidence_available: bool = False,
        verification_state: str = "UNVERIFIED",
        provenance: Optional[Dict[str, Any]] = None,
    ) -> ThoughtStep:
        """
        Append an immutable thought step to the workspace.
        """
        allowed, reason = self.can_deliberate()
        if not allowed and purpose != ThoughtPurpose.STOP:
            raise WorkspaceBudgetExceededError(f"Cannot add thought step: {reason}")

        step = ThoughtStep(
            thought_id=f"th_{uuid.uuid4().hex[:10]}",
            step_index=len(self.thought_steps) + 1,
            purpose=purpose,
            content=content,
            input_references=input_references or [],
            hypothesis_reference=hypothesis_reference,
            evidence_references=evidence_references or [],
            candidate_action=candidate_action,
            result=result,
            uncertainty=uncertainty,
            confidence_available=confidence_available,
            verification_state=verification_state,
            provenance=provenance or {},
            timestamp=time.time(),
        )
        self.thought_steps.append(step)
        self.updated_at = time.time()
        return step

    def record_hypothesis(
        self,
        statement: str,
        hypothesis_id: Optional[str] = None,
        supporting_evidence: Optional[List[str]] = None,
        confidence: float = 0.5,
    ) -> str:
        """Register or update an active hypothesis."""
        if len(self.hypotheses) >= self.policy.max_hypotheses and hypothesis_id not in self.hypotheses:
            raise WorkspaceBudgetExceededError(
                f"Maximum hypotheses ({self.policy.max_hypotheses}) reached in workspace."
            )
        h_id = hypothesis_id or f"hyp_{uuid.uuid4().hex[:8]}"
        self.hypotheses[h_id] = {
            "hypothesis_id": h_id,
            "statement": statement,
            "supporting_evidence": supporting_evidence or [],
            "confidence": confidence,
            "timestamp": time.time(),
        }
        self.updated_at = time.time()
        return h_id

    def add_evidence(
        self,
        content: str,
        source: str = "deliberation",
        evidence_id: Optional[str] = None,
        confidence: float = 0.8,
    ) -> str:
        """Record an evidence item in the workspace."""
        if len(self.evidence) >= self.policy.max_evidence_items:
            # Drop oldest evidence if ceiling reached
            self.evidence.pop(0)

        ev_id = evidence_id or f"ev_{uuid.uuid4().hex[:8]}"
        self.evidence.append({
            "evidence_id": ev_id,
            "content": content,
            "source": source,
            "confidence": confidence,
            "timestamp": time.time(),
        })
        self.updated_at = time.time()
        return ev_id

    def record_weakness(self, weakness: str) -> None:
        """Note a detected weakness in the current working solution."""
        if weakness not in self.detected_weaknesses:
            self.detected_weaknesses.append(weakness)
        self.updated_at = time.time()

    def record_revision(self, reason: str, direction: str) -> int:
        """Record an explicit revision cycle."""
        rev_record = {
            "revision_index": len(self.revisions) + 1,
            "reason": reason,
            "direction": direction,
            "prior_conclusion": self.current_working_conclusion,
            "timestamp": time.time(),
        }
        self.revisions.append(rev_record)
        self.updated_at = time.time()
        return len(self.revisions)

    def set_conclusion(self, conclusion: str) -> None:
        """Update current working conclusion."""
        self.current_working_conclusion = conclusion
        self.updated_at = time.time()

    def mark_stopped(self, condition: str, reason: str) -> None:
        """Conclude deliberation with terminal condition."""
        self.stopping_state = condition
        self.is_concluded = True
        self.updated_at = time.time()
        self.add_thought_step(
            purpose=ThoughtPurpose.STOP,
            content=f"Deliberation concluded with condition [{condition}]: {reason}",
            result=self.current_working_conclusion,
            verification_state="CONCLUDED",
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize workspace state cleanly to dictionary."""
        return {
            "workspace_id": self.workspace_id,
            "task_id": self.task_id,
            "owner_id": self.owner_id,
            "session_id": self.session_id,
            "objective": self.objective,
            "policy": self.policy.to_dict(),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "step_count": len(self.thought_steps),
            "revision_count": len(self.revisions),
            "thought_steps": [s.to_dict() for s in self.thought_steps],
            "hypotheses": self.hypotheses,
            "evidence": self.evidence,
            "unresolved_questions": self.unresolved_questions,
            "candidate_approaches": self.candidate_approaches,
            "current_working_conclusion": self.current_working_conclusion,
            "detected_weaknesses": self.detected_weaknesses,
            "critiques": self.critiques,
            "revisions": self.revisions,
            "uncertainty_state": self.uncertainty_state,
            "stopping_state": self.stopping_state,
            "is_concluded": self.is_concluded,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ThinkingWorkspace":
        d = dict(data)
        policy_data = d.get("policy", {})
        policy = ThinkingPolicy.from_dict(policy_data) if isinstance(policy_data, dict) else policy_data
        steps = [ThoughtStep.from_dict(s) for s in d.get("thought_steps", [])]

        ws = cls(
            workspace_id=d.get("workspace_id", f"ws_{uuid.uuid4().hex[:10]}"),
            task_id=d.get("task_id", "task_default"),
            owner_id=d.get("owner_id", "default_user"),
            session_id=d.get("session_id", "default_session"),
            objective=d.get("objective", ""),
            policy=policy,
            created_at=d.get("created_at", time.time()),
            updated_at=d.get("updated_at", time.time()),
            hypotheses=d.get("hypotheses", {}),
            evidence=d.get("evidence", []),
            unresolved_questions=d.get("unresolved_questions", []),
            candidate_approaches=d.get("candidate_approaches", []),
            current_working_conclusion=d.get("current_working_conclusion"),
            detected_weaknesses=d.get("detected_weaknesses", []),
            critiques=d.get("critiques", []),
            revisions=d.get("revisions", []),
            uncertainty_state=d.get("uncertainty_state", {}),
            stopping_state=d.get("stopping_state"),
            is_concluded=d.get("is_concluded", False),
        )
        ws.thought_steps = steps
        return ws
