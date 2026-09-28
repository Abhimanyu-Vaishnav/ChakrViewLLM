"""
Strongly Typed Data Models for Federated Cognition (Step 26).

Defines foundational contracts for cooperative multi-agent intelligence:
- Agent Identity, Roles, Capabilities, Status & Contracts
- Typed Inter-Agent Message Protocol & Envelope Validation
- Federated Tasks, Subproblems & Delegation Limits
- Conflict Records, Evidence Accounting & Synthesis Candidates
- Sanitized Public Federated Audit Traces

CRITICAL ARCHITECTURAL AXIOMS:
1. AGENT != AUTHORITY
   No agent can independently authorize external capabilities, model mutation,
   or memory promotion.
2. STRICT ISOLATION:
   All agents, messages, tasks, and memory queries are tenant- and session-scoped.
3. IMMUTABILITY:
   Runtime multi-agent coordination never updates model parameters.
"""

from dataclasses import dataclass, field, asdict
from enum import Enum
import hashlib
import json
import time
from typing import Dict, List, Optional, Any, Set
import uuid

from chakrview.cognition.unified.models import DecisionState


class AgentRole(str, Enum):
    """Sovereign cognitive role assigned to a logical agent."""
    ANALYST = "ANALYST"
    RESEARCHER = "RESEARCHER"
    CRITIC = "CRITIC"
    PLANNER = "PLANNER"
    SYNTHESIZER = "SYNTHESIZER"
    VERIFIER = "VERIFIER"
    OBSERVER = "OBSERVER"


class AgentStatus(str, Enum):
    """Operational lifecycle status of a registered agent."""
    READY = "READY"
    BUSY = "BUSY"
    UNAVAILABLE = "UNAVAILABLE"
    FAILED = "FAILED"
    QUARANTINED = "QUARANTINED"


class AgentCapability(str, Enum):
    """Specific bounded capability domain granted to an agent role."""
    REASONING = "REASONING"
    MEMORY_QUERY = "MEMORY_QUERY"
    CRITICAL_EVALUATION = "CRITICAL_EVALUATION"
    PLANNING = "PLANNING"
    SYNTHESIS = "SYNTHESIS"
    VERIFICATION = "VERIFICATION"
    OBSERVATION = "OBSERVATION"


class MessageType(str, Enum):
    """Typed inter-agent message category."""
    TASK_REQUEST = "TASK_REQUEST"
    TASK_RESPONSE = "TASK_RESPONSE"
    CRITIQUE_REQUEST = "CRITIQUE_REQUEST"
    CRITIQUE_RESPONSE = "CRITIQUE_RESPONSE"
    CONFLICT_NOTIFICATION = "CONFLICT_NOTIFICATION"
    SYNTHESIS_PROPOSAL = "SYNTHESIS_PROPOSAL"
    HEARTBEAT = "HEARTBEAT"
    ERROR = "ERROR"


class MessagePriority(str, Enum):
    """Priority level for message scheduling within cooperative rounds."""
    LOW = "LOW"
    NORMAL = "NORMAL"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class ConflictState(str, Enum):
    """Evaluation status of a detected inter-agent disagreement."""
    AGREEMENT = "AGREEMENT"
    PARTIAL_AGREEMENT = "PARTIAL_AGREEMENT"
    CONFLICT = "CONFLICT"
    UNRESOLVED = "UNRESOLVED"
    RESOLVED = "RESOLVED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


# ============================================================================
# 1. Agent Identity & Contract
# ============================================================================

@dataclass
class AgentIdentity:
    """
    Deterministic identity container for a logical cognitive agent.
    """
    agent_id: str
    role: AgentRole
    tenant_id: str
    session_id: str
    capabilities: List[AgentCapability] = field(default_factory=list)
    version: str = "1.0.0"
    status: AgentStatus = AgentStatus.READY
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "role": self.role.value,
            "tenant_id": self.tenant_id,
            "session_id": self.session_id,
            "capabilities": [c.value for c in self.capabilities],
            "version": self.version,
            "status": self.status.value,
            "created_at": self.created_at,
        }


@dataclass
class AgentContract:
    """
    Formal operational specification and execution boundary for an agent.
    """
    identity: AgentIdentity
    input_types: List[str] = field(default_factory=lambda: ["text", "structured_task"])
    output_types: List[str] = field(default_factory=lambda: ["evidence", "analysis", "synthesis"])
    max_execution_time_ms: float = 5000.0
    max_tokens_budget: int = 512
    requires_gate_for_capabilities: bool = True

    def validate(self) -> None:
        """Validate structural integrity of agent contract."""
        if not self.identity.agent_id:
            raise ValueError("Agent ID cannot be empty.")
        if not self.identity.tenant_id:
            raise ValueError("Tenant ID cannot be empty.")
        if self.max_tokens_budget > 512:
            raise ValueError(f"Agent token budget ({self.max_tokens_budget}) exceeds ChakrMicro context ceiling (512).")
        if not self.requires_gate_for_capabilities:
            raise PermissionError("All federated agents MUST require CapabilityGate authorization.")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "identity": self.identity.to_dict(),
            "input_types": list(self.input_types),
            "output_types": list(self.output_types),
            "max_execution_time_ms": self.max_execution_time_ms,
            "max_tokens_budget": self.max_tokens_budget,
            "requires_gate_for_capabilities": self.requires_gate_for_capabilities,
        }


# ============================================================================
# 2. Agent Message Protocol
# ============================================================================

@dataclass
class AgentMessage:
    """
    Typed, tamper-evident message container for inter-agent communication.
    """
    message_id: str
    sender_agent_id: str
    receiver_agent_id: str  # specific agent ID or "BROADCAST"
    tenant_id: str
    session_id: str
    correlation_id: str
    message_type: MessageType
    priority: MessagePriority = MessagePriority.NORMAL
    payload: Dict[str, Any] = field(default_factory=dict)
    provenance: str = "agent_computation"
    timestamp: float = field(default_factory=time.time)
    sequence_number: int = 0
    parent_message_id: Optional[str] = None
    integrity_hash: str = ""

    def __post_init__(self) -> None:
        if not self.integrity_hash:
            self.integrity_hash = self.compute_hash()

    def compute_hash(self) -> str:
        """Compute deterministic SHA-256 fingerprint over message content."""
        canonical_dict = {
            "message_id": self.message_id,
            "sender_agent_id": self.sender_agent_id,
            "receiver_agent_id": self.receiver_agent_id,
            "tenant_id": self.tenant_id,
            "session_id": self.session_id,
            "correlation_id": self.correlation_id,
            "message_type": self.message_type.value,
            "priority": self.priority.value,
            "payload": self.payload,
            "provenance": self.provenance,
            "sequence_number": self.sequence_number,
            "parent_message_id": self.parent_message_id,
        }
        encoded = json.dumps(canonical_dict, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def verify_integrity(self) -> bool:
        """Verify message payload and metadata have not been tampered with."""
        return self.integrity_hash == self.compute_hash()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "message_id": self.message_id,
            "sender_agent_id": self.sender_agent_id,
            "receiver_agent_id": self.receiver_agent_id,
            "tenant_id": self.tenant_id,
            "session_id": self.session_id,
            "correlation_id": self.correlation_id,
            "message_type": self.message_type.value,
            "priority": self.priority.value,
            "payload": self.payload,
            "provenance": self.provenance,
            "timestamp": self.timestamp,
            "sequence_number": self.sequence_number,
            "parent_message_id": self.parent_message_id,
            "integrity_hash": self.integrity_hash,
        }


@dataclass
class MessageEnvelope:
    """
    Validated envelope protecting message routing and provenance tracking.
    """
    envelope_id: str = field(default_factory=lambda: f"env_{uuid.uuid4().hex[:10]}")
    message: AgentMessage = field(default_factory=lambda: None)  # type: ignore
    verified: bool = False
    routed_at: float = field(default_factory=time.time)

    def validate_envelope(self) -> bool:
        if self.message is None:
            return False
        self.verified = self.message.verify_integrity()
        return self.verified


# ============================================================================
# 3. Federated Task & Decomposition
# ============================================================================

@dataclass
class AgentTask:
    """
    Bounded subtask allocated to a specific logical agent role.
    """
    task_id: str
    parent_task_id: Optional[str]
    assigned_role: AgentRole
    assigned_agent_id: Optional[str] = None
    objective: str = ""
    constraints: Dict[str, Any] = field(default_factory=dict)
    required_evidence: List[str] = field(default_factory=list)
    expected_output_type: str = "analysis"
    priority: MessagePriority = MessagePriority.NORMAL
    status: str = "PENDING"  # PENDING, IN_PROGRESS, COMPLETED, FAILED, TIMED_OUT
    depth: int = 0
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    result: Optional[Dict[str, Any]] = None

    def complete(self, result: Dict[str, Any]) -> None:
        self.status = "COMPLETED"
        self.completed_at = time.time()
        self.result = result

    def fail(self, reason: str) -> None:
        self.status = "FAILED"
        self.completed_at = time.time()
        self.result = {"error": reason}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "parent_task_id": self.parent_task_id,
            "assigned_role": self.assigned_role.value,
            "assigned_agent_id": self.assigned_agent_id,
            "objective": self.objective,
            "constraints": self.constraints,
            "required_evidence": self.required_evidence,
            "expected_output_type": self.expected_output_type,
            "priority": self.priority.value,
            "status": self.status,
            "depth": self.depth,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "result": self.result,
        }


# ============================================================================
# 4. Conflicts, Synthesis & Evidence
# ============================================================================

@dataclass
class FederatedConflictRecord:
    """
    Auditable record of a factual or logical conflict across participating agents.
    """
    conflict_id: str
    task_id: str
    conflicting_agent_ids: List[str]
    claims: List[Dict[str, Any]] = field(default_factory=list)
    supporting_evidence: List[Dict[str, Any]] = field(default_factory=list)
    contradicting_evidence: List[Dict[str, Any]] = field(default_factory=list)
    state: ConflictState = ConflictState.UNRESOLVED
    resolution_notes: Optional[str] = None
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "conflict_id": self.conflict_id,
            "task_id": self.task_id,
            "conflicting_agent_ids": list(self.conflicting_agent_ids),
            "claims": self.claims,
            "supporting_evidence": self.supporting_evidence,
            "contradicting_evidence": self.contradicting_evidence,
            "state": self.state.value,
            "resolution_notes": self.resolution_notes,
            "created_at": self.created_at,
        }


@dataclass
class FederatedSynthesisCandidate:
    """
    Cohesive multi-agent synthesis candidate with explicit minority opinion preservation.
    """
    candidate_id: str
    task_id: str
    primary_synthesis: str
    supporting_evidence: List[Dict[str, Any]] = field(default_factory=list)
    unresolved_contradictions: List[Dict[str, Any]] = field(default_factory=list)
    minority_opinions: List[Dict[str, Any]] = field(default_factory=list)
    uncertainty_notes: Optional[str] = None
    recommended_decision_state: DecisionState = DecisionState.ANSWER
    confidence: float = 0.80

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "task_id": self.task_id,
            "primary_synthesis": self.primary_synthesis,
            "supporting_evidence": self.supporting_evidence,
            "unresolved_contradictions": self.unresolved_contradictions,
            "minority_opinions": self.minority_opinions,
            "uncertainty_notes": self.uncertainty_notes,
            "recommended_decision_state": self.recommended_decision_state.value,
            "confidence": self.confidence,
        }


# ============================================================================
# 5. Public Audit Trace
# ============================================================================

@dataclass
class SafePublicFederatedTrace:
    """
    Sanitized public trace of federated cognitive execution.
    Never exposes internal private scratchpads, raw logits, or sensitive tokens.
    """
    trace_id: str
    task_id: str
    tenant_id: str
    session_id: str
    participating_agents: List[Dict[str, str]] = field(default_factory=list)
    execution_rounds: int = 0
    total_messages_exchanged: int = 0
    evidence_items_count: int = 0
    conflicts_detected_count: int = 0
    decision_state: str = "ANSWER"
    latency_ms: float = 0.0
    failures_count: int = 0
    retries_count: int = 0
    hardware_profile: str = "STANDARD"
    trace_fingerprint: str = ""
    status: str = "COMPLETED"
    weights_modified: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "task_id": self.task_id,
            "tenant_id": self.tenant_id,
            "session_id": self.session_id,
            "participating_agents": self.participating_agents,
            "execution_rounds": self.execution_rounds,
            "total_messages_exchanged": self.total_messages_exchanged,
            "evidence_items_count": self.evidence_items_count,
            "conflicts_detected_count": self.conflicts_detected_count,
            "decision_state": self.decision_state,
            "latency_ms": self.latency_ms,
            "failures_count": self.failures_count,
            "retries_count": self.retries_count,
            "hardware_profile": self.hardware_profile,
            "trace_fingerprint": self.trace_fingerprint,
            "status": self.status,
            "weights_modified": self.weights_modified,
        }
