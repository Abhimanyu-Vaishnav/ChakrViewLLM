"""
Memory & Continual Cognition Data Models for ChakrView (Step 24).

Defines strongly typed, serializable structures for:
- Explicit Provenance Sources: USER_PROVIDED, SYSTEM_OBSERVED, CAPABILITY_RESULT,
  REASONING_DERIVED, CRITICAL_THINKING_DERIVED, RETRIEVED_SOURCE, TRAINING_APPROVED, UNKNOWN
- Explicit Verification States: CANDIDATE, UNVERIFIED, VERIFIED, CONTRADICTED,
  QUARANTINED, ARCHIVED, REJECTED
- Memory Lifecycle States: ACTIVE, ARCHIVED, EXPIRED, QUARANTINED, DELETED
- Contradiction States: UNRESOLVED, UNDER_REVIEW, RESOLVED, PERSISTENT_CONFLICT
- Structured Episodic Experience (Episode)
- Versioned Semantic Memory (SemanticMemory)
- Structured Contradiction Record (MemoryContradiction)
- Retrieval Query and Ranked Candidates

Architectural Boundaries:
- DATA != AUTHORITY
- MEMORY != AUTHORITY
- EXPERIENCE != AUTHORITY
- PROVENANCE != AUTHORITY
Memory may provide evidence/context, but never independently authorizes actions.
"""

from dataclasses import dataclass, field, asdict
from enum import Enum
import hashlib
import time
from typing import Dict, List, Optional, Any


class MemoryProvenanceSource(str, Enum):
    """
    Formal taxonomy of memory origin citations.
    PROVENANCE != AUTHORITY: Provenance indicates origin, never capability authorization.
    """
    USER_PROVIDED = "USER_PROVIDED"
    SYSTEM_OBSERVED = "SYSTEM_OBSERVED"
    CAPABILITY_RESULT = "CAPABILITY_RESULT"
    REASONING_DERIVED = "REASONING_DERIVED"
    CRITICAL_THINKING_DERIVED = "CRITICAL_THINKING_DERIVED"
    RETRIEVED_SOURCE = "RETRIEVED_SOURCE"
    TRAINING_APPROVED = "TRAINING_APPROVED"
    UNKNOWN = "UNKNOWN"


class MemoryVerificationState(str, Enum):
    """
    Formal verification states for persistent memory records.
    CANDIDATE is never treated as VERIFIED.
    QUARANTINED or REJECTED memories are blocked from trusted retrieval.
    """
    CANDIDATE = "CANDIDATE"             # Newly consolidated/observed, unvetted
    UNVERIFIED = "UNVERIFIED"           # Ingestion complete, pending formal check
    VERIFIED = "VERIFIED"               # Formally verified against ground truth or policy
    CONTRADICTED = "CONTRADICTED"       # Conflicting assertion exists
    QUARANTINED = "QUARANTINED"         # Suspected unsafe, injection, or corrupted
    ARCHIVED = "ARCHIVED"               # Retired from active working view
    REJECTED = "REJECTED"               # Refuted by evidence or verification


class MemoryLifecycleStatus(str, Enum):
    """Lifecycle retention state of a memory record."""
    ACTIVE = "ACTIVE"
    ARCHIVED = "ARCHIVED"
    EXPIRED = "EXPIRED"
    QUARANTINED = "QUARANTINED"
    DELETED = "DELETED"


class ContradictionResolutionState(str, Enum):
    """Resolution state of a detected contradiction across memories."""
    UNRESOLVED = "UNRESOLVED"
    UNDER_REVIEW = "UNDER_REVIEW"
    RESOLVED = "RESOLVED"
    PERSISTENT_CONFLICT = "PERSISTENT_CONFLICT"


@dataclass
class Episode:
    """
    Structured record of an experienced event or task interaction.
    Preserves what actually happened (situation, action/response, outcome)
    separately from subsequent interpretations or synthesized facts.
    """
    episode_id: str
    tenant_id: str
    session_id: str
    situation: str
    action_or_response: str
    outcome: str
    timestamp: float = field(default_factory=time.time)
    task_id: Optional[str] = None
    evidence_refs: List[str] = field(default_factory=list)
    confidence: float = 0.8
    provenance: MemoryProvenanceSource = MemoryProvenanceSource.SYSTEM_OBSERVED
    verification_status: MemoryVerificationState = MemoryVerificationState.UNVERIFIED
    lifecycle_status: MemoryLifecycleStatus = MemoryLifecycleStatus.ACTIVE
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence must be in [0.0, 1.0], got {self.confidence}")
        if not self.tenant_id:
            raise ValueError("tenant_id cannot be empty.")
        if not self.session_id:
            raise ValueError("session_id cannot be empty.")
        if not self.situation.strip():
            raise ValueError("situation cannot be empty.")

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["provenance"] = self.provenance.value
        data["verification_status"] = self.verification_status.value
        data["lifecycle_status"] = self.lifecycle_status.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Episode":
        d = dict(data)
        if "provenance" in d:
            d["provenance"] = MemoryProvenanceSource(d["provenance"])
        if "verification_status" in d:
            d["verification_status"] = MemoryVerificationState(d["verification_status"])
        if "lifecycle_status" in d:
            d["lifecycle_status"] = MemoryLifecycleStatus(d["lifecycle_status"])
        return cls(**d)


@dataclass
class SemanticMemory:
    """
    Versioned, structured declarative knowledge extracted from verified experiences.
    Does not destructively overwrite prior information; preserves revision versions.
    """
    memory_id: str
    tenant_id: str
    subject: str
    predicate: str
    object_value: str
    session_id: str = "default_session"
    provenance: MemoryProvenanceSource = MemoryProvenanceSource.SYSTEM_OBSERVED
    confidence: float = 0.8
    verification_status: MemoryVerificationState = MemoryVerificationState.CANDIDATE
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    version: int = 1
    previous_version_id: Optional[str] = None
    contradiction_refs: List[str] = field(default_factory=list)
    lifecycle_status: MemoryLifecycleStatus = MemoryLifecycleStatus.ACTIVE
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence must be in [0.0, 1.0], got {self.confidence}")
        if not self.tenant_id:
            raise ValueError("tenant_id cannot be empty.")
        if not self.subject.strip():
            raise ValueError("subject cannot be empty.")
        if not self.predicate.strip():
            raise ValueError("predicate cannot be empty.")
        if not self.object_value.strip():
            raise ValueError("object_value cannot be empty.")

    @property
    def statement(self) -> str:
        """Formatted subject-predicate-object proposition."""
        return f"{self.subject} {self.predicate} {self.object_value}"

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["provenance"] = self.provenance.value
        data["verification_status"] = self.verification_status.value
        data["lifecycle_status"] = self.lifecycle_status.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SemanticMemory":
        d = dict(data)
        if "provenance" in d:
            d["provenance"] = MemoryProvenanceSource(d["provenance"])
        if "verification_status" in d:
            d["verification_status"] = MemoryVerificationState(d["verification_status"])
        if "lifecycle_status" in d:
            d["lifecycle_status"] = MemoryLifecycleStatus(d["lifecycle_status"])
        return cls(**d)


@dataclass
class MemoryContradiction:
    """
    Representation of an explicit factual contradiction between memories.
    Prevents silent overwrite and maintains auditability.
    """
    contradiction_id: str
    tenant_id: str
    conflicting_memory_ids: List[str]
    contradiction_type: str = "attribute_conflict"
    severity: str = "MEDIUM"  # LOW, MEDIUM, HIGH, CRITICAL
    provenance: MemoryProvenanceSource = MemoryProvenanceSource.SYSTEM_OBSERVED
    timestamp: float = field(default_factory=time.time)
    resolution_state: ContradictionResolutionState = ContradictionResolutionState.UNRESOLVED
    resolution_evidence: List[str] = field(default_factory=list)
    resolved_memory_id: Optional[str] = None
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["provenance"] = self.provenance.value
        data["resolution_state"] = self.resolution_state.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryContradiction":
        d = dict(data)
        if "provenance" in d:
            d["provenance"] = MemoryProvenanceSource(d["provenance"])
        if "resolution_state" in d:
            d["resolution_state"] = ContradictionResolutionState(d["resolution_state"])
        return cls(**d)


@dataclass
class MemoryRetrievalQuery:
    """
    Bounded, tenant-isolated memory search query.
    """
    query_text: str
    tenant_id: str
    session_id: Optional[str] = None
    top_k: int = 5
    min_confidence: float = 0.0
    min_score: float = 0.0
    allowed_types: Optional[List[str]] = None
    trusted_only: bool = True  # Rejects CANDIDATE, QUARANTINED, REJECTED
    include_cross_session: bool = False


@dataclass
class MemoryRetrievalCandidate:
    """
    Scored, explained retrieval result for memory recall.
    """
    memory_id: str
    memory_type: str
    content: str
    score: float
    relevance_score: float
    confidence: float
    verification_factor: float
    recency_factor: float
    contradiction_factor: float
    provenance: MemoryProvenanceSource
    verification_status: MemoryVerificationState
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["provenance"] = self.provenance.value
        data["verification_status"] = self.verification_status.value
        return data


@dataclass
class MemoryRetrievalResult:
    """
    Complete response containing ranked candidates and execution audit metrics.
    """
    query: str
    tenant_id: str
    candidates: List[MemoryRetrievalCandidate]
    total_scanned: int
    latency_ms: float
    budget_used: int
    execution_profile: str = "STANDARD"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "tenant_id": self.tenant_id,
            "candidates": [c.to_dict() for c in self.candidates],
            "total_scanned": self.total_scanned,
            "latency_ms": self.latency_ms,
            "budget_used": self.budget_used,
            "execution_profile": self.execution_profile,
        }
