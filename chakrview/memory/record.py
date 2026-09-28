"""
Strongly Typed Memory Record Model for ChakrView (Step 16).

Provides structured, serializable persistent memory records supporting:
- Memory Taxonomy: Episodic, Semantic, User/Profile, Working, Knowledge
- Explicit Validity States: Active, Superseded, Expired, Archived
- Granular Provenance: Source tracking, user scope, session binding
- Dual-Metric Scoring: Independent Importance vs Confidence
- Temporal Modeling: Validity windows, revision numbering, supersession links
"""

from dataclasses import dataclass, field, asdict
from enum import Enum
import time
from typing import Dict, List, Optional, Any


class MemoryType(str, Enum):
    """Taxonomy of persistent personal and cognitive memory."""
    EPISODIC = "episodic"         # Past events, completed tasks, conversation transcripts, reports
    SEMANTIC = "semantic"         # Stable user/domain facts (e.g., "Client X operates 3 factories")
    USER_PROFILE = "user_profile" # Durable user preferences, formatting, style, professional context
    WORKING = "working"           # Temporary task/session context bridge
    KNOWLEDGE = "knowledge"       # Grounded external/domain knowledge references


class MemoryValidity(str, Enum):
    """Explicit lifecycle validity states for memory records."""
    ACTIVE = "active"             # Currently valid and searchable
    SUPERSEDED = "superseded"     # Overridden by a newer version or confirmed revision
    EXPIRED = "expired"           # Temporal validity window has ended
    ARCHIVED = "archived"         # Manually or automatically retired from primary search


@dataclass
class MemoryProvenance:
    """
    Fine-grained provenance tracking where information originated.
    """
    source_type: str = "conversation"  # conversation, document, explicit_user, task_result, artifact
    source_id: Optional[str] = None    # turn_id, doc_id, task_id, artifact_id
    session_id: Optional[str] = None
    user_id: str = "default_user"
    extracted_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryProvenance":
        return cls(**data)


@dataclass
class TemporalMetadata:
    """
    Temporal boundaries, version tracking, and validity spans.
    """
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    valid_from: Optional[float] = None
    valid_until: Optional[float] = None
    superseded_by: Optional[str] = None
    version: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TemporalMetadata":
        return cls(**data)


@dataclass
class MemoryRecord:
    """
    Core persistent memory record in ChakrView.

    Attributes:
        memory_id: Unique record identifier.
        memory_type: Category of memory (episodic, semantic, user_profile, etc.).
        content: The actual memory text or declarative proposition.
        owner_id: User/owner identifier enforcing strict multi-user privacy isolation.
        validity: Current lifecycle state (ACTIVE, SUPERSEDED, EXPIRED, ARCHIVED).
        importance: Degree of future utility/significance in [0.0, 1.0].
        confidence: Degree of certainty that information is factually correct in [0.0, 1.0].
        provenance: Fine-grained origin citation.
        temporal: Creation, expiration, and revision history.
        tags: Categorical tags for indexing and filtering.
        embedding: Optional normalized embedding vector (e.g. 128-dim from Step 14 encoder).
        metadata: Domain and custom extensible attributes.
    """
    memory_id: str
    memory_type: MemoryType
    content: str
    owner_id: str = "default_user"
    validity: MemoryValidity = MemoryValidity.ACTIVE
    importance: float = 0.5
    confidence: float = 0.8
    provenance: MemoryProvenance = field(default_factory=MemoryProvenance)
    temporal: TemporalMetadata = field(default_factory=TemporalMetadata)
    tags: List[str] = field(default_factory=list)
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not (0.0 <= self.importance <= 1.0):
            raise ValueError(f"importance must be in [0.0, 1.0], got {self.importance}")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence must be in [0.0, 1.0], got {self.confidence}")
        if not self.content or not self.content.strip():
            raise ValueError("content cannot be empty.")
        if not self.owner_id:
            raise ValueError("owner_id cannot be empty.")

    def is_valid_at(self, timestamp: Optional[float] = None) -> bool:
        """Check if memory is temporally valid at a given timestamp (default: now)."""
        if self.validity != MemoryValidity.ACTIVE:
            return False
        ts = time.time() if timestamp is None else timestamp
        if self.temporal.valid_from is not None and ts < self.temporal.valid_from:
            return False
        if self.temporal.valid_until is not None and ts > self.temporal.valid_until:
            return False
        return True

    def mark_superseded(self, newer_memory_id: str) -> None:
        """Mark this record as superseded by a newer memory."""
        self.validity = MemoryValidity.SUPERSEDED
        self.temporal.superseded_by = newer_memory_id
        self.temporal.updated_at = time.time()

    def mark_expired(self) -> None:
        """Mark this record as temporally expired."""
        self.validity = MemoryValidity.EXPIRED
        self.temporal.updated_at = time.time()

    def mark_archived(self) -> None:
        """Mark this record as archived."""
        self.validity = MemoryValidity.ARCHIVED
        self.temporal.updated_at = time.time()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize memory record to structured dictionary."""
        return {
            "memory_id": self.memory_id,
            "memory_type": self.memory_type.value,
            "content": self.content,
            "owner_id": self.owner_id,
            "validity": self.validity.value,
            "importance": self.importance,
            "confidence": self.confidence,
            "provenance": self.provenance.to_dict(),
            "temporal": self.temporal.to_dict(),
            "tags": self.tags,
            "embedding": self.embedding,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryRecord":
        """Deserialize memory record from dictionary."""
        d = dict(data)
        d["memory_type"] = MemoryType(d["memory_type"])
        d["validity"] = MemoryValidity(d.get("validity", "active"))
        if isinstance(d.get("provenance"), dict):
            d["provenance"] = MemoryProvenance.from_dict(d["provenance"])
        if isinstance(d.get("temporal"), dict):
            d["temporal"] = TemporalMetadata.from_dict(d["temporal"])
        return cls(**d)
