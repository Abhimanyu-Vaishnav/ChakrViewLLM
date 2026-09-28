"""
Epistemic Knowledge State Representation for ChakrView (Step 18).

Provides formal representation of knowledge assertions and their epistemic status:
- KNOWN: Verified fact backed by reliable evidence.
- UNKNOWN: Explicitly acknowledged gap; system knows that it does NOT know.
- UNCERTAIN: Proposition with weak, noisy, or unvalidated evidence.
- CONFLICTING: Multiple contradictory assertions exist in memory or knowledge.
- STALE: Previously verified information that has expired or been superseded.
- UNAVAILABLE: Source, sensor, or subsystem is unreachable or disconnected.

CRITICAL ARCHITECTURAL DISTINCTIONS:
- UNKNOWN != FALSE
  (e.g., "User is a tax consultant = UNKNOWN" does NOT mean "User is NOT a tax consultant").
- UNAVAILABLE != UNKNOWN
  (e.g., "Current room temperature = UNAVAILABLE" means sensor is disconnected;
   the true temperature exists in reality, but access is blocked).
- DISABLED != UNAVAILABLE
  (e.g., A capability shut down by policy vs a failed hardware connection).
"""

from dataclasses import dataclass, field, asdict
from enum import Enum
import time
from typing import Dict, List, Optional, Any, Tuple
import uuid


class EpistemicStatus(str, Enum):
    """
    Formal epistemic qualification of a knowledge assertion.
    """
    KNOWN = "KNOWN"
    UNKNOWN = "UNKNOWN"
    UNCERTAIN = "UNCERTAIN"
    CONFLICTING = "CONFLICTING"
    STALE = "STALE"
    UNAVAILABLE = "UNAVAILABLE"


@dataclass
class KnowledgeAssertion:
    """
    Strongly typed factual assertion with explicit epistemic qualification and provenance.
    """
    subject: str
    predicate: str
    value: Any
    assertion_id: str = field(default_factory=lambda: f"assert_{uuid.uuid4().hex[:8]}")
    status: EpistemicStatus = EpistemicStatus.KNOWN
    confidence: float = 1.0
    source: str = "system"
    source_id: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    valid_until: Optional[float] = None
    evidence_refs: List[str] = field(default_factory=list)
    provenance: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"Confidence must be in [0.0, 1.0], got {self.confidence}")

    def is_valid_at(self, ts: Optional[float] = None) -> bool:
        """Check whether assertion remains temporally valid at timestamp ts."""
        check_time = ts if ts is not None else time.time()
        if self.valid_until is not None and check_time > self.valid_until:
            return False
        return self.status not in (EpistemicStatus.STALE, EpistemicStatus.UNAVAILABLE)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KnowledgeAssertion":
        d = dict(data)
        d["status"] = EpistemicStatus(d["status"])
        return cls(**d)


@dataclass
class KnowledgeState:
    """
    Inspectable knowledge state tracking active factual assertions.
    """
    assertions: Dict[str, KnowledgeAssertion] = field(default_factory=dict)

    def assert_fact(
        self,
        subject: str,
        predicate: str,
        value: Any,
        status: EpistemicStatus = EpistemicStatus.KNOWN,
        confidence: float = 1.0,
        source: str = "system",
        source_id: Optional[str] = None,
        valid_until: Optional[float] = None,
        evidence_refs: Optional[List[str]] = None,
        provenance: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> KnowledgeAssertion:
        """
        Record or update a knowledge assertion in the state layer.
        """
        assertion = KnowledgeAssertion(
            subject=subject,
            predicate=predicate,
            value=value,
            status=status,
            confidence=confidence,
            source=source,
            source_id=source_id,
            valid_until=valid_until,
            evidence_refs=evidence_refs or [],
            provenance=provenance or {},
            metadata=metadata or {},
        )
        self.assertions[assertion.assertion_id] = assertion
        return assertion

    def get_assertion(self, assertion_id: str) -> Optional[KnowledgeAssertion]:
        """Retrieve assertion by ID."""
        return self.assertions.get(assertion_id)

    def query(
        self,
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
        status: Optional[EpistemicStatus] = None,
    ) -> List[KnowledgeAssertion]:
        """
        Query assertions matching given filters.
        """
        results: List[KnowledgeAssertion] = []
        for a in self.assertions.values():
            if subject is not None and a.subject != subject:
                continue
            if predicate is not None and a.predicate != predicate:
                continue
            if status is not None and a.status != status:
                continue
            results.append(a)
        return results

    def mark_stale(self, assertion_id: str) -> None:
        """Mark an assertion as STALE."""
        if assertion_id in self.assertions:
            self.assertions[assertion_id].status = EpistemicStatus.STALE

    def mark_conflicting(self, assertion_ids: List[str]) -> None:
        """Mark multiple assertions as CONFLICTING."""
        for aid in assertion_ids:
            if aid in self.assertions:
                self.assertions[aid].status = EpistemicStatus.CONFLICTING

    def to_dict(self) -> Dict[str, Any]:
        return {
            "assertions": {k: v.to_dict() for k, v in self.assertions.items()}
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KnowledgeState":
        assertions = {
            k: KnowledgeAssertion.from_dict(v)
            for k, v in data.get("assertions", {}).items()
        }
        return cls(assertions=assertions)
