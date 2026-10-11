"""
Evidence Model and Evidence Store for ChakrView Reasoning (Step 19).

Distinguishes between FACT, OBSERVATION, MEMORY, RETRIEVED_KNOWLEDGE, CAPABILITY_RESULT,
USER_ASSERTION, HYPOTHESIS, INFERENCE, and ASSUMPTION with rigorous provenance tracking.
Integrates directly with Step 18 Epistemic Knowledge State.
"""

from dataclasses import dataclass, field
from enum import Enum
import time
from typing import Dict, List, Optional, Any, Union
import uuid

from chakrview.state.epistemic import KnowledgeAssertion, EpistemicStatus


class EvidenceType(str, Enum):
    """
    Formal categorization of evidence sources.
    Different evidence types have different baseline reliability and authority.
    """
    FACT = "FACT"                                  # Verified axiom, ground truth mathematical definition
    OBSERVATION = "OBSERVATION"                    # Verified runtime telemetry, sensor read, system state
    MEMORY = "MEMORY"                              # Retrieved persistent personal or conversational memory
    RETRIEVED_KNOWLEDGE = "RETRIEVED_KNOWLEDGE"    # Retrieved document chunk / external corpus snippet
    CAPABILITY_RESULT = "CAPABILITY_RESULT"        # Result returned from executed capability provider
    USER_ASSERTION = "USER_ASSERTION"              # Statement provided by user prompt or instruction
    HYPOTHESIS = "HYPOTHESIS"                      # Candidate explanation formed during reasoning
    INFERENCE = "INFERENCE"                        # Derived conclusion from prior premises
    ASSUMPTION = "ASSUMPTION"                      # Working premise without verified backing evidence


# Baseline default reliabilities reflecting epistemic strength
DEFAULT_RELIABILITIES: Dict[EvidenceType, float] = {
    EvidenceType.FACT: 1.0,
    EvidenceType.OBSERVATION: 0.95,
    EvidenceType.MEMORY: 0.85,
    EvidenceType.RETRIEVED_KNOWLEDGE: 0.80,
    EvidenceType.CAPABILITY_RESULT: 0.85,
    EvidenceType.USER_ASSERTION: 0.70,
    EvidenceType.INFERENCE: 0.75,
    EvidenceType.HYPOTHESIS: 0.50,
    EvidenceType.ASSUMPTION: 0.40,
}


@dataclass
class EvidenceItem:
    """
    Strongly typed evidence unit with cryptographic/source provenance.
    """
    evidence_id: str = field(default_factory=lambda: f"ev_{uuid.uuid4().hex[:8]}")
    evidence_type: EvidenceType = EvidenceType.OBSERVATION
    source_type: str = "state"
    source_id: str = "system"
    content: Any = ""
    confidence: float = 1.0
    reliability: float = 0.9
    timestamp: float = field(default_factory=time.time)
    provenance: Dict[str, Any] = field(default_factory=dict)
    is_untrusted: bool = False

    def __post_init__(self) -> None:
        # Constrain scores to [0.0, 1.0]
        self.confidence = max(0.0, min(1.0, float(self.confidence)))
        self.reliability = max(0.0, min(1.0, float(self.reliability)))

    def effective_weight(self) -> float:
        """Composite weight combining confidence and intrinsic reliability."""
        return self.confidence * self.reliability

    def to_dict(self) -> Dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "evidence_type": self.evidence_type.value,
            "source_type": self.source_type,
            "source_id": self.source_id,
            "content": self.content,
            "confidence": self.confidence,
            "reliability": self.reliability,
            "timestamp": self.timestamp,
            "provenance": dict(self.provenance),
            "is_untrusted": self.is_untrusted,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvidenceItem":
        d = dict(data)
        d["evidence_type"] = EvidenceType(d["evidence_type"])
        return cls(**d)

    @classmethod
    def from_knowledge_assertion(cls, assertion: KnowledgeAssertion) -> "EvidenceItem":
        """Factory from Step 18 KnowledgeAssertion."""
        # Map epistemic status to evidence type
        status = getattr(assertion, "status", getattr(assertion, "epistemic_status", EpistemicStatus.KNOWN))
        if status == EpistemicStatus.KNOWN:
            ev_type = EvidenceType.FACT if assertion.confidence >= 0.99 else EvidenceType.OBSERVATION
        elif status == EpistemicStatus.UNCERTAIN:
            ev_type = EvidenceType.ASSUMPTION
        elif status in (EpistemicStatus.CONFLICTING, EpistemicStatus.STALE):
            ev_type = EvidenceType.ASSUMPTION
        else:
            ev_type = EvidenceType.OBSERVATION

        reliability = DEFAULT_RELIABILITIES.get(ev_type, 0.8)
        content_repr = f"{assertion.subject} {assertion.predicate} {assertion.value}"

        return cls(
            evidence_id=f"ev_ka_{assertion.assertion_id[:8]}",
            evidence_type=ev_type,
            source_type="epistemic_state",
            source_id=assertion.source,
            content=content_repr,
            confidence=assertion.confidence,
            reliability=reliability,
            timestamp=assertion.timestamp,
            provenance={"assertion_id": assertion.assertion_id, **assertion.provenance},
            is_untrusted=False,
        )

    @classmethod
    def from_capability_result(cls, result: Any, capability_id: str) -> "EvidenceItem":
        """Factory from Step 17 CapabilityResult."""
        output_content = getattr(result, "output", str(result))
        is_untrusted = getattr(result, "is_untrusted", False)
        return cls(
            evidence_id=f"ev_cap_{uuid.uuid4().hex[:8]}",
            evidence_type=EvidenceType.CAPABILITY_RESULT,
            source_type="capability",
            source_id=capability_id,
            content=output_content,
            confidence=0.9 if getattr(result, "success", True) else 0.2,
            reliability=DEFAULT_RELIABILITIES[EvidenceType.CAPABILITY_RESULT],
            timestamp=time.time(),
            provenance={
                "capability_id": capability_id,
                "execution_time_ms": getattr(result, "execution_time_ms", 0.0),
            },
            is_untrusted=is_untrusted,
        )


class EvidenceStore:
    """
    Bounded in-memory collection of evidence for a reasoning session.
    """

    def __init__(self, max_items: int = 50) -> None:
        self.max_items = max_items
        self._items: Dict[str, EvidenceItem] = {}

    def add(self, item: EvidenceItem) -> bool:
        """Add an evidence item subject to capacity limits."""
        if len(self._items) >= self.max_items and item.evidence_id not in self._items:
            # Capacity reached: if new item has lower weight than worst existing, reject it
            worst_id = min(self._items.keys(), key=lambda k: self._items[k].effective_weight())
            if item.effective_weight() <= self._items[worst_id].effective_weight():
                return False
            # Evict worst
            del self._items[worst_id]

        self._items[item.evidence_id] = item
        return True

    def get(self, evidence_id: str) -> Optional[EvidenceItem]:
        return self._items.get(evidence_id)

    def all_items(self) -> List[EvidenceItem]:
        return list(self._items.values())

    def filter(
        self,
        evidence_type: Optional[EvidenceType] = None,
        min_confidence: float = 0.0,
        min_reliability: float = 0.0,
    ) -> List[EvidenceItem]:
        results = []
        for item in self._items.values():
            if evidence_type is not None and item.evidence_type != evidence_type:
                continue
            if item.confidence < min_confidence:
                continue
            if item.reliability < min_reliability:
                continue
            results.append(item)
        return results

    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_items": self.max_items,
            "items": [item.to_dict() for item in self._items.values()],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvidenceStore":
        store = cls(max_items=data.get("max_items", 50))
        for item_data in data.get("items", []):
            store.add(EvidenceItem.from_dict(item_data))
        return store
