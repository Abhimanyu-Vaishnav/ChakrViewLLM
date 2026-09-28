"""
Contradiction Detection and Evaluation for ChakrView Reasoning (Step 19).

Integrates with Step 16 memory conflict logic and Step 18 Epistemic Knowledge State.
Explicitly identifies and tracks contradictions between evidence items without
arbitrarily overwriting or discarding unresolved discrepancies.
"""

from dataclasses import dataclass, field
from enum import Enum
import time
from typing import Dict, List, Optional, Any, Tuple
import uuid

from chakrview.reasoning.evidence import EvidenceItem, EvidenceStore, EvidenceType


class ContradictionSeverity(str, Enum):
    """Impact severity of an evidence contradiction."""
    MINOR = "MINOR"                # Minor numerical variance or descriptive difference
    SIGNIFICANT = "SIGNIFICANT"    # Conflicting factual assertions on the same subject
    CRITICAL = "CRITICAL"          # Mutually exclusive axioms or safety-relevant contradictions


class ContradictionStatus(str, Enum):
    """Resolution status of a contradiction."""
    UNRESOLVED = "UNRESOLVED"
    RESOLVED_SOURCE_A = "RESOLVED_SOURCE_A"
    RESOLVED_SOURCE_B = "RESOLVED_SOURCE_B"
    COMPROMISE = "COMPROMISE"
    PERSISTENT_UNCERTAINTY = "PERSISTENT_UNCERTAINTY"


@dataclass
class Contradiction:
    """
    Formally documented conflict between two pieces of evidence.
    """
    contradiction_id: str = field(default_factory=lambda: f"contra_{uuid.uuid4().hex[:8]}")
    evidence_a_id: str = ""
    evidence_b_id: str = ""
    topic: str = ""
    severity: ContradictionSeverity = ContradictionSeverity.SIGNIFICANT
    status: ContradictionStatus = ContradictionStatus.UNRESOLVED
    evaluated_factors: Dict[str, Any] = field(default_factory=dict)
    resolution_rationale: Optional[str] = None
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "contradiction_id": self.contradiction_id,
            "evidence_a_id": self.evidence_a_id,
            "evidence_b_id": self.evidence_b_id,
            "topic": self.topic,
            "severity": self.severity.value,
            "status": self.status.value,
            "evaluated_factors": dict(self.evaluated_factors),
            "resolution_rationale": self.resolution_rationale,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Contradiction":
        d = dict(data)
        d["severity"] = ContradictionSeverity(d["severity"])
        d["status"] = ContradictionStatus(d["status"])
        return cls(**d)


class ContradictionDetector:
    """
    Detects semantic or attribute contradictions across evidence items.
    """

    def detect_contradiction(
        self,
        ev_a: EvidenceItem,
        ev_b: EvidenceItem,
        topic: str = "",
    ) -> Optional[Contradiction]:
        """
        Check if two evidence items conflict on a given topic or content assertion.
        """
        # If either content explicitly asserts contradiction or opposing polarity
        content_a = str(ev_a.content).strip().lower()
        content_b = str(ev_b.content).strip().lower()

        is_conflict = False
        severity = ContradictionSeverity.SIGNIFICANT

        # Simple semantic oppositions
        opposites = [
            ("yes", "no"),
            ("true", "false"),
            ("open", "closed"),
            ("enabled", "disabled"),
            ("available", "unavailable"),
            ("safe", "dangerous"),
            ("success", "failure"),
        ]
        for term_a, term_b in opposites:
            if (term_a in content_a and term_b in content_b) or (term_b in content_a and term_a in content_b):
                is_conflict = True
                if term_a in ("safe", "enabled"):
                    severity = ContradictionSeverity.CRITICAL
                break

        # Check negation
        if not is_conflict:
            if (f"not {content_b}" in content_a) or (f"not {content_a}" in content_b):
                is_conflict = True

        if not is_conflict:
            return None

        # Infer topic from shared tokens if not explicitly provided
        inferred_topic = topic
        if not inferred_topic:
            parts_a = content_a.split()
            parts_b = content_b.split()
            common = [w for w in parts_a if w in parts_b and len(w) > 2]
            inferred_topic = common[0] if common else "proposition"

        # Contradiction detected
        return Contradiction(
            evidence_a_id=ev_a.evidence_id,
            evidence_b_id=ev_b.evidence_id,
            topic=inferred_topic,
            severity=severity,
            status=ContradictionStatus.UNRESOLVED,
            evaluated_factors={
                "weight_a": ev_a.effective_weight(),
                "weight_b": ev_b.effective_weight(),
                "type_a": ev_a.evidence_type.value,
                "type_b": ev_b.evidence_type.value,
                "timestamp_a": ev_a.timestamp,
                "timestamp_b": ev_b.timestamp,
            },
        )

    def evaluate_resolution(
        self,
        contradiction: Contradiction,
        evidence_store: EvidenceStore,
    ) -> Contradiction:
        """
        Evaluate provenance, reliability, recency, and corroboration to attempt resolution.
        Crucial: Never silently pick an answer if evidence is genuinely conflicting!
        """
        ev_a = evidence_store.get(contradiction.evidence_a_id)
        ev_b = evidence_store.get(contradiction.evidence_b_id)

        if not ev_a or not ev_b:
            contradiction.status = ContradictionStatus.UNRESOLVED
            contradiction.resolution_rationale = "Missing evidence items for resolution."
            return contradiction

        w_a = ev_a.effective_weight()
        w_b = ev_b.effective_weight()

        # Check for clear hierarchy (e.g. FACT vs ASSUMPTION)
        diff = abs(w_a - w_b)
        if diff >= 0.35:
            if w_a > w_b:
                contradiction.status = ContradictionStatus.RESOLVED_SOURCE_A
                contradiction.resolution_rationale = (
                    f"Resolved in favor of {ev_a.evidence_type.value} (weight={w_a:.2f}) "
                    f"over {ev_b.evidence_type.value} (weight={w_b:.2f})."
                )
            else:
                contradiction.status = ContradictionStatus.RESOLVED_SOURCE_B
                contradiction.resolution_rationale = (
                    f"Resolved in favor of {ev_b.evidence_type.value} (weight={w_b:.2f}) "
                    f"over {ev_a.evidence_type.value} (weight={w_a:.2f})."
                )
        elif abs(ev_a.timestamp - ev_b.timestamp) > 3600 and ev_a.evidence_type == ev_b.evidence_type:
            # Temporal recency if weights are comparable and same evidence type
            if ev_a.timestamp > ev_b.timestamp:
                contradiction.status = ContradictionStatus.RESOLVED_SOURCE_A
                contradiction.resolution_rationale = "Resolved via temporal recency for matching evidence types."
            else:
                contradiction.status = ContradictionStatus.RESOLVED_SOURCE_B
                contradiction.resolution_rationale = "Resolved via temporal recency for matching evidence types."
        else:
            # Genuine conflicting evidence without decisive factor -> MUST remain uncertain!
            contradiction.status = ContradictionStatus.PERSISTENT_UNCERTAINTY
            contradiction.resolution_rationale = (
                f"Conflicting evidence weights are closely matched ({w_a:.2f} vs {w_b:.2f}); "
                "preserving epistemic uncertainty rather than fabricating resolution."
            )

        return contradiction
