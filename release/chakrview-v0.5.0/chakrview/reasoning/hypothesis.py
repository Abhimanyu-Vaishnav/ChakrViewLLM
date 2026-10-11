"""
Hypothesis Model and Hypothesis Engine for ChakrView Reasoning (Step 19).

Represents candidate explanations with supporting and contradicting evidence,
preserving the fundamental epistemic rule UNKNOWN != FALSE.
"""

from dataclasses import dataclass, field
from enum import Enum
import time
from typing import Dict, List, Optional, Any
import uuid

from chakrview.reasoning.evidence import EvidenceItem, EvidenceStore


class HypothesisStatus(str, Enum):
    """Evaluation status of a candidate hypothesis."""
    SUPPORTED = "SUPPORTED"          # Significant supporting evidence, no substantial contradictions
    PLAUSIBLE = "PLAUSIBLE"          # Consistent with knowledge but limited corroboration
    UNCERTAIN = "UNCERTAIN"          # Conflicting or insufficient evidence; UNKNOWN != FALSE
    CONTRADICTED = "CONTRADICTED"    # Evidence directly disputes hypothesis premises
    REJECTED = "REJECTED"            # Decisively refuted by verified facts or observations


@dataclass
class Hypothesis:
    """
    Structured hypothesis tracking evidence, assumptions, confidence, and revision lineage.
    """
    hypothesis_id: str = field(default_factory=lambda: f"hyp_{uuid.uuid4().hex[:8]}")
    statement: str = ""
    supporting_evidence_ids: List[str] = field(default_factory=list)
    contradicting_evidence_ids: List[str] = field(default_factory=list)
    confidence: float = 0.5
    assumptions: List[str] = field(default_factory=list)
    status: HypothesisStatus = HypothesisStatus.PLAUSIBLE
    revision_history: List[Dict[str, Any]] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def revise(
        self,
        new_statement: str,
        reason: str,
        new_supporting_ids: Optional[List[str]] = None,
        new_contradicting_ids: Optional[List[str]] = None,
    ) -> None:
        """Record revision while preserving historical audit lineage."""
        self.revision_history.append({
            "previous_statement": self.statement,
            "previous_status": self.status.value,
            "previous_confidence": self.confidence,
            "reason": reason,
            "timestamp": time.time(),
        })
        self.statement = new_statement
        if new_supporting_ids is not None:
            self.supporting_evidence_ids = list(new_supporting_ids)
        if new_contradicting_ids is not None:
            self.contradicting_evidence_ids = list(new_contradicting_ids)
        self.updated_at = time.time()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "statement": self.statement,
            "supporting_evidence_ids": list(self.supporting_evidence_ids),
            "contradicting_evidence_ids": list(self.contradicting_evidence_ids),
            "confidence": self.confidence,
            "assumptions": list(self.assumptions),
            "status": self.status.value,
            "revision_history": list(self.revision_history),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Hypothesis":
        d = dict(data)
        d["status"] = HypothesisStatus(d["status"])
        return cls(**d)


class HypothesisEngine:
    """
    Evaluates, ranks, and updates hypotheses against an evidence store.
    """

    def evaluate_hypothesis(
        self,
        hypothesis: Hypothesis,
        evidence_store: EvidenceStore,
    ) -> Hypothesis:
        """
        Evaluate hypothesis support, contradiction, status, and confidence.
        Preserves UNKNOWN != FALSE: lack of evidence leaves status UNCERTAIN / PLAUSIBLE.
        """
        support_weight = 0.0
        contradict_weight = 0.0

        for ev_id in hypothesis.supporting_evidence_ids:
            ev = evidence_store.get(ev_id)
            if ev is not None:
                support_weight += ev.effective_weight()

        for ev_id in hypothesis.contradicting_evidence_ids:
            ev = evidence_store.get(ev_id)
            if ev is not None:
                contradict_weight += ev.effective_weight()

        total_weight = support_weight + contradict_weight

        if total_weight == 0.0:
            # Epistemic rule: No evidence does NOT mean false!
            hypothesis.status = HypothesisStatus.UNCERTAIN if hypothesis.assumptions else HypothesisStatus.PLAUSIBLE
            hypothesis.confidence = 0.5
        elif contradict_weight > 0 and support_weight == 0.0:
            if contradict_weight >= 1.5:
                hypothesis.status = HypothesisStatus.REJECTED
                hypothesis.confidence = 0.05
            else:
                hypothesis.status = HypothesisStatus.CONTRADICTED
                hypothesis.confidence = max(0.1, 0.5 - 0.3 * contradict_weight)
        elif support_weight > 0 and contradict_weight == 0.0:
            hypothesis.status = HypothesisStatus.SUPPORTED
            hypothesis.confidence = min(0.98, 0.5 + 0.25 * support_weight)
        else:
            # Mixed evidence
            ratio = support_weight / (total_weight + 1e-6)
            hypothesis.confidence = ratio
            if ratio >= 0.75:
                hypothesis.status = HypothesisStatus.SUPPORTED
            elif ratio <= 0.25:
                hypothesis.status = HypothesisStatus.CONTRADICTED
            else:
                hypothesis.status = HypothesisStatus.UNCERTAIN

        hypothesis.updated_at = time.time()
        return hypothesis
