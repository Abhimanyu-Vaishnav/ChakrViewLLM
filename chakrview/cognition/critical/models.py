"""
Critical Thinking Models and Primitives for ChakrView (Step 23).

Explicitly represents:
1. Hypothesis
2. Evidence
3. Assumption
4. CounterEvidence
5. AlternativeExplanation
6. Contradiction
7. VerificationResult
8. CriticalThinkingTrace

Architectural Guarantees:
- DATA != AUTHORITY, REASONING != AUTHORITY, THINKING != AUTHORITY
- No fabricated evidence: if unavailable, evidence_available = False
- No artificial counter-evidence: if unavailable, status = NOT_AVAILABLE
- Absence of evidence is NEVER converted into evidence
"""

from dataclasses import dataclass, field, asdict
from enum import Enum
import time
from typing import Dict, List, Optional, Any
import uuid


class EvidenceStatus(str, Enum):
    """Availability and verification status of evidence."""
    AVAILABLE = "AVAILABLE"
    UNAVAILABLE = "UNAVAILABLE"
    UNVERIFIED = "UNVERIFIED"
    VERIFIED = "VERIFIED"


class CounterEvidenceStatus(str, Enum):
    """Evaluation status of counter-evidence search."""
    NOT_AVAILABLE = "NOT_AVAILABLE"      # Search conducted; no counter-evidence found/available
    NONE_FOUND = "NONE_FOUND"            # Evaluated and none discovered
    IDENTIFIED = "IDENTIFIED"            # Potential counter-evidence identified
    CONFIRMED = "CONFIRMED"              # Counter-evidence verified and corroborating contradiction
    REFUTED = "REFUTED"                  # Counter-evidence tested and disproved


class HypothesisStatus(str, Enum):
    """Epistemic status of candidate hypothesis."""
    CANDIDATE = "CANDIDATE"              # Newly generated, unexamined
    TESTED = "TESTED"                    # Evaluated against available evidence
    SUPPORTED = "SUPPORTED"              # High empirical/logical backing, no fatal contradiction
    CHALLENGED = "CHALLENGED"            # Active counter-evidence or severe assumption fragility
    REFUTED = "REFUTED"                  # Decisively disproven by counter-evidence or contradiction
    UNCERTAIN = "UNCERTAIN"              # Insufficient evidence; UNKNOWN != FALSE


class ContradictionSeverity(str, Enum):
    """Impact severity of an identified contradiction."""
    LOW = "LOW"                          # Minor tension, resolvable without thesis collapse
    MEDIUM = "MEDIUM"                    # Substantial conflict requiring hypothesis revision
    HIGH = "HIGH"                        # Severe logical or factual contradiction
    FATAL = "FATAL"                      # Absolute mutual exclusion, invalidating hypothesis


class AssumptionCriticality(str, Enum):
    """Importance of an underlying assumption."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


@dataclass
class Assumption:
    """
    Explicitly extracted working premise or presupposition.
    """
    assumption_id: str = field(default_factory=lambda: f"asm_{uuid.uuid4().hex[:8]}")
    statement: str = ""
    is_explicit: bool = True
    plausibility: float = 0.5
    criticality: AssumptionCriticality = AssumptionCriticality.MEDIUM
    falsifiable: bool = True
    falsification_condition: str = ""
    validated: bool = False
    validation_notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "assumption_id": self.assumption_id,
            "statement": self.statement,
            "is_explicit": self.is_explicit,
            "plausibility": self.plausibility,
            "criticality": self.criticality.value,
            "falsifiable": self.falsifiable,
            "falsification_condition": self.falsification_condition,
            "validated": self.validated,
            "validation_notes": self.validation_notes,
        }


@dataclass
class Evidence:
    """
    Strongly typed evidence unit with source provenance and availability flag.
    """
    evidence_id: str = field(default_factory=lambda: f"ev_{uuid.uuid4().hex[:8]}")
    content: str = ""
    source: str = "unknown"
    reliability: float = 0.8
    is_verified: bool = False
    evidence_available: bool = True  # Strict: False if evidence cannot be retrieved
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "content": self.content,
            "source": self.source,
            "reliability": self.reliability,
            "is_verified": self.is_verified,
            "evidence_available": self.evidence_available,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


@dataclass
class CounterEvidence:
    """
    Explicit challenge to a target hypothesis.
    """
    counter_id: str = field(default_factory=lambda: f"cnt_{uuid.uuid4().hex[:8]}")
    target_hypothesis_id: str = ""
    content: str = ""
    source: str = "critical_search"
    strength: float = 0.7
    status: CounterEvidenceStatus = CounterEvidenceStatus.NOT_AVAILABLE
    falsifies_target: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "counter_id": self.counter_id,
            "target_hypothesis_id": self.target_hypothesis_id,
            "content": self.content,
            "source": self.source,
            "strength": self.strength,
            "status": self.status.value,
            "falsifies_target": self.falsifies_target,
            "metadata": self.metadata,
        }


@dataclass
class AlternativeExplanation:
    """
    Viable alternative account formulated to counteract confirmation bias.
    """
    alt_id: str = field(default_factory=lambda: f"alt_{uuid.uuid4().hex[:8]}")
    target_hypothesis_id: str = ""
    explanation: str = ""
    plausibility: float = 0.5
    distinguishing_tests: List[str] = field(default_factory=list)
    evidence_requirements: List[str] = field(default_factory=list)
    contradictions_with_target: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alt_id": self.alt_id,
            "target_hypothesis_id": self.target_hypothesis_id,
            "explanation": self.explanation,
            "plausibility": self.plausibility,
            "distinguishing_tests": list(self.distinguishing_tests),
            "evidence_requirements": list(self.evidence_requirements),
            "contradictions_with_target": list(self.contradictions_with_target),
        }


@dataclass
class Contradiction:
    """
    Detected logical or empirical tension between two propositions.
    """
    contradiction_id: str = field(default_factory=lambda: f"con_{uuid.uuid4().hex[:8]}")
    source_a_id: str = ""
    source_b_id: str = ""
    description: str = ""
    severity: ContradictionSeverity = ContradictionSeverity.MEDIUM
    resolved: bool = False
    resolution_notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "contradiction_id": self.contradiction_id,
            "source_a_id": self.source_a_id,
            "source_b_id": self.source_b_id,
            "description": self.description,
            "severity": self.severity.value,
            "resolved": self.resolved,
            "resolution_notes": self.resolution_notes,
        }


@dataclass
class Hypothesis:
    """
    Candidate conclusion or explanation evaluated through critical examination.
    """
    hypothesis_id: str = field(default_factory=lambda: f"hyp_{uuid.uuid4().hex[:8]}")
    statement: str = ""
    prior_plausibility: float = 0.5
    assumptions: List[Assumption] = field(default_factory=list)
    supporting_evidence_ids: List[str] = field(default_factory=list)
    contradicting_evidence_ids: List[str] = field(default_factory=list)
    alternative_explanation_ids: List[str] = field(default_factory=list)
    missing_information: List[str] = field(default_factory=list)
    status: HypothesisStatus = HypothesisStatus.CANDIDATE
    confidence: float = 0.5
    supporting_requirements: List[str] = field(default_factory=list)
    contradicting_requirements: List[str] = field(default_factory=list)
    verification_requirements: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "statement": self.statement,
            "prior_plausibility": self.prior_plausibility,
            "assumptions": [a.to_dict() for a in self.assumptions],
            "supporting_evidence_ids": list(self.supporting_evidence_ids),
            "contradicting_evidence_ids": list(self.contradicting_evidence_ids),
            "alternative_explanation_ids": list(self.alternative_explanation_ids),
            "missing_information": list(self.missing_information),
            "status": self.status.value,
            "confidence": self.confidence,
            "supporting_requirements": list(self.supporting_requirements),
            "contradicting_requirements": list(self.contradicting_requirements),
            "verification_requirements": list(self.verification_requirements),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": self.metadata,
        }


@dataclass
class VerificationResult:
    """
    Outcome of rigorous critical verification.
    """
    verified: bool = False
    passed_checks: List[str] = field(default_factory=list)
    failed_checks: List[str] = field(default_factory=list)
    contradictions_detected: int = 0
    assumptions_validated: bool = False
    counter_evidence_status: CounterEvidenceStatus = CounterEvidenceStatus.NOT_AVAILABLE
    summary: str = ""
    confidence_score: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "verified": self.verified,
            "passed_checks": list(self.passed_checks),
            "failed_checks": list(self.failed_checks),
            "contradictions_detected": self.contradictions_detected,
            "assumptions_validated": self.assumptions_validated,
            "counter_evidence_status": self.counter_evidence_status.value,
            "summary": self.summary,
            "confidence_score": self.confidence_score,
            "details": self.details,
        }


@dataclass
class CriticalThinkingTrace:
    """
    Auditable, transparent trace of a critical thinking workflow execution.
    """
    trace_id: str = field(default_factory=lambda: f"ct_trace_{uuid.uuid4().hex[:8]}")
    question: str = ""
    hypotheses: List[Hypothesis] = field(default_factory=list)
    evidence_pool: List[Evidence] = field(default_factory=list)
    assumptions: List[Assumption] = field(default_factory=list)
    counter_evidence: List[CounterEvidence] = field(default_factory=list)
    alternatives: List[AlternativeExplanation] = field(default_factory=list)
    contradictions: List[Contradiction] = field(default_factory=list)
    verification_results: List[VerificationResult] = field(default_factory=list)
    decision: Optional[str] = None
    uncertainty_acknowledged: bool = False
    step_records: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    execution_time_ms: float = 0.0
    safe_summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "question": self.question,
            "hypotheses": [h.to_dict() for h in self.hypotheses],
            "evidence_pool": [e.to_dict() for e in self.evidence_pool],
            "assumptions": [a.to_dict() for a in self.assumptions],
            "counter_evidence": [c.to_dict() for c in self.counter_evidence],
            "alternatives": [alt.to_dict() for alt in self.alternatives],
            "contradictions": [con.to_dict() for con in self.contradictions],
            "verification_results": [v.to_dict() for v in self.verification_results],
            "decision": self.decision,
            "uncertainty_acknowledged": self.uncertainty_acknowledged,
            "step_records": self.step_records,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
            "execution_time_ms": self.execution_time_ms,
            "safe_summary": self.safe_summary,
        }


# Aliases for cross-module compatibility
CriticalHypothesis = Hypothesis
CriticalEvidence = Evidence
CriticalContradiction = Contradiction
CriticalVerificationResult = VerificationResult
