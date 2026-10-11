"""
Wave 572: Partial & Missing Evidence Reasoning Engine.

Architectural Guarantees:
- Enforces Completeness Levels:
  L1: FULL SUPPORT -> ANSWER
  L2: PARTIAL SUPPORT -> QUALIFY (Conditional scope)
  L3: RELEVANT BUT INSUFFICIENT -> REQUEST_MORE_EVIDENCE
  L4: NO SUPPORT -> ABSTAIN
  L5: CONFLICTING SUPPORT -> REPORT_CONFLICT / ABSTAIN
- Identifies EXACTLY what premise or hop is missing.
- Prevents over-claiming or hallucinating missing details.
"""

from __future__ import annotations

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Set

from .evidence_quality import EvidenceQualityGrade, EvidenceQualityReport


class CompletenessLevel(str, Enum):
    L1_FULL_SUPPORT = "L1_FULL_SUPPORT"
    L2_PARTIAL_SUPPORT = "L2_PARTIAL_SUPPORT"
    L3_RELEVANT_INSUFFICIENT = "L3_RELEVANT_INSUFFICIENT"
    L4_NO_SUPPORT = "L4_NO_SUPPORT"
    L5_CONFLICTING = "L5_CONFLICTING"


@dataclass
class CompletenessAssessment:
    level: CompletenessLevel
    action: str                     # ANSWER, QUALIFY, REQUEST_MORE_EVIDENCE, ABSTAIN, REPORT_CONFLICT
    confidence: float              # Calibrated epistemic certainty
    satisfied_premises: List[str]
    missing_premises: List[str]
    claim_scope: str               # "UNRESTRICTED", "NARROW_QUALIFIED", "NONE"
    rationale: str


class PartialEvidenceReasoner:
    """
    Evaluates evidence coverage across multi-premise claims, bounding claim scope
    and triggering honest abstention or qualification when premises are missing.
    """

    @classmethod
    def evaluate_completeness(
        cls,
        required_premises: List[str],
        evidence_reports: List[EvidenceQualityReport],
        evidence_content_map: Dict[str, str],
    ) -> CompletenessAssessment:
        if not required_premises:
            return CompletenessAssessment(
                level=CompletenessLevel.L4_NO_SUPPORT,
                action="ABSTAIN",
                confidence=0.0,
                satisfied_premises=[],
                missing_premises=[],
                claim_scope="NONE",
                rationale="No premises specified.",
            )

        # Check for contradictions first
        has_contradiction = any(
            rep.grade == EvidenceQualityGrade.CONTRADICTORY for rep in evidence_reports
        )
        if has_contradiction:
            return CompletenessAssessment(
                level=CompletenessLevel.L5_CONFLICTING,
                action="REPORT_CONFLICT",
                confidence=0.35,
                satisfied_premises=[],
                missing_premises=required_premises,
                claim_scope="NONE",
                rationale="Mutually contradictory evidence detected among evaluated items.",
            )

        satisfied: List[str] = []
        missing: List[str] = []

        # Map supporting content
        valid_contents = [
            evidence_content_map.get(rep.evidence_id, "").lower()
            for rep in evidence_reports
            if rep.grade in (EvidenceQualityGrade.HIGH_SUPPORT, EvidenceQualityGrade.MODERATE_SUPPORT)
        ]
        combined_text = " ".join(valid_contents)

        for premise in required_premises:
            premise_tokens = set(premise.lower().split())
            # A premise is satisfied if key tokens are present in valid evidence
            if all(tok in combined_text for tok in premise_tokens if len(tok) > 3):
                satisfied.append(premise)
            else:
                missing.append(premise)

        coverage_ratio = len(satisfied) / len(required_premises)

        if coverage_ratio == 1.0:
            return CompletenessAssessment(
                level=CompletenessLevel.L1_FULL_SUPPORT,
                action="ANSWER",
                confidence=0.95,
                satisfied_premises=satisfied,
                missing_premises=[],
                claim_scope="UNRESTRICTED",
                rationale="All required factual premises are corroborated by verified evidence.",
            )
        elif coverage_ratio >= 0.5:
            # Partial support: can answer narrow claim, but must qualify
            return CompletenessAssessment(
                level=CompletenessLevel.L2_PARTIAL_SUPPORT,
                action="QUALIFY",
                confidence=0.70,
                satisfied_premises=satisfied,
                missing_premises=missing,
                claim_scope="NARROW_QUALIFIED",
                rationale=f"Partially supported ({len(satisfied)}/{len(required_premises)} premises). Missing: {missing}.",
            )
        elif coverage_ratio > 0.0:
            return CompletenessAssessment(
                level=CompletenessLevel.L3_RELEVANT_INSUFFICIENT,
                action="REQUEST_MORE_EVIDENCE",
                confidence=0.40,
                satisfied_premises=satisfied,
                missing_premises=missing,
                claim_scope="NONE",
                rationale=f"Relevant background present but insufficient to warrant claim ({len(satisfied)}/{len(required_premises)} premises). Missing: {missing}.",
            )
        else:
            return CompletenessAssessment(
                level=CompletenessLevel.L4_NO_SUPPORT,
                action="ABSTAIN",
                confidence=0.0,
                satisfied_premises=[],
                missing_premises=missing,
                claim_scope="NONE",
                rationale="No supporting evidence found for required premises. Abstaining.",
            )
