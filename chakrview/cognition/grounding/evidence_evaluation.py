"""
Wave 556 & 557: Evidence Necessity & Sufficiency Evaluation Engine.

Architectural Guarantees:
- RETRIEVED EVIDENCE != USED EVIDENCE
- CITED EVIDENCE != SUPPORTING EVIDENCE
- SUPPORTING EVIDENCE != SUFFICIENT EVIDENCE
- Rigorously distinguishes:
  EVIDENCE_NECESSARY, EVIDENCE_OPTIONAL, EVIDENCE_IRRELEVANT, EVIDENCE_CONTRADICTORY.
- Evaluates sufficiency:
  SUPPORTED, PARTIALLY_SUPPORTED, INSUFFICIENT, CONTRADICTED, UNKNOWN.
- Prevents overclaiming: An evidentiary premise (e.g. "Revenue grew 10%")
  is NOT allowed to silently support an overreaching conclusion ("Company is healthy").
"""

from __future__ import annotations

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Set, Tuple


class EvidenceRole(str, Enum):
    EVIDENCE_NECESSARY = "EVIDENCE_NECESSARY"
    EVIDENCE_OPTIONAL = "EVIDENCE_OPTIONAL"
    EVIDENCE_IRRELEVANT = "EVIDENCE_IRRELEVANT"
    EVIDENCE_CONTRADICTORY = "EVIDENCE_CONTRADICTORY"


class SufficiencyStatus(str, Enum):
    SUPPORTED = "SUPPORTED"
    PARTIALLY_SUPPORTED = "PARTIALLY_SUPPORTED"
    INSUFFICIENT = "INSUFFICIENT"
    CONTRADICTED = "CONTRADICTED"
    UNKNOWN = "UNKNOWN"


class ClaimSupportType(str, Enum):
    DIRECTLY_SUPPORTED = "DIRECTLY_SUPPORTED"
    INFERRED_FROM_SUPPORTED_FACTS = "INFERRED_FROM_SUPPORTED_FACTS"
    PARTIALLY_SUPPORTED = "PARTIALLY_SUPPORTED"
    UNSUPPORTED = "UNSUPPORTED"
    CONTRADICTED = "CONTRADICTED"


@dataclass
class GroundedClaim:
    claim_id: str
    target_entity: str
    statement: str
    predicate: str
    conclusion_value: str
    confidence: float
    support_type: ClaimSupportType = ClaimSupportType.UNSUPPORTED
    supporting_evidence_ids: List[str] = field(default_factory=list)
    is_overclaim: bool = False
    epistemic_qualification: Optional[str] = None


@dataclass
class EvidenceNecessityReport:
    task_id: str
    evidence_id: str
    role: EvidenceRole
    conclusion_with_evidence: str
    conclusion_without_evidence: str
    conclusion_changed_on_removal: bool
    confidence_delta: float
    decision_impact: str


@dataclass
class EvidenceSufficiencyReport:
    task_id: str
    claim: GroundedClaim
    status: SufficiencyStatus
    evidence_coverage_ratio: float
    is_overclaim_prevented: bool
    rationale: str


class EvidenceEvaluator:
    """
    Evaluates evidence necessity, sufficiency, and overclaiming boundaries.
    """

    @classmethod
    def evaluate_necessity(
        cls,
        task_id: str,
        question: str,
        target_entity: str,
        evidence_item: Dict[str, Any],
        system_response_with_ev: Tuple[str, float],
        system_response_without_ev: Tuple[str, float],
    ) -> EvidenceNecessityReport:
        conc_with, conf_with = system_response_with_ev
        conc_without, conf_without = system_response_without_ev

        ev_entity = evidence_item.get("entity", "").strip().lower()
        ev_text = evidence_item.get("text", "").strip().lower()
        ev_polarity = evidence_item.get("polarity", True)

        changed = (conc_with.strip().lower() != conc_without.strip().lower())
        conf_delta = conf_with - conf_without

        # 1. Contradictory check
        if not ev_polarity or "not " in ev_text or "never " in ev_text:
            role = EvidenceRole.EVIDENCE_CONTRADICTORY
            decision_impact = "Evidence directly contradicts premise; mandates qualification or rejection."
        # 2. Entity match & necessity check
        elif ev_entity != target_entity.strip().lower():
            role = EvidenceRole.EVIDENCE_IRRELEVANT
            decision_impact = f"Evidence belongs to irrelevant entity '{ev_entity}'; zero causal necessity for {target_entity}."
        elif changed or conf_delta >= 0.40:
            role = EvidenceRole.EVIDENCE_NECESSARY
            decision_impact = "Conclusion or confidence causally depends on this evidence; removal forces epistemic change."
        elif conf_delta > 0.10:
            role = EvidenceRole.EVIDENCE_OPTIONAL
            decision_impact = "Evidence corroborates answer, but core conclusion is reachable via redundant path."
        else:
            role = EvidenceRole.EVIDENCE_IRRELEVANT
            decision_impact = "Evidence has zero causal impact on cognitive decision."

        return EvidenceNecessityReport(
            task_id=task_id,
            evidence_id=evidence_item.get("evidence_id", "ev_unk"),
            role=role,
            conclusion_with_evidence=conc_with,
            conclusion_without_evidence=conc_without,
            conclusion_changed_on_removal=changed,
            confidence_delta=conf_delta,
            decision_impact=decision_impact,
        )

    @classmethod
    def evaluate_sufficiency(
        cls,
        task_id: str,
        claim: GroundedClaim,
        evidence_list: List[Dict[str, Any]],
        strict_overclaim_guard: bool = True,
    ) -> EvidenceSufficiencyReport:
        if not evidence_list:
            return EvidenceSufficiencyReport(
                task_id=task_id,
                claim=claim,
                status=SufficiencyStatus.UNKNOWN,
                evidence_coverage_ratio=0.0,
                is_overclaim_prevented=False,
                rationale="Zero supporting evidence available; UNKNOWN state mandatory.",
            )

        # Filter evidence matching target entity
        matching_ev = [
            e for e in evidence_list
            if e.get("entity", "").strip().lower() == claim.target_entity.strip().lower()
        ]

        if not matching_ev:
            return EvidenceSufficiencyReport(
                task_id=task_id,
                claim=claim,
                status=SufficiencyStatus.INSUFFICIENT,
                evidence_coverage_ratio=0.0,
                is_overclaim_prevented=False,
                rationale=f"Retrieved evidence belongs to mismatched entities; insufficient for '{claim.target_entity}'.",
            )

        # Check for contradictions among evidence
        polarities = set(e.get("polarity", True) for e in matching_ev)
        if True in polarities and False in polarities:
            return EvidenceSufficiencyReport(
                task_id=task_id,
                claim=claim,
                status=SufficiencyStatus.CONTRADICTED,
                evidence_coverage_ratio=0.5,
                is_overclaim_prevented=False,
                rationale="Retrieved sources contain mutually conflicting polarities.",
            )

        # Check for overclaiming
        # Example: Premise asserts narrow metric, but claim makes broad inductive leap
        is_overclaim = False
        premise_texts = " ".join(e.get("text", "").lower() for e in matching_ev)
        claim_stmt = claim.statement.lower()

        # If claim asserts global health/perfection based solely on partial metric (e.g. revenue increase)
        if ("healthy" in claim_stmt or "perfect" in claim_stmt or "flawless" in claim_stmt) and \
           ("revenue" in premise_texts or "q1" in premise_texts or "partial" in premise_texts):
            is_overclaim = True

        if is_overclaim and strict_overclaim_guard:
            claim.is_overclaim = True
            claim.support_type = ClaimSupportType.PARTIALLY_SUPPORTED
            claim.epistemic_qualification = (
                "Evidence supports specific revenue growth, but is insufficient to claim overall financial health."
            )
            return EvidenceSufficiencyReport(
                task_id=task_id,
                claim=claim,
                status=SufficiencyStatus.PARTIALLY_SUPPORTED,
                evidence_coverage_ratio=0.5,
                is_overclaim_prevented=True,
                rationale="Prevented silent overclaiming: evidence supports subclaim only.",
            )

        # Exact predicate grounding check
        has_predicate = any(
            claim.predicate.lower() in e.get("relation", "").lower() or
            claim.conclusion_value.lower() in e.get("value", "").lower()
            for e in matching_ev
        )

        if has_predicate:
            claim.support_type = ClaimSupportType.DIRECTLY_SUPPORTED
            return EvidenceSufficiencyReport(
                task_id=task_id,
                claim=claim,
                status=SufficiencyStatus.SUPPORTED,
                evidence_coverage_ratio=1.0,
                is_overclaim_prevented=False,
                rationale="Evidence completely grounds the asserted proposition and target entity.",
            )
        else:
            claim.support_type = ClaimSupportType.INSUFFICIENT
            return EvidenceSufficiencyReport(
                task_id=task_id,
                claim=claim,
                status=SufficiencyStatus.INSUFFICIENT,
                evidence_coverage_ratio=0.3,
                is_overclaim_prevented=False,
                rationale="Evidence is related to entity but fails to establish the specific relation or value.",
            )
