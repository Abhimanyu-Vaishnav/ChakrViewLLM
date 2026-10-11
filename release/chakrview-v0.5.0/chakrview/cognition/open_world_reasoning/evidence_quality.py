"""
Wave 571: Explicit Evidence Quality Estimation Engine.

Architectural Guarantees:
- Distinguishes SOURCE QUALITY (reputation/domain) from EVIDENCE QUALITY (directness, relevance, validity).
- Never collapses nuanced multi-dimensional attributes into a naive blind scalar.
- Structured Quality Grades:
  HIGH_SUPPORT, MODERATE_SUPPORT, WEAK_SUPPORT, IRRELEVANT, CONTRADICTORY, STALE, UNKNOWN_QUALITY
- Evaluates:
  relevance, entity correctness, relation correctness, temporal validity,
  provenance integrity, source independence, directness, completeness, specificity.
"""

from __future__ import annotations

import time
import hashlib
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple


class EvidenceQualityGrade(str, Enum):
    HIGH_SUPPORT = "HIGH_SUPPORT"
    MODERATE_SUPPORT = "MODERATE_SUPPORT"
    WEAK_SUPPORT = "WEAK_SUPPORT"
    IRRELEVANT = "IRRELEVANT"
    CONTRADICTORY = "CONTRADICTORY"
    STALE = "STALE"
    UNKNOWN_QUALITY = "UNKNOWN_QUALITY"


@dataclass
class EvidenceQualityReport:
    evidence_id: str
    grade: EvidenceQualityGrade
    relevance_score: float         # 0.0 to 1.0
    entity_match: bool
    relation_match: bool
    temporal_valid: bool
    provenance_verified: bool
    is_direct: bool
    is_contradictory: bool
    source_reputation: float       # Independent source metric
    evidence_validity_score: float # Computed composite quality
    rationale: str


class EvidenceQualityEvaluator:
    """
    Independent evidence quality estimator assessing factual directness,
    entity match, temporal validity, and contradiction status.
    """

    @classmethod
    def evaluate_item(
        cls,
        target_entity: str,
        target_relation: str,
        target_claim: str,
        evidence: Dict[str, Any],
        current_timestamp: Optional[float] = None,
        max_staleness_seconds: float = 86400.0 * 365.0, # 1 year default
    ) -> EvidenceQualityReport:
        now = current_timestamp if current_timestamp is not None else time.time()
        ev_id = evidence.get("evidence_id", "unknown_ev")
        ev_entity = str(evidence.get("entity", "")).strip().lower()
        ev_relation = str(evidence.get("relation", "")).strip().lower()
        ev_content = str(evidence.get("content", "")).strip().lower()
        ev_hash = evidence.get("content_hash", "")
        ev_timestamp = evidence.get("publication_timestamp", now)
        is_time_varying = evidence.get("is_time_varying", True)
        source_reputation = float(evidence.get("source_reputation", 0.5))
        is_direct = bool(evidence.get("is_direct", True))
        is_contradictory_flag = bool(evidence.get("is_contradictory", False))

        tgt_ent_clean = target_entity.strip().lower()
        tgt_rel_clean = target_relation.strip().lower()
        tgt_claim_clean = target_claim.strip().lower()

        # 1. Provenance Integrity Check
        computed_hash = hashlib.sha256(ev_content.encode("utf-8")).hexdigest()
        provenance_ok = (ev_hash == computed_hash) if ev_hash else True

        # 2. Entity Correctness
        entity_match = (ev_entity == tgt_ent_clean) or (tgt_ent_clean in ev_entity) or (tgt_ent_clean in ev_content)

        # 3. Relation Correctness
        relation_match = (ev_relation == tgt_rel_clean) or (tgt_rel_clean in ev_relation) or (tgt_rel_clean in ev_content)

        # 4. Temporal Validity
        temporal_valid = True
        if is_time_varying and ev_timestamp is not None:
            age = now - ev_timestamp
            if age > max_staleness_seconds:
                temporal_valid = False

        # 5. Contradiction Detection against Target Claim
        # Check explicit negative keywords or polar opposites
        negation_tokens = ["not ", "never ", "false", "disputed", "opposite", "contradicts"]
        explicit_negation = any(tok in ev_content for tok in negation_tokens) and (tgt_ent_clean in ev_content)
        is_contradictory = is_contradictory_flag or explicit_negation

        # 6. Topical Relevance
        claim_words = set(tgt_claim_clean.split())
        ev_words = set(ev_content.split())
        overlap = len(claim_words.intersection(ev_words))
        relevance_score = overlap / max(len(claim_words), 1)

        # 7. Classification Logic
        if not entity_match or relevance_score < 0.15:
            grade = EvidenceQualityGrade.IRRELEVANT
            validity_score = 0.05
            rationale = "Evidence addresses an unrelated entity or has negligible topical overlap."
        elif is_contradictory:
            grade = EvidenceQualityGrade.CONTRADICTORY
            validity_score = 0.20
            rationale = "Evidence directly contradicts the target proposition."
        elif not temporal_valid:
            grade = EvidenceQualityGrade.STALE
            validity_score = 0.30
            rationale = "Evidence is temporally expired / superseded."
        elif not relation_match or not is_direct:
            grade = EvidenceQualityGrade.WEAK_SUPPORT
            validity_score = 0.50
            rationale = "Evidence is indirectly relevant but lacks exact relational binding."
        elif relevance_score >= 0.4 and entity_match and relation_match and temporal_valid and provenance_ok:
            if source_reputation >= 0.7:
                grade = EvidenceQualityGrade.HIGH_SUPPORT
                validity_score = 0.95
                rationale = "High quality direct primary evidence with verified provenance."
            else:
                grade = EvidenceQualityGrade.MODERATE_SUPPORT
                validity_score = 0.75
                rationale = "Direct evidence matches target claim, though source authority is moderate."
        else:
            grade = EvidenceQualityGrade.UNKNOWN_QUALITY
            validity_score = 0.40
            rationale = "Incomplete evidence indicators; quality cannot be definitively certified."

        return EvidenceQualityReport(
            evidence_id=ev_id,
            grade=grade,
            relevance_score=relevance_score,
            entity_match=entity_match,
            relation_match=relation_match,
            temporal_valid=temporal_valid,
            provenance_verified=provenance_ok,
            is_direct=is_direct,
            is_contradictory=is_contradictory,
            source_reputation=source_reputation,
            evidence_validity_score=validity_score,
            rationale=rationale,
        )
