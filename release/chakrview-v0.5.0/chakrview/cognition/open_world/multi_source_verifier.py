"""
Wave 548: Multi-Source Independent Verification Engine.

Enforces:
- Compares multiple sources rather than trusting the first hit.
- Detects independent corroboration vs duplicated syndicated reporting (anti-naive-voting).
- Detects contradiction between sources.
- Calibrates evidence coverage and source reputation disparity.
- Never manufactures artificial consensus.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple, Set

from .knowledge_interface import (
    EvidenceItem,
    KnowledgeSource,
    VerificationStatus,
    KnowledgeCandidate,
)


@dataclass
class SourceIndependenceAssessment:
    total_sources: int
    unique_publishers: int
    distinct_content_clusters: int
    is_genuinely_independent: bool
    syndication_or_copy_detected: bool
    independence_score: float  # 0.0 to 1.0


@dataclass
class VerificationOutcome:
    status: VerificationStatus
    verdict_confidence: float
    corroborating_count: int
    contradicting_count: int
    independence: SourceIndependenceAssessment
    reputable_source_agreed: bool
    conflict_detected: bool
    rationale: str


class MultiSourceVerifier:
    """
    Rigorously verifies evidence across multiple sources, detecting syndicated repetition
    and genuine contradictions.
    """

    @classmethod
    def assess_independence(
        cls,
        evidence_items: List[EvidenceItem],
        sources: Dict[str, KnowledgeSource],
    ) -> SourceIndependenceAssessment:
        if not evidence_items:
            return SourceIndependenceAssessment(
                total_sources=0,
                unique_publishers=0,
                distinct_content_clusters=0,
                is_genuinely_independent=False,
                syndication_or_copy_detected=False,
                independence_score=0.0,
            )

        publishers: Set[str] = set()
        hashes: Set[str] = set()

        for ev in evidence_items:
            src = sources.get(ev.source_id)
            if src and src.publisher_or_author:
                publishers.add(src.publisher_or_author.strip().lower())
            else:
                publishers.add(ev.source_id)
            hashes.add(ev.content_hash)

        total = len(evidence_items)
        unique_pub = len(publishers)
        unique_hashes = len(hashes)

        # If 3+ sources share identical text or publisher, syndication is detected
        syndicated = (total >= 3 and (unique_pub == 1 or unique_hashes == 1))
        indep_score = min(1.0, (unique_pub / max(1, total)) * (unique_hashes / max(1, total)))
        is_independent = indep_score >= 0.6 and not syndicated

        return SourceIndependenceAssessment(
            total_sources=total,
            unique_publishers=unique_pub,
            distinct_content_clusters=unique_hashes,
            is_genuinely_independent=is_independent,
            syndication_or_copy_detected=syndicated,
            independence_score=indep_score,
        )

    @classmethod
    def verify_candidate(
        cls,
        candidate: KnowledgeCandidate,
        sources: Dict[str, KnowledgeSource],
        required_quorum: int = 2,
    ) -> VerificationOutcome:
        ev_items = candidate.evidence_items
        if not ev_items:
            return VerificationOutcome(
                status=VerificationStatus.INSUFFICIENT_EVIDENCE,
                verdict_confidence=0.0,
                corroborating_count=0,
                contradicting_count=0,
                independence=cls.assess_independence([], {}),
                reputable_source_agreed=False,
                conflict_detected=False,
                rationale="Zero evidence items provided.",
            )

        independence = cls.assess_independence(ev_items, sources)

        # Partition values
        value_counts: Dict[str, List[EvidenceItem]] = {}
        for ev in ev_items:
            norm_val = ev.claimed_value.strip().lower()
            if norm_val not in value_counts:
                value_counts[norm_val] = []
            value_counts[norm_val].append(ev)

        # Check for direct contradictions (multiple distinct claimed values)
        if len(value_counts) > 1:
            # Contradiction detected
            # Check if one side has overwhelming reputable authority (e.g. 0.95 vs 0.3)
            sorted_clusters = sorted(
                value_counts.items(),
                key=lambda item: max(sources.get(e.source_id, KnowledgeSource("", None, "")).reputation_score for e in item[1]),
                reverse=True,
            )
            top_val, top_evs = sorted_clusters[0]
            second_val, second_evs = sorted_clusters[1]

            top_rep = max(sources.get(e.source_id, KnowledgeSource("", None, "")).reputation_score for e in top_evs)
            second_rep = max(sources.get(e.source_id, KnowledgeSource("", None, "")).reputation_score for e in second_evs)

            if top_rep >= 0.85 and second_rep < 0.40:
                # Clear authority dominance over weak source
                return VerificationOutcome(
                    status=VerificationStatus.SUPPORTED,
                    verdict_confidence=top_rep,
                    corroborating_count=len(top_evs),
                    contradicting_count=len(second_evs),
                    independence=independence,
                    reputable_source_agreed=True,
                    conflict_detected=True,
                    rationale=f"High-reputation source ({top_rep:.2f}) verified '{top_val}'; discredited weak source ({second_rep:.2f}).",
                )

            return VerificationOutcome(
                status=VerificationStatus.CONTRADICTED,
                verdict_confidence=0.5,
                corroborating_count=len(top_evs),
                contradicting_count=len(second_evs),
                independence=independence,
                reputable_source_agreed=False,
                conflict_detected=True,
                rationale=f"Direct contradiction between sources: '{top_val}' vs '{second_val}'. Preserving conflict.",
            )

        # Single consistent value claimed
        val, matching_evs = next(iter(value_counts.items()))
        reputable = any(
            sources.get(e.source_id, KnowledgeSource("", None, "")).reputation_score >= 0.80
            for e in matching_evs
        )

        if len(matching_evs) >= required_quorum and independence.is_genuinely_independent and reputable:
            status = VerificationStatus.VERIFIED
            conf = 0.95
            rat = f"Verified across {len(matching_evs)} independent reputable sources."
        elif len(matching_evs) >= 1 and reputable:
            status = VerificationStatus.SUPPORTED
            conf = 0.80
            rat = "Supported by reputable source, but full independent multi-source quorum not met."
        elif independence.syndication_or_copy_detected:
            status = VerificationStatus.PARTIALLY_SUPPORTED
            conf = 0.60
            rat = "Multiple sources observed, but syndication/copying detected; counts as single observation."
        else:
            status = VerificationStatus.PARTIALLY_SUPPORTED
            conf = 0.60
            rat = "Low corroboration quorum."

        return VerificationOutcome(
            status=status,
            verdict_confidence=conf,
            corroborating_count=len(matching_evs),
            contradicting_count=0,
            independence=independence,
            reputable_source_agreed=reputable,
            conflict_detected=False,
            rationale=rat,
        )
