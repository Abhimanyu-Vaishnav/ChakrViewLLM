"""
Learning-Worthiness & Novelty Decision Engine.
Wave 528: Governed evaluation of candidate experiences to decide whether, how,
and where an incoming experience should influence future cognitive behavior.

Decisions:
- IGNORE: Duplicate, uninformative, or redundant experience.
- STORE_EPISODICALLY: Useful raw history, but insufficient for abstraction.
- REQUEST_REVALIDATION: Ambiguous or borderline consistency; needs corroboration.
- CONSOLIDATE: Multiple corroborating episodes ready for abstract synthesis.
- PROMOTE_TO_STRATEGY: Verified high-utility execution policy for a task family.
- PROMOTE_TO_KNOWLEDGE: Grounded semantic assertion verified against contradictions.
- REJECT: Corrupted, unverified peer claim, or severely degraded artifact.

Evaluates discrete factors:
- novelty
- repetition
- confidence
- task relevance
- outcome quality
- failure severity
- supporting evidence count
- contradiction count
- provenance quality
- peer trust
- staleness (TTL)
- existing strategy coverage
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import time

class LearningDecision(str, Enum):
    IGNORE = "IGNORE"
    STORE_EPISODICALLY = "STORE_EPISODICALLY"
    REQUEST_REVALIDATION = "REQUEST_REVALIDATION"
    CONSOLIDATE = "CONSOLIDATE"
    PROMOTE_TO_STRATEGY = "PROMOTE_TO_STRATEGY"
    PROMOTE_TO_KNOWLEDGE = "PROMOTE_TO_KNOWLEDGE"
    REJECT = "REJECT"

@dataclass
class LearningWorthinessProfile:
    novelty_score: float              # 0.0 (exact duplicate) to 1.0 (completely novel task structure)
    repetition_count: int             # Number of times this task structure has been observed
    confidence: float                 # Output / model confidence (0.0 to 1.0)
    task_relevance: float             # Alignment with target cognitive capabilities (0.0 to 1.0)
    outcome_quality: float            # Success rating (1.0 = success, 0.0 = failure)
    failure_severity: float           # 0.0 (minor) to 1.0 (critical deadlock / crash)
    supporting_evidence_count: int    # Number of corroborating episodes or peer verifications
    contradiction_count: int          # Number of direct conflicts
    provenance_valid: bool            # Cryptographic hash and manifest valid
    peer_trust_score: float           # 0.0 (unverified / untrusted) to 1.0 (cryptographically signed peer)
    age_seconds: float                # Age since generation
    max_ttl_seconds: float = 86400 * 7 # 7 days staleness threshold
    existing_strategy_coverage: bool = False # Whether an optimal strategy is already locked in

class LearningWorthinessEngine:
    @staticmethod
    def evaluate(profile: LearningWorthinessProfile) -> Dict[str, Any]:
        """
        Evaluates the profile against governed safety rules.
        Exposes explicit factors rather than a single black-box score.
        """
        factors = {
            "is_corrupt_or_untrusted": not profile.provenance_valid or (profile.peer_trust_score < 0.3),
            "is_stale": profile.age_seconds > profile.max_ttl_seconds,
            "has_contradictions": profile.contradiction_count > 0,
            "is_duplicate": profile.novelty_score < 0.1 and profile.existing_strategy_coverage,
            "is_high_confidence_success": profile.confidence >= 0.85 and profile.outcome_quality >= 0.9,
            "is_severe_failure": profile.failure_severity >= 0.7,
            "has_consolidation_quorum": profile.supporting_evidence_count >= 2
        }

        # Rule 1: Security & Trust Floor
        if factors["is_corrupt_or_untrusted"]:
            return {
                "decision": LearningDecision.REJECT,
                "reason": "Provenance invalid or peer untrusted",
                "factors": factors
            }

        # Rule 2: Staleness Check
        if factors["is_stale"]:
            return {
                "decision": LearningDecision.IGNORE,
                "reason": "Experience exceeded TTL staleness horizon",
                "factors": factors
            }

        # Rule 3: Contradiction Handling
        if factors["has_contradictions"]:
            return {
                "decision": LearningDecision.REQUEST_REVALIDATION,
                "reason": f"Detected {profile.contradiction_count} conflicting observations; requires revalidation",
                "factors": factors
            }

        # Rule 4: Duplicate / Redundant
        if factors["is_duplicate"]:
            return {
                "decision": LearningDecision.IGNORE,
                "reason": "Task family already covered by verified strategy with zero novelty",
                "factors": factors
            }

        # Rule 5: Promotion to Strategy (requires quorum + high confidence)
        if factors["has_consolidation_quorum"] and factors["is_high_confidence_success"]:
            return {
                "decision": LearningDecision.PROMOTE_TO_STRATEGY,
                "reason": "Quorum satisfied with high confidence success",
                "factors": factors
            }

        # Rule 6: Ready for Consolidation
        if factors["has_consolidation_quorum"] or factors["is_severe_failure"]:
            return {
                "decision": LearningDecision.CONSOLIDATE,
                "reason": "Sufficient corroboration or severe failure pattern requires abstraction",
                "factors": factors
            }

        # Rule 7: Baseline Episodic Archival
        if profile.confidence >= 0.5:
            return {
                "decision": LearningDecision.STORE_EPISODICALLY,
                "reason": "Valid novel observation stored for future quorum consolidation",
                "factors": factors
            }

        return {
            "decision": LearningDecision.REQUEST_REVALIDATION,
            "reason": "Low confidence observation requires additional evidence",
            "factors": factors
        }
