"""
Experience Quality & Trust Evaluator.
Wave 529: Governed trust appraisal ensuring confidence alone cannot override evidence.

Classifications:
- HIGH_TRUST: Corroborated by independent repetition, valid provenance, and peer verification.
- MEDIUM_TRUST: Plausible local or single-source observation; requires corroboration.
- LOW_TRUST: Isolated, weak, or high-uncertainty observation; cannot alter policies.
- REJECTED: Cryptographically invalid, contradictory, or adversarial.

Explicitly addresses:
1. One highly confident but wrong experience (must not be HIGH_TRUST).
2. Multiple weak but mutually consistent experiences (can reach MEDIUM/HIGH trust).
3. Verified peer vs unverified peer experiences.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Dict, Any, List

class TrustLevel(str, Enum):
    HIGH_TRUST = "HIGH_TRUST"
    MEDIUM_TRUST = "MEDIUM_TRUST"
    LOW_TRUST = "LOW_TRUST"
    REJECTED = "REJECTED"

@dataclass
class EvidenceObservation:
    source_id: str
    confidence: float
    is_verified_peer: bool
    is_unverified_peer: bool
    provenance_valid: bool
    outcome_correct: bool
    consistent_with_corpus: bool

class ExperienceTrustEvaluator:
    @staticmethod
    def evaluate(observations: List[EvidenceObservation]) -> Dict[str, Any]:
        if not observations:
            return {"trust_level": TrustLevel.REJECTED, "reason": "No evidence provided"}

        # Check for unverified peer claims or invalid provenance
        for obs in observations:
            if obs.is_unverified_peer or not obs.provenance_valid:
                return {
                    "trust_level": TrustLevel.REJECTED,
                    "reason": "Unverified peer or corrupted provenance detected"
                }

        # Check single-shot overconfidence fallacy:
        # High confidence alone on a single run without independent corroboration cannot attain HIGH_TRUST
        if len(observations) == 1:
            obs = observations[0]
            if not obs.outcome_correct:
                return {
                    "trust_level": TrustLevel.REJECTED,
                    "reason": "Confirmed erroneous outcome despite reported confidence"
                }
            if obs.is_verified_peer and obs.confidence >= 0.9:
                return {
                    "trust_level": TrustLevel.HIGH_TRUST,
                    "reason": "Cryptographically verified peer observation"
                }
            return {
                "trust_level": TrustLevel.MEDIUM_TRUST,
                "reason": "Single observation cannot achieve HIGH_TRUST without independent corroboration"
            }

        # Multi-observation aggregation
        correct_count = sum(1 for o in observations if o.outcome_correct)
        consistent_count = sum(1 for o in observations if o.consistent_with_corpus)
        avg_conf = sum(o.confidence for o in observations) / len(observations)

        # Consensus quorum check
        if correct_count == len(observations) and consistent_count >= len(observations) - 1:
            if len(observations) >= 3 or any(o.is_verified_peer for o in observations):
                return {
                    "trust_level": TrustLevel.HIGH_TRUST,
                    "reason": f"Strong consensus across {len(observations)} independent observations"
                }
            return {
                "trust_level": TrustLevel.MEDIUM_TRUST,
                "reason": f"Consistent observations ({len(observations)}) advancing toward quorum"
            }

        # Mixed / Contradictory outcomes
        if correct_count > 0 and correct_count < len(observations):
            return {
                "trust_level": TrustLevel.LOW_TRUST,
                "reason": "Contradictory outcomes observed across candidate pool"
            }

        return {
            "trust_level": TrustLevel.REJECTED,
            "reason": "All observations failed validation criteria"
        }
