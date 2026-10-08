"""
Conflict Resolution & Knowledge Revalidation Engine.
Wave 531: Epistemically safe resolution of contradictory experiences and knowledge claims.

Resolves conflicts using:
- Provenance validity
- Corroborating independent evidence
- Empirical verification vs subjective heuristic
- Confidence dominance margin
- Recency weighting

Decisions:
- ACCEPT_A / ACCEPT_B: Clear evidence dominance.
- CONTEXTUALIZE: Both are valid under different task conditions / premise splits.
- REVALIDATE: Balanced or ambiguous conflict; triggers verification task.
- REJECT_BOTH: Both claims show corrupted provenance or fail tests.
- UNKNOWN: Insufficient evidence to decide (epistemic safety default).
"""

from enum import Enum
from dataclasses import dataclass
from typing import Dict, Any, Optional

class ConflictResolutionDecision(str, Enum):
    ACCEPT_A = "ACCEPT_A"
    ACCEPT_B = "ACCEPT_B"
    CONTEXTUALIZE = "CONTEXTUALIZE"
    REVALIDATE = "REVALIDATE"
    REJECT_BOTH = "REJECT_BOTH"
    UNKNOWN = "UNKNOWN"

@dataclass
class ConflictingClaim:
    claim_id: str
    assertion: str
    confidence: float
    supporting_evidence_count: int
    provenance_valid: bool
    context_condition: str

class ConflictResolver:
    @staticmethod
    def resolve(claim_a: ConflictingClaim, claim_b: ConflictingClaim) -> Dict[str, Any]:
        # Rule 1: Provenance validity
        if not claim_a.provenance_valid and not claim_b.provenance_valid:
            return {"decision": ConflictResolutionDecision.REJECT_BOTH, "reason": "Both claims have corrupted provenance"}
        if not claim_a.provenance_valid:
            return {"decision": ConflictResolutionDecision.ACCEPT_B, "reason": "Claim A has invalid provenance"}
        if not claim_b.provenance_valid:
            return {"decision": ConflictResolutionDecision.ACCEPT_A, "reason": "Claim B has invalid provenance"}

        # Rule 2: Contextualization
        if claim_a.context_condition != claim_b.context_condition and claim_a.context_condition != "global":
            return {
                "decision": ConflictResolutionDecision.CONTEXTUALIZE,
                "reason": f"Both claims valid under distinct contexts: '{claim_a.context_condition}' vs '{claim_b.context_condition}'"
            }

        # Rule 3: Significant evidence count disparity
        evidence_diff = claim_a.supporting_evidence_count - claim_b.supporting_evidence_count
        if evidence_diff >= 3 and claim_a.confidence >= claim_b.confidence:
            return {"decision": ConflictResolutionDecision.ACCEPT_A, "reason": "Claim A possesses dominant evidence quorum"}
        if evidence_diff <= -3 and claim_b.confidence >= claim_a.confidence:
            return {"decision": ConflictResolutionDecision.ACCEPT_B, "reason": "Claim B possesses dominant evidence quorum"}

        # Rule 4: Close contention or weak evidence -> Epistemic Safety
        if claim_a.supporting_evidence_count < 2 and claim_b.supporting_evidence_count < 2:
            return {
                "decision": ConflictResolutionDecision.UNKNOWN,
                "reason": "Insufficient evidence on both sides to resolve conflict epistemically safely"
            }

        # Rule 5: Active contention requiring benchmark
        return {
            "decision": ConflictResolutionDecision.REVALIDATE,
            "reason": "Equally weighted conflicting evidence requires formal revalidation probe"
        }
