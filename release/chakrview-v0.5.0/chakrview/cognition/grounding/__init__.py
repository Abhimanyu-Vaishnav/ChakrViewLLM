"""
ChakrView Evidence-Grounded Cognition & Causal Intervention Framework.
Waves 556-565: Evidence necessity, sufficiency, counterfactual substitution,
removal ablations, multi-hop evidence reasoning, contradiction intervention,
temporal evidence grounding, claim-level attribution, and adversarial stress-testing.
"""

from .evidence_evaluation import (
    EvidenceRole,
    SufficiencyStatus,
    ClaimSupportType,
    EvidenceEvaluator,
    GroundedClaim,
    EvidenceNecessityReport,
    EvidenceSufficiencyReport,
)
from .causal_interventions import (
    EvidenceInterventionEngine,
    InterventionType,
    CounterfactualTestResult,
    RemovalAblationReport,
)
from .multihop_grounding import (
    MultiHopEvidenceChainEngine,
    EvidenceHopLink,
    MultiHopChainReport,
)
from .answer_attribution import (
    FinalAnswerAttributionEngine,
    AttributedClaimGraph,
    ClaimAttributionNode,
)

__all__ = [
    "EvidenceRole",
    "SufficiencyStatus",
    "ClaimSupportType",
    "EvidenceEvaluator",
    "GroundedClaim",
    "EvidenceNecessityReport",
    "EvidenceSufficiencyReport",
    "EvidenceInterventionEngine",
    "InterventionType",
    "CounterfactualTestResult",
    "RemovalAblationReport",
    "MultiHopEvidenceChainEngine",
    "EvidenceHopLink",
    "MultiHopChainReport",
    "FinalAnswerAttributionEngine",
    "AttributedClaimGraph",
    "ClaimAttributionNode",
]
