"""
ChakrView Critical Thinking Subsystem (Step 23).

Public exports for critical thinking primitives, traces, and orchestration engine.
"""

from chakrview.cognition.critical.models import (
    Hypothesis,
    Evidence,
    Assumption,
    CounterEvidence,
    AlternativeExplanation,
    Contradiction,
    VerificationResult,
    CriticalThinkingTrace,
    EvidenceStatus,
    CounterEvidenceStatus,
    HypothesisStatus,
    ContradictionSeverity,
    AssumptionCriticality,
    CriticalHypothesis,
    CriticalEvidence,
    CriticalContradiction,
    CriticalVerificationResult,
)
from chakrview.cognition.critical.engine import (
    CriticalThinkingEngine,
    CriticalThinkingConfig,
)

__all__ = [
    "Hypothesis",
    "Evidence",
    "Assumption",
    "CounterEvidence",
    "AlternativeExplanation",
    "Contradiction",
    "VerificationResult",
    "CriticalThinkingTrace",
    "EvidenceStatus",
    "CounterEvidenceStatus",
    "HypothesisStatus",
    "ContradictionSeverity",
    "AssumptionCriticality",
    "CriticalHypothesis",
    "CriticalEvidence",
    "CriticalContradiction",
    "CriticalVerificationResult",
    "CriticalThinkingEngine",
    "CriticalThinkingConfig",
]
