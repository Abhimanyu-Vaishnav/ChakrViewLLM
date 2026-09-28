"""
ChakrView Governed Cognitive Reasoning Subsystem (Step 19).

Provides structured, inspectable, machine-readable computational reasoning:
Understand -> Decompose -> Retrieve Evidence -> Form Hypotheses -> Reason (Inference)
-> Check Consistency -> Estimate Uncertainty -> Decide -> Act -> Observe -> Verify
-> Revise -> Finalize.

Preserves the fundamental invariants:
- UNKNOWN != FALSE
- DATA != AUTHORITY
- REASONING != AUTHORITY
- CAPABILITY EXISTENCE != AUTHORIZATION
"""

from chakrview.reasoning.task import (
    ReasoningTask,
    ReasoningPhase,
    ReasoningStatus,
    ReasoningTaskType,
)
from chakrview.reasoning.decomposition import (
    Subproblem,
    SubproblemStatus,
    DecompositionTree,
    ProblemDecomposer,
    DecompositionLimitError,
)
from chakrview.reasoning.evidence import (
    EvidenceType,
    EvidenceItem,
    EvidenceStore,
    DEFAULT_RELIABILITIES,
)
from chakrview.reasoning.hypothesis import (
    HypothesisStatus,
    Hypothesis,
    HypothesisEngine,
)
from chakrview.reasoning.inference import (
    InferenceType,
    Inference,
    InferenceEngine,
)
from chakrview.reasoning.contradiction import (
    ContradictionSeverity,
    ContradictionStatus,
    Contradiction,
    ContradictionDetector,
)
from chakrview.reasoning.decision import (
    DecisionCandidate,
    Decision,
    DecisionEngine,
)
from chakrview.reasoning.verification import (
    VerificationStatus,
    VerificationCriteria,
    VerificationResult,
    VerificationEngine,
)
from chakrview.reasoning.trace import (
    ReasoningTrace,
)
from chakrview.reasoning.policies import (
    ReasoningPolicy,
    get_standard_policy,
    get_strict_policy,
    get_fast_policy,
)
from chakrview.reasoning.engine import (
    GovernedReasoningEngine,
)

__all__ = [
    # Task
    "ReasoningTask",
    "ReasoningPhase",
    "ReasoningStatus",
    "ReasoningTaskType",
    # Decomposition
    "Subproblem",
    "SubproblemStatus",
    "DecompositionTree",
    "ProblemDecomposer",
    "DecompositionLimitError",
    # Evidence
    "EvidenceType",
    "EvidenceItem",
    "EvidenceStore",
    "DEFAULT_RELIABILITIES",
    # Hypothesis
    "HypothesisStatus",
    "Hypothesis",
    "HypothesisEngine",
    # Inference
    "InferenceType",
    "Inference",
    "InferenceEngine",
    # Contradiction
    "ContradictionSeverity",
    "ContradictionStatus",
    "Contradiction",
    "ContradictionDetector",
    # Decision
    "DecisionCandidate",
    "Decision",
    "DecisionEngine",
    # Verification
    "VerificationStatus",
    "VerificationCriteria",
    "VerificationResult",
    "VerificationEngine",
    # Trace
    "ReasoningTrace",
    # Policies
    "ReasoningPolicy",
    "get_standard_policy",
    "get_strict_policy",
    "get_fast_policy",
    # Engine
    "GovernedReasoningEngine",
]
