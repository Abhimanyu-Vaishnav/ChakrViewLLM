"""
ChakrView Neural Reasoning Integration & Intelligence Loop (Step 20).

Subsystem bridging:
ChakrMicro Neural Core <-> Cognitive State <-> Governed Reasoning <-> Capability Gate <-> Learning Loop
"""

from chakrview.intelligence.contracts import (
    LearningRecordStatus,
    UncertaintyMetric,
    NeuralInferenceRequest,
    NeuralInferenceResult,
    LearningRecord,
)
from chakrview.intelligence.context import (
    ContextSourceType,
    ContextBudget,
    ContextItem,
    AssembledIntelligenceContext,
    IntelligenceContextBuilder,
)
from chakrview.intelligence.inference import NeuralInferenceEngine
from chakrview.intelligence.feedback import (
    FeedbackCategory,
    RuntimeObservation,
    RuntimeEvaluation,
    FeedbackCollector,
)
from chakrview.intelligence.learning import (
    ModelUpdateSafetyError,
    TenantIsolationError,
    ModelVersionArtifact,
    LearningPipeline,
    ModelUpdateManager,
)
from chakrview.intelligence.pipeline import (
    IntelligenceLoopOutcome,
    NeuralIntelligenceLoop,
)

__all__ = [
    # Contracts
    "LearningRecordStatus",
    "UncertaintyMetric",
    "NeuralInferenceRequest",
    "NeuralInferenceResult",
    "LearningRecord",
    # Context
    "ContextSourceType",
    "ContextBudget",
    "ContextItem",
    "AssembledIntelligenceContext",
    "IntelligenceContextBuilder",
    # Inference
    "NeuralInferenceEngine",
    # Feedback
    "FeedbackCategory",
    "RuntimeObservation",
    "RuntimeEvaluation",
    "FeedbackCollector",
    # Learning
    "ModelUpdateSafetyError",
    "TenantIsolationError",
    "ModelVersionArtifact",
    "LearningPipeline",
    "ModelUpdateManager",
    # Pipeline
    "IntelligenceLoopOutcome",
    "NeuralIntelligenceLoop",
]
