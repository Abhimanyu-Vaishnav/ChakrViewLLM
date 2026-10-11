"""
ChakrView Continual Cognition Architecture.
Waves 537-545: Truth-seeking, continuous learning, knowledge revision,
unlearning, cross-task/domain transfer, adaptive curriculum v2,
long-horizon resilience, resource awareness, and emotional context handling.
"""

from .continual_stream import (
    ContinualTaskStream,
    ContinualTask,
    ContinualEvaluationReport,
    run_continual_stream_benchmark,
)
from .knowledge_revision import (
    GovernedBeliefStore,
    BeliefRecord,
    BeliefRevisionEvent,
    RevisionVerdict,
)
from .negative_unlearning import (
    NegativeTransferDetector,
    UnlearningController,
    UnlearningStatus,
)
from .structural_transfer import (
    StructuralTransferEngine,
    TransferClassification,
    TransferEvaluationResult,
)
from .curriculum_v2 import (
    AdaptiveCurriculumV2,
    CapabilityGap,
    GapType,
)
from .resource_governor import (
    ResourceAwareCognitiveGovernor,
    SystemResourceConstraint,
    ExecutionBudgetMode,
)
from .truth_and_context import (
    TruthSeekingEngine,
    TruthCategory,
    EmotionalContext,
    UserEmotionalContextDetector,
    TruthfulResponse,
)

__all__ = [
    "ContinualTaskStream",
    "ContinualTask",
    "ContinualEvaluationReport",
    "run_continual_stream_benchmark",
    "GovernedBeliefStore",
    "BeliefRecord",
    "BeliefRevisionEvent",
    "RevisionVerdict",
    "NegativeTransferDetector",
    "UnlearningController",
    "UnlearningStatus",
    "StructuralTransferEngine",
    "TransferClassification",
    "TransferEvaluationResult",
    "AdaptiveCurriculumV2",
    "CapabilityGap",
    "GapType",
    "ResourceAwareCognitiveGovernor",
    "SystemResourceConstraint",
    "ExecutionBudgetMode",
    "TruthSeekingEngine",
    "TruthCategory",
    "EmotionalContext",
    "UserEmotionalContextDetector",
    "TruthfulResponse",
]
