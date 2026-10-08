"""
ChakrView Open-World Knowledge & Governed Epistemic Architecture.
Waves 546-555: Tool-assisted reasoning, multi-source verification,
contradiction preservation, temporal knowledge freshness, unknown-handling/abstention,
evidence attribution, and open-world continual learning.
"""

from .knowledge_interface import (
    OpenWorldKnowledgeInterface,
    KnowledgeRequest,
    KnowledgeCandidate,
    KnowledgeSource,
    SourceType,
    EvidenceItem,
    VerificationStatus,
    AttributionAssessment,
)
from .multi_source_verifier import (
    MultiSourceVerifier,
    SourceIndependenceAssessment,
    VerificationOutcome,
)
from .contradictory_store import (
    ContradictoryKnowledgeStore,
    ContradictionRecord,
    ContradictionResolutionVerdict,
)
from .temporal_knowledge import (
    TemporalKnowledgeManager,
    TemporalValidity,
    TemporalFactRecord,
)
from .unknown_handler import (
    UnknownHandlingEngine,
    InquiryClass,
    CognitiveDecision,
    EpistemicResolution,
)
from .tool_reasoning import (
    CognitiveToolOrchestrator,
    ToolType,
    ToolExecutionResult,
    ToolNecessity,
)
from .open_world_continual import (
    OpenWorldContinualEngine,
    OpenWorldLearningCycleReport,
)

__all__ = [
    "OpenWorldKnowledgeInterface",
    "KnowledgeRequest",
    "KnowledgeCandidate",
    "KnowledgeSource",
    "SourceType",
    "EvidenceItem",
    "VerificationStatus",
    "AttributionAssessment",
    "MultiSourceVerifier",
    "SourceIndependenceAssessment",
    "VerificationOutcome",
    "ContradictoryKnowledgeStore",
    "ContradictionRecord",
    "ContradictionResolutionVerdict",
    "TemporalKnowledgeManager",
    "TemporalValidity",
    "TemporalFactRecord",
    "UnknownHandlingEngine",
    "InquiryClass",
    "CognitiveDecision",
    "EpistemicResolution",
    "CognitiveToolOrchestrator",
    "ToolType",
    "ToolExecutionResult",
    "ToolNecessity",
    "OpenWorldContinualEngine",
    "OpenWorldLearningCycleReport",
]
