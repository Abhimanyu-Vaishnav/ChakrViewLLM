"""
ChakrView Runtime Architecture Layer (Step 9).

Provides the modular, layered software framework decoupling knowledge, skills,
integrity, hardware adaptation, and version lineage from the frozen neural core.
"""

from chakrview.runtime.knowledge import (
    KnowledgeSource,
    KnowledgeDocument,
    KnowledgeChunk,
    KnowledgeIndex,
    InMemoryKnowledgeIndex,
    Retriever,
    SimpleRetriever,
    ContextProvider,
    KnowledgeAdapter,
)
from chakrview.runtime.skills import (
    SkillDomain,
    SkillPolicy,
    Skill,
    SkillExecutionPlan,
    SkillRegistry,
    SkillProfile,
)
from chakrview.runtime.improvement import (
    ChangeType,
    RiskLevel,
    ProposalStatus,
    ImprovementProposal,
    ImprovementProposalManager,
)
from chakrview.runtime.integrity import (
    HealthStatus,
    HealthCheckResult,
    HealthReport,
    ArtifactVerifier,
    QuarantineRecord,
    QuarantineManager,
    RollbackManager,
)
from chakrview.runtime.hardware import (
    ComputeDevice,
    PrecisionType,
    HardwareProfile,
    HardwareCapabilityDetector,
    ModelExecutionPlan,
    RuntimePlanner,
)
from chakrview.runtime.versioning import (
    BrainProfileType,
    BrainVersionManifest,
    VersionLineageTracker,
)
from chakrview.runtime.sampling import (
    SamplingStrategy,
    SamplingConfig,
    Sampler,
    apply_repetition_penalty,
    apply_top_k,
    apply_top_p,
    apply_min_prob,
)
from chakrview.runtime.inference import (
    StopReason,
    StreamToken,
    GenerationConfig,
    InferenceMetrics,
    GenerationResult,
    InferenceSession,
)

__all__ = [
    # Knowledge
    "KnowledgeSource",
    "KnowledgeDocument",
    "KnowledgeChunk",
    "KnowledgeIndex",
    "InMemoryKnowledgeIndex",
    "Retriever",
    "SimpleRetriever",
    "ContextProvider",
    "KnowledgeAdapter",
    # Skills
    "SkillDomain",
    "SkillPolicy",
    "Skill",
    "SkillExecutionPlan",
    "SkillRegistry",
    "SkillProfile",
    # Self-Improvement
    "ChangeType",
    "RiskLevel",
    "ProposalStatus",
    "ImprovementProposal",
    "ImprovementProposalManager",
    # Integrity & Rollback
    "HealthStatus",
    "HealthCheckResult",
    "HealthReport",
    "ArtifactVerifier",
    "QuarantineRecord",
    "QuarantineManager",
    "RollbackManager",
    # Hardware Adaptation
    "ComputeDevice",
    "PrecisionType",
    "HardwareProfile",
    "HardwareCapabilityDetector",
    "ModelExecutionPlan",
    "RuntimePlanner",
    # Versioning & Lineage
    "BrainProfileType",
    "BrainVersionManifest",
    "VersionLineageTracker",
    # Sampling
    "SamplingStrategy",
    "SamplingConfig",
    "Sampler",
    "apply_repetition_penalty",
    "apply_top_k",
    "apply_top_p",
    "apply_min_prob",
    # Inference
    "StopReason",
    "StreamToken",
    "GenerationConfig",
    "InferenceMetrics",
    "GenerationResult",
    "InferenceSession",
]

