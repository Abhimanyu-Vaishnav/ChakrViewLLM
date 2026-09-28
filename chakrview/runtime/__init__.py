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
    KnowledgeProvenance,
    DocumentChunker,
    DocumentIngester,
    BM25KnowledgeIndex,
    LexicalRetriever,
)
from chakrview.runtime.skills import (
    SkillDomain,
    SkillPolicy,
    Skill,
    SkillExecutionPlan,
    SkillRegistry,
    SkillProfile,
    SkillResolver,
    RuleBasedSkillResolver,
    get_standard_skill_registry,
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
from chakrview.runtime.context import (
    ContextBudget,
    AssembledContext,
    PromptContextBuilder,
)
from chakrview.runtime.tools import (
    ToolResult,
    Tool,
    CalculatorTool,
    TextUtilityTool,
    ToolRegistry,
    ToolExecutor,
    get_standard_tool_registry,
)
from chakrview.runtime.inference import (
    StopReason,
    StreamToken,
    GenerationConfig,
    InferenceMetrics,
    GenerationResult,
    RAGResponse,
    ChatResponse,
    InferenceSession,
)
from chakrview.runtime.memory import (
    MemoryType,
    ConversationTurn,
    MemoryItem,
    WorkingMemory,
    ConversationState,
    ConversationStore,
    MemoryExtractor,
    ConversationSummarizer,
)
from chakrview.runtime.retrieval import (
    RetrievalSourceType,
    RetrievalCandidate,
    RetrievalQuery,
    RetrievalResult,
    EmbeddingProvider,
    DeterministicHashEmbeddingProvider,
    VectorIndex,
    InMemoryVectorIndex,
    HybridRetriever,
    UnifiedRetriever,
    NeuralSemanticEmbeddingProvider,
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
    "KnowledgeProvenance",
    "DocumentChunker",
    "DocumentIngester",
    "BM25KnowledgeIndex",
    "LexicalRetriever",
    # Skills
    "SkillDomain",
    "SkillPolicy",
    "Skill",
    "SkillExecutionPlan",
    "SkillRegistry",
    "SkillProfile",
    "SkillResolver",
    "RuleBasedSkillResolver",
    "get_standard_skill_registry",
    # Context
    "ContextBudget",
    "AssembledContext",
    "PromptContextBuilder",
    # Governed Tools
    "ToolResult",
    "Tool",
    "CalculatorTool",
    "TextUtilityTool",
    "ToolRegistry",
    "ToolExecutor",
    "get_standard_tool_registry",
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
    "RAGResponse",
    "ChatResponse",
    "InferenceSession",
    # Conversational State & Memory (Step 12)
    "MemoryType",
    "ConversationTurn",
    "MemoryItem",
    "WorkingMemory",
    "ConversationState",
    "ConversationStore",
    "MemoryExtractor",
    "ConversationSummarizer",
    # Hybrid Retrieval & Semantic Foundations (Step 13)
    "RetrievalSourceType",
    "RetrievalCandidate",
    "RetrievalQuery",
    "RetrievalResult",
    "EmbeddingProvider",
    "DeterministicHashEmbeddingProvider",
    "VectorIndex",
    "InMemoryVectorIndex",
    "HybridRetriever",
    "UnifiedRetriever",
    "NeuralSemanticEmbeddingProvider",
]


