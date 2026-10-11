"""
ChakrView Persistent Personal Memory, Consolidation & Learning Foundation (Step 16).

Provides:
- Memory Taxonomy: Episodic, Semantic, User Profile, Working, Knowledge
- Storage Abstraction with strict multi-user privacy isolation
- Dual-metric importance & confidence scoring
- Multi-tier duplicate detection (exact, normalized, semantic)
- Conflict detection and tracking
- Memory consolidation engine preserving provenance
- Temporal memory, expiration, and revision history
- Multi-criteria hybrid retrieval (lexical, semantic, recency, importance, confidence)
- Cross-conversation recall and case/document comparison
- Governed learning feedback recording without model weight mutation
- Data != Authority security enforcement
"""

from chakrview.memory.record import (
    MemoryType,
    MemoryValidity,
    MemoryProvenance,
    TemporalMetadata,
    MemoryRecord,
)
from chakrview.memory.store import (
    MemoryStore,
    InMemoryMemoryStore,
    MemoryStoreError,
    MemoryNotFoundError,
    MemoryAccessDeniedError,
)
from chakrview.memory.scoring import (
    MemoryScorer,
    MemoryScoreResult,
)
from chakrview.memory.deduplication import (
    MatchLevel,
    DeduplicationResult,
    MemoryDeduplicator,
)
from chakrview.memory.conflict import (
    ConflictStatus,
    MemoryConflict,
    ConflictDetector,
)
from chakrview.memory.consolidation import (
    ConsolidationCandidate,
    MemoryConsolidator,
)
from chakrview.memory.temporal import (
    TemporalLineageNode,
    TemporalMemoryManager,
)
from chakrview.memory.retriever import (
    MemoryRetrievalCandidate,
    PersistentMemoryRetriever,
)
from chakrview.memory.comparison import (
    ComparisonFacet,
    CaseComparisonResult,
    CaseComparator,
)
from chakrview.memory.learning import (
    FeedbackSignal,
    LearningCategory,
    LearningCandidate,
    LearningFeedbackManager,
)
from chakrview.memory.security import (
    SecurityViolationError,
    MemorySecurityPolicy,
)
from chakrview.memory.adapter import (
    WorkingMemoryAdapter,
    KnowledgeMemoryAdapter,
)
from chakrview.memory.manager import (
    PersonalMemoryManager,
)

# Step 24 Continual Cognition & Memory Foundation
from chakrview.memory.models import (
    Episode,
    SemanticMemory,
    MemoryContradiction,
    MemoryProvenanceSource,
    MemoryVerificationState,
    MemoryLifecycleStatus,
    ContradictionResolutionState,
    MemoryRetrievalQuery,
    MemoryRetrievalCandidate as ContinualMemoryCandidate,
    MemoryRetrievalResult,
)
from chakrview.memory.working import (
    WorkingMemory,
    WorkingMemoryConfig,
)
from chakrview.memory.episodic import (
    EpisodicMemoryStore,
)
from chakrview.memory.semantic import (
    SemanticMemoryStore,
)
from chakrview.memory.contradiction import (
    ContradictionManager,
)
from chakrview.memory.retrieval import (
    ContinualMemoryRetriever,
)
from chakrview.memory.consolidation import (
    ExperienceConsolidationEngine,
)
from chakrview.memory.lifecycle import (
    MemoryLifecycleManager,
    LifecycleAuditEntry,
)
from chakrview.memory.policy import (
    MemoryExecutionPolicy,
)
from chakrview.memory.governance import (
    MemoryGovernanceBridge,
    MemoryGovernanceError,
)
from chakrview.memory.storage import (
    ContinualMemoryStorage,
    MemoryStorageSchemaError,
)
from chakrview.memory.engine import (
    ContinualCognitionEngine,
)

__all__ = [
    # Step 16 Exports
    "MemoryType",
    "MemoryValidity",
    "MemoryProvenance",
    "TemporalMetadata",
    "MemoryRecord",
    "MemoryStore",
    "InMemoryMemoryStore",
    "MemoryStoreError",
    "MemoryNotFoundError",
    "MemoryAccessDeniedError",
    "MemoryScorer",
    "MemoryScoreResult",
    "MatchLevel",
    "DeduplicationResult",
    "MemoryDeduplicator",
    "ConflictStatus",
    "MemoryConflict",
    "ConflictDetector",
    "ConsolidationCandidate",
    "MemoryConsolidator",
    "TemporalLineageNode",
    "TemporalMemoryManager",
    "MemoryRetrievalCandidate",
    "PersistentMemoryRetriever",
    "ComparisonFacet",
    "CaseComparisonResult",
    "CaseComparator",
    "FeedbackSignal",
    "LearningCategory",
    "LearningCandidate",
    "LearningFeedbackManager",
    "SecurityViolationError",
    "MemorySecurityPolicy",
    "WorkingMemoryAdapter",
    "KnowledgeMemoryAdapter",
    "PersonalMemoryManager",
    # Step 24 Exports
    "Episode",
    "SemanticMemory",
    "MemoryContradiction",
    "MemoryProvenanceSource",
    "MemoryVerificationState",
    "MemoryLifecycleStatus",
    "ContradictionResolutionState",
    "MemoryRetrievalQuery",
    "ContinualMemoryCandidate",
    "MemoryRetrievalResult",
    "WorkingMemory",
    "WorkingMemoryConfig",
    "EpisodicMemoryStore",
    "SemanticMemoryStore",
    "ContradictionManager",
    "ContinualMemoryRetriever",
    "ExperienceConsolidationEngine",
    "MemoryLifecycleManager",
    "LifecycleAuditEntry",
    "MemoryExecutionPolicy",
    "MemoryGovernanceBridge",
    "MemoryGovernanceError",
    "ContinualMemoryStorage",
    "MemoryStorageSchemaError",
    "ContinualCognitionEngine",
]
