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

__all__ = [
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
]
