"""
ChakrView Persistent Project Brain (PPB) Package.
Steps 78–81 Architecture:
- Step 78: Persistent Project Brain Foundation (Models, Storage, State, Identity).
- Step 79: Incremental Project Scanning (Bounded, Resumable, Progress Tracking).
- Step 80: Project Knowledge Index & Retrieval (Targeted, Low-overhead, Provenance-grounded).
- Step 81: Change-Aware Brain Maintenance (Selective Invalidation, Revalidation, Historical Versioning).
"""

from chakrview.cognition.ppb.models import (
    CURRENT_SCHEMA_VERSION,
    EpistemicStatus,
    KnowledgeRecordType,
    KnowledgeRecord,
    ProjectIdentity,
    ProjectBrainState,
    PPBStorageSchemaError,
)
from chakrview.cognition.ppb.storage import PersistentBrainStorage
from chakrview.cognition.ppb.brain import PersistentProjectBrain
from chakrview.cognition.ppb.scanner import (
    ScanBatchProgress,
    IncrementalProjectScanner,
)
from chakrview.cognition.ppb.retrieval import (
    PPBRetrievalBudget,
    RetrievedKnowledgeBundle,
    ProjectKnowledgeRetriever,
)
from chakrview.cognition.ppb.maintainer import (
    MaintenanceReport,
    ChangeAwareBrainMaintainer,
)

__all__ = [
    "CURRENT_SCHEMA_VERSION",
    "EpistemicStatus",
    "KnowledgeRecordType",
    "KnowledgeRecord",
    "ProjectIdentity",
    "ProjectBrainState",
    "PPBStorageSchemaError",
    "PersistentBrainStorage",
    "PersistentProjectBrain",
    "ScanBatchProgress",
    "IncrementalProjectScanner",
    "PPBRetrievalBudget",
    "RetrievedKnowledgeBundle",
    "ProjectKnowledgeRetriever",
    "MaintenanceReport",
    "ChangeAwareBrainMaintainer",
]
