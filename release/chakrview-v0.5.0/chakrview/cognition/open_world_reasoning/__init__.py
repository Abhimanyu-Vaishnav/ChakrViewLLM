"""
ChakrView Waves 571-580: Open-World Robust Reasoning & Evidence Uncertainty Core.
"""

from .evidence_quality import (
    EvidenceQualityGrade,
    EvidenceQualityReport,
    EvidenceQualityEvaluator,
)
from .partial_evidence import (
    CompletenessLevel,
    CompletenessAssessment,
    PartialEvidenceReasoner,
)
from .ambiguity_resolver import (
    AmbiguityType,
    EntityResolutionState,
    EntityDisambiguationResult,
    AmbiguityResolutionEngine,
)
from .conflict_fusion import (
    ConflictType,
    SourceLineageRecord,
    ConflictClassificationResult,
    ConflictingEvidenceFuser,
)
from .multidoc_graph import (
    DocumentNode,
    ClaimEdge,
    MultiDocumentEvidenceGraph,
    MultiDocumentReasoner,
)
from .adversarial_shield import (
    AdversarialShieldResult,
    EvidenceDataSanitizer,
)
from .temporal_state import (
    TemporalScope,
    TemporalStateTransition,
    TemporalWorldStateReasoner,
)
from .evidence_budget import (
    ResearchBudget,
    ResourceCostTracker,
    BudgetStoppingPolicy,
    BudgetDecision,
    ResourceAwareEvidenceGovernor,
)
from .research_planner import (
    EpistemicKnowledgeState,
    ResearchRequirement,
    OpenWorldResearchPlanner,
)
from .open_world_master_gate import (
    OpenWorldRobustCognitionMasterGate,
    MasterEvaluationRecord,
)

__all__ = [
    "EvidenceQualityGrade",
    "EvidenceQualityReport",
    "EvidenceQualityEvaluator",
    "CompletenessLevel",
    "CompletenessAssessment",
    "PartialEvidenceReasoner",
    "AmbiguityType",
    "EntityResolutionState",
    "EntityDisambiguationResult",
    "AmbiguityResolutionEngine",
    "ConflictType",
    "SourceLineageRecord",
    "ConflictClassificationResult",
    "ConflictingEvidenceFuser",
    "DocumentNode",
    "ClaimEdge",
    "MultiDocumentEvidenceGraph",
    "MultiDocumentReasoner",
    "AdversarialShieldResult",
    "EvidenceDataSanitizer",
    "TemporalScope",
    "TemporalStateTransition",
    "TemporalWorldStateReasoner",
    "ResearchBudget",
    "ResourceCostTracker",
    "BudgetStoppingPolicy",
    "BudgetDecision",
    "ResourceAwareEvidenceGovernor",
    "EpistemicKnowledgeState",
    "ResearchRequirement",
    "OpenWorldResearchPlanner",
    "OpenWorldRobustCognitionMasterGate",
    "MasterEvaluationRecord",
]
