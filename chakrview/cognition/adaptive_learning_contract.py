"""
ChakrView v0.3.0 Adaptive Learning Contract.
Wave 520: Formalizes the boundary taxonomy between:
1. EPISODIC EXPERIENCE: Raw, un-abstracted single-run execution history.
2. ABSTRACT EXPERIENCE: Generalized causal lesson derived across multiple episodes.
3. STRATEGY: Action policy / execution dispatch schema for future tasks.
4. KNOWLEDGE: Verified factual / relational assertions.
5. PARAMETRIC UPDATE: Changes to trainable neural weights outside canonical baseline.

System Invariant:
The system must NEVER silently treat one category as another.
Memory lookup != neural learning.
Experience memorization != abstraction.
Canonical baseline weights (3,443,136 params, SHA c5571c...00a282da) remain immutable (Delta W = 0).
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Any, Optional
import time
import hashlib
import json

class LearningArtifactType(str, Enum):
    EPISODIC_EXPERIENCE = "EPISODIC_EXPERIENCE"
    ABSTRACT_EXPERIENCE = "ABSTRACT_EXPERIENCE"
    STRATEGY = "STRATEGY"
    KNOWLEDGE = "KNOWLEDGE"
    PARAMETRIC_UPDATE = "PARAMETRIC_UPDATE"

@dataclass(frozen=True)
class ProvenanceMetadata:
    runtime_version: str = "v0.3.0_adaptive_intelligence"
    benchmark_version: str = "v0.3_adaptive_benchmark"
    candidate_id: str = "v0.3_candidate_default"
    seed: int = 42
    config_hash: str = "c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da"
    timestamp: float = field(default_factory=time.time)

@dataclass
class EpisodicExperience:
    """Raw single-task trajectory execution history."""
    episode_id: str
    task_id: str
    task_structure: Dict[str, Any]
    strategy_used: str
    execution_steps: List[Dict[str, Any]]
    success: bool
    confidence: float
    error_type: Optional[str] = None
    provenance: ProvenanceMetadata = field(default_factory=ProvenanceMetadata)
    artifact_type: LearningArtifactType = LearningArtifactType.EPISODIC_EXPERIENCE

    def compute_hash(self) -> str:
        payload = json.dumps({
            "episode_id": self.episode_id,
            "task_id": self.task_id,
            "task_structure": self.task_structure,
            "strategy": self.strategy_used,
            "success": self.success,
            "confidence": self.confidence,
            "error_type": self.error_type
        }, sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

@dataclass
class AbstractExperience:
    """Generalized causal lesson derived from >=2 structurally related episodes."""
    abstraction_id: str
    task_family: str
    supporting_episode_ids: List[str]
    contradictory_episode_ids: List[str]
    success_pattern: str
    failure_pattern: str
    recommended_strategy: str
    confidence: float
    sample_count: int
    provenance: ProvenanceMetadata = field(default_factory=ProvenanceMetadata)
    artifact_type: LearningArtifactType = LearningArtifactType.ABSTRACT_EXPERIENCE

    def validate_abstraction(self) -> bool:
        """Enforces rule: cannot claim abstraction from fewer than 2 distinct episodes."""
        return len(self.supporting_episode_ids) >= 2 and self.sample_count >= 2

@dataclass
class StrategyPolicy:
    """Action policy for choosing execution paths given a task specification."""
    strategy_name: str
    task_family_mapping: Dict[str, float]  # task_family -> expected_utility
    retry_budget: int
    direct_route_threshold: float
    recurrent_steps_cap: int
    peer_assistance_allowed: bool
    provenance: ProvenanceMetadata = field(default_factory=ProvenanceMetadata)
    artifact_type: LearningArtifactType = LearningArtifactType.STRATEGY

@dataclass
class VerifiedKnowledge:
    """Grounded semantic relation or state assertion verified by governance."""
    knowledge_id: str
    subject: str
    relation: str
    target: str
    grounding_confidence: float
    verification_source: str
    provenance: ProvenanceMetadata = field(default_factory=ProvenanceMetadata)
    artifact_type: LearningArtifactType = LearningArtifactType.KNOWLEDGE

@dataclass
class ParametricUpdateRecord:
    """Isolated neural parameter delta outside canonical baseline."""
    update_id: str
    target_module: str  # e.g. "CandidateRelationalAdapter", NEVER "ChakrMicroBaseline"
    param_count: int
    param_delta_norm: float
    baseline_delta_w: float = 0.0  # Must be strictly 0.0
    checkpoint_path: str = ""
    provenance: ProvenanceMetadata = field(default_factory=ProvenanceMetadata)
    artifact_type: LearningArtifactType = LearningArtifactType.PARAMETRIC_UPDATE

class AdaptiveLearningContract:
    """
    Governance authority ensuring strict type separation and invariant safety.
    """
    @staticmethod
    def verify_artifact_isolation(artifact: Any) -> bool:
        if isinstance(artifact, EpisodicExperience):
            return artifact.artifact_type == LearningArtifactType.EPISODIC_EXPERIENCE
        elif isinstance(artifact, AbstractExperience):
            if not artifact.validate_abstraction():
                raise ValueError("AbstractExperience requires >= 2 supporting episodes to qualify as abstraction.")
            return artifact.artifact_type == LearningArtifactType.ABSTRACT_EXPERIENCE
        elif isinstance(artifact, StrategyPolicy):
            return artifact.artifact_type == LearningArtifactType.STRATEGY
        elif isinstance(artifact, VerifiedKnowledge):
            return artifact.artifact_type == LearningArtifactType.KNOWLEDGE
        elif isinstance(artifact, ParametricUpdateRecord):
            if artifact.baseline_delta_w != 0.0 or artifact.target_module == "ChakrMicroBaseline":
                raise ValueError("Invariant violation: Canonical baseline cannot receive parametric updates.")
            return artifact.artifact_type == LearningArtifactType.PARAMETRIC_UPDATE
        return False
