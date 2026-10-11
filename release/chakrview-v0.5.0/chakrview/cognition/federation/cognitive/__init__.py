"""
Federated Cognitive Orchestration & Distributed Reasoning Graph (Step 42).

Public API for the cognitive federation layer bridging:
    Distributed Runtime (Step 41) -> Distributed Cognitive System
"""

from chakrview.cognition.federation.cognitive.errors import (
    FederatedCognitiveError,
    CognitiveContextOverflowError,
    CognitiveContextTenantViolationError,
    CognitiveStepDependencyError,
    CognitiveGraphCycleError,
    CognitiveEpisodeStateError,
    CognitiveEpisodeNotFoundError,
    CognitiveSynthesisError,
    NeuralCapabilityContextOverflowError,
    NeuralWeightMutationError,
    SecretLeakageInContextError,
)
from chakrview.cognition.federation.cognitive.models import (
    CognitiveRole,
    CognitiveStepState,
    CognitiveEpisodeState,
    CognitiveStep,
    CognitiveTaskGraph,
    CognitiveContextEnvelope,
    ConflictRecord,
    SynthesisResult,
    CognitiveEpisode,
    CAPABILITY_NEURAL_INFERENCE,
    CAPABILITY_ANALYST,
    CAPABILITY_RESEARCHER,
    CAPABILITY_CRITIC,
    CAPABILITY_SYNTHESIZER,
    CAPABILITY_VERIFIER,
    MAX_CONTEXT_TOKENS,
    MAX_ENVELOPE_CONTEXT_TOKENS,
)
from chakrview.cognition.federation.cognitive.capabilities import (
    FederatedNeuralCapability,
    SimpleCognitiveCapability,
    GovernedKnowledgeRetrievalCapability,
)
from chakrview.cognition.federation.cognitive.bridge import FederatedReasoningBridge
from chakrview.cognition.federation.cognitive.synthesis import CognitiveSynthesisEngine
from chakrview.cognition.federation.cognitive.episode import CognitiveEpisodeManager
from chakrview.cognition.federation.cognitive.engine import FederatedCognitiveEngine
from chakrview.cognition.federation.cognitive.memory import PersistentCognitiveMemoryAdapter

__all__ = [
    # Errors
    "FederatedCognitiveError",
    "CognitiveContextOverflowError",
    "CognitiveContextTenantViolationError",
    "CognitiveStepDependencyError",
    "CognitiveGraphCycleError",
    "CognitiveEpisodeStateError",
    "CognitiveEpisodeNotFoundError",
    "CognitiveSynthesisError",
    "NeuralCapabilityContextOverflowError",
    "NeuralWeightMutationError",
    "SecretLeakageInContextError",
    # Models
    "CognitiveRole",
    "CognitiveStepState",
    "CognitiveEpisodeState",
    "CognitiveStep",
    "CognitiveTaskGraph",
    "CognitiveContextEnvelope",
    "ConflictRecord",
    "SynthesisResult",
    "CognitiveEpisode",
    "CAPABILITY_NEURAL_INFERENCE",
    "CAPABILITY_ANALYST",
    "CAPABILITY_RESEARCHER",
    "CAPABILITY_CRITIC",
    "CAPABILITY_SYNTHESIZER",
    "CAPABILITY_VERIFIER",
    "MAX_CONTEXT_TOKENS",
    "MAX_ENVELOPE_CONTEXT_TOKENS",
    # Components
    "FederatedNeuralCapability",
    "SimpleCognitiveCapability",
    "GovernedKnowledgeRetrievalCapability",
    "FederatedReasoningBridge",
    "CognitiveSynthesisEngine",
    "CognitiveEpisodeManager",
    "FederatedCognitiveEngine",
    "PersistentCognitiveMemoryAdapter",
]
