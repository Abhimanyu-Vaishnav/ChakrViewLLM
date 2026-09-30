"""
Typed error hierarchy for the Federated Cognitive Orchestration layer (Step 42).

All errors are fail-closed: unrecognised failures raise the most restrictive
base class rather than silently passing.
"""


class FederatedCognitiveError(Exception):
    """Base error for all federated cognitive orchestration failures."""


class CognitiveContextOverflowError(FederatedCognitiveError):
    """Raised when a CognitiveContextEnvelope exceeds the 512-token ceiling."""


class CognitiveContextTenantViolationError(FederatedCognitiveError):
    """Raised when an envelope tenant_id mismatches the expected tenant."""


class CognitiveStepDependencyError(FederatedCognitiveError):
    """Raised when a step transition violates its dependency contract."""


class CognitiveGraphCycleError(FederatedCognitiveError):
    """Raised when the CognitiveTaskGraph contains a directed cycle."""


class CognitiveEpisodeStateError(FederatedCognitiveError):
    """Raised when an invalid episode state transition is attempted."""


class CognitiveEpisodeNotFoundError(FederatedCognitiveError):
    """Raised when an episode lookup finds no matching ID."""


class CognitiveSynthesisError(FederatedCognitiveError):
    """Raised when synthesis over distributed results is impossible."""


class NeuralCapabilityContextOverflowError(FederatedCognitiveError):
    """Raised when total token count exceeds the 512-token ceiling at inference time."""


class NeuralWeightMutationError(FederatedCognitiveError):
    """Raised when the neural weight hash changes after a forward pass (ΔW != 0)."""


class SecretLeakageInContextError(FederatedCognitiveError):
    """Raised when prohibited secrets are detected in a cognitive context envelope."""
