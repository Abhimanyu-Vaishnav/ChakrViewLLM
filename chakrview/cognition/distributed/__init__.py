"""
Distributed Federated Cognition & Secure Agent Transport (Step 27).

Public exports for the distributed federation subsystem:
- Distributed Node Model (NodeIdentity, NodeRole, NodeStatus, NodeCapabilities, NodeRegistration, etc.)
- Transport Abstraction (Transport, LoopbackTransport, TransportResponse, TransportStatus, TransportError, etc.)
- Secure Message Envelope (DistributedMessageEnvelope, MessageHeader, MessageRoute, MessageIntegrity)
- Replay Protection & Cryptographic Identity (ReplayProtectionTracker, MessageSigner, MessageVerifier, etc.)
- Distributed Node Registry (DistributedNodeRegistry, DuplicateNodeError, NodeCapacityExceededError)
- Distributed Task Routing (DistributedTaskRouter, DistributedRouteDecision, NoEligibleNodeError)
- Resilience (TimeoutPolicy, RetryPolicy, CircuitBreaker, CircuitBreakerState)
- Distributed Execution Policy (DistributedExecutionPolicy)
- Observability (DistributedObservabilityMetrics)
- Distributed Engine (DistributedFederatedCognitionEngine)
"""

from chakrview.cognition.distributed.models import (
    MAX_FEDERATION_NODES,
    MAX_AGENTS_PER_NODE,
    MAX_FEDERATION_DEPTH,
    MAX_REMOTE_TASKS_PER_CYCLE,
    MAX_MESSAGE_HOPS,
    DEFAULT_MESSAGE_TTL_SECONDS,
    NodeRole,
    NodeStatus,
    NodeTrustState,
    TransportStatus,
    NodeIdentity,
    NodeEndpoint,
    NodeCapabilities,
    NodeResourceProfile,
    NodeHealth,
    NodeRegistration,
    MessageHeader,
    MessageRoute,
    MessageIntegrity,
    DistributedMessageEnvelope,
    DistributedRouteDecision,
    SafePublicDistributedTrace,
)
from chakrview.cognition.distributed.transport import (
    Transport,
    LoopbackTransport,
    TransportResponse,
    TransportError,
    TransportTimeout,
    TransportUnavailable,
    TransportProtocolError,
)
from chakrview.cognition.distributed.security import (
    SecurityError,
    ReplayAttackError,
    MessageExpiredError,
    ExcessiveHopsError,
    TenantRoutingError,
    MessageTamperingError,
    ReplayProtectionTracker,
    MessageSigner,
    MessageVerifier,
    DeterministicHmacMessageSigner,
    DeterministicHmacMessageVerifier,
    NodeIdentityProvider,
    LocalNodeIdentityProvider,
    AttestationProvider,
    LocalAttestationProvider,
    create_distributed_envelope,
)
from chakrview.cognition.distributed.registry import (
    RegistryError,
    NodeCapacityExceededError,
    DuplicateNodeError,
    NodeNotFoundError,
    DistributedNodeRegistry,
)
from chakrview.cognition.distributed.resilience import (
    CircuitBreakerState,
    TimeoutPolicy,
    RetryPolicy,
    FailureRecord,
    CircuitBreaker,
)
from chakrview.cognition.distributed.router import (
    RoutingError,
    NoEligibleNodeError,
    DistributedTaskRouter,
)
from chakrview.cognition.distributed.policy import DistributedExecutionPolicy
from chakrview.cognition.distributed.observability import DistributedObservabilityMetrics
from chakrview.cognition.distributed.engine import DistributedFederatedCognitionEngine

__all__ = [
    # Constants
    "MAX_FEDERATION_NODES",
    "MAX_AGENTS_PER_NODE",
    "MAX_FEDERATION_DEPTH",
    "MAX_REMOTE_TASKS_PER_CYCLE",
    "MAX_MESSAGE_HOPS",
    "DEFAULT_MESSAGE_TTL_SECONDS",
    # Enums
    "NodeRole",
    "NodeStatus",
    "NodeTrustState",
    "TransportStatus",
    "CircuitBreakerState",
    # Models
    "NodeIdentity",
    "NodeEndpoint",
    "NodeCapabilities",
    "NodeResourceProfile",
    "NodeHealth",
    "NodeRegistration",
    "MessageHeader",
    "MessageRoute",
    "MessageIntegrity",
    "DistributedMessageEnvelope",
    "DistributedRouteDecision",
    "SafePublicDistributedTrace",
    # Transport
    "Transport",
    "LoopbackTransport",
    "TransportResponse",
    "TransportError",
    "TransportTimeout",
    "TransportUnavailable",
    "TransportProtocolError",
    # Security
    "SecurityError",
    "ReplayAttackError",
    "MessageExpiredError",
    "ExcessiveHopsError",
    "TenantRoutingError",
    "MessageTamperingError",
    "ReplayProtectionTracker",
    "MessageSigner",
    "MessageVerifier",
    "DeterministicHmacMessageSigner",
    "DeterministicHmacMessageVerifier",
    "NodeIdentityProvider",
    "LocalNodeIdentityProvider",
    "AttestationProvider",
    "LocalAttestationProvider",
    "create_distributed_envelope",
    # Registry
    "RegistryError",
    "NodeCapacityExceededError",
    "DuplicateNodeError",
    "NodeNotFoundError",
    "DistributedNodeRegistry",
    # Resilience
    "TimeoutPolicy",
    "RetryPolicy",
    "FailureRecord",
    "CircuitBreaker",
    # Router
    "RoutingError",
    "NoEligibleNodeError",
    "DistributedTaskRouter",
    # Policy & Observability
    "DistributedExecutionPolicy",
    "DistributedObservabilityMetrics",
    # Engine
    "DistributedFederatedCognitionEngine",
]
