"""
Distributed Federation Coordination Subsystem for ChakrView (Step 33).

Provides bounded, deterministic, multi-engine federation coordination:
- Engine Identity (strictly separated from Peer, Session, Key, and Cert identity)
- Monotonic Security State Versions
- Deterministic Cryptographic State Digests (Replay, Trust, Revocation, Peer, Composite)
- Bounded Replay State Synchronization
- Bounded Trust State Consistency (REMOTE_CLAIM != LOCAL_AUTHORITY)
- Monotonic, Idempotent Revocation Propagation (REVOKED -> NEVER ACTIVE AGAIN)
- Coordination Handshake Protocol
- Distributed Federation Coordinator

CRITICAL ARCHITECTURAL AXIOMS:
- LOCAL_AUTHORITY > PEER_AUTHORITY & REMOTE_ENGINE != LOCAL_AUTHORITY
- FEDERATION_COORDINATION != AUTHORITY_TRANSFER
- FEDERATION_HANDSHAKE != TRUST_GRANT
- ZERO SECRET LEAKAGE: Private keys and session secrets are never shared.
- ZERO NEURAL MUTATION: Model weights remain strictly frozen (ΔW = 0).
"""

from chakrview.cognition.federation.models import (
    FederationEngineIdentity,
    SecurityStateVersion,
    ReplayStateDigest,
    TrustStateDigest,
    RevocationStateDigest,
    PeerStateDigest,
    FederationSecurityStateDigest,
    HandshakeStatus,
    FederationHandshakeRequest,
    FederationHandshakeResponse,
    ReplaySyncMessage,
    TrustSyncRecord,
    TrustSyncMessage,
    RevocationSyncRecord,
    RevocationSyncMessage,
    RevocationTargetType,
    MAX_FEDERATION_ENGINES,
    MAX_SYNC_MESSAGE_IDS,
    MAX_REVOCATION_SYNC_BATCH,
    MAX_STATE_HISTORY,
    FEDERATION_PROTOCOL_VERSION,
)
from chakrview.cognition.federation.errors import (
    FederationCoordinationError,
    EngineIdentityError,
    ProtocolMismatchError,
    HandshakeError,
    StateVersionError,
    StaleStateError,
    StateDigestConflictError,
    ReplaySyncError,
    TrustSyncError,
    RevocationSyncError,
    CoordinationCapacityError,
)
from chakrview.cognition.federation.identity import FederationEngineIdentityProvider
from chakrview.cognition.federation.state import FederationStateManager
from chakrview.cognition.federation.replay_sync import ReplayStateSynchronizer
from chakrview.cognition.federation.trust_sync import TrustStateSynchronizer
from chakrview.cognition.federation.revocation_sync import RevocationStateSynchronizer
from chakrview.cognition.federation.handshake import FederationHandshakeManager
from chakrview.cognition.federation.coordinator import DistributedFederationCoordinator

__all__ = [
    # Models
    "FederationEngineIdentity",
    "SecurityStateVersion",
    "ReplayStateDigest",
    "TrustStateDigest",
    "RevocationStateDigest",
    "PeerStateDigest",
    "FederationSecurityStateDigest",
    "HandshakeStatus",
    "FederationHandshakeRequest",
    "FederationHandshakeResponse",
    "ReplaySyncMessage",
    "TrustSyncRecord",
    "TrustSyncMessage",
    "RevocationSyncRecord",
    "RevocationSyncMessage",
    "RevocationTargetType",
    "MAX_FEDERATION_ENGINES",
    "MAX_SYNC_MESSAGE_IDS",
    "MAX_REVOCATION_SYNC_BATCH",
    "MAX_STATE_HISTORY",
    "FEDERATION_PROTOCOL_VERSION",
    # Errors
    "FederationCoordinationError",
    "EngineIdentityError",
    "ProtocolMismatchError",
    "HandshakeError",
    "StateVersionError",
    "StaleStateError",
    "StateDigestConflictError",
    "ReplaySyncError",
    "TrustSyncError",
    "RevocationSyncError",
    "CoordinationCapacityError",
    # Components
    "FederationEngineIdentityProvider",
    "FederationStateManager",
    "ReplayStateSynchronizer",
    "TrustStateSynchronizer",
    "RevocationStateSynchronizer",
    "FederationHandshakeManager",
    "DistributedFederationCoordinator",
]
