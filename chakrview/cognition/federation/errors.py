"""
Strongly Typed Error Hierarchy for Distributed Federation Coordination (Step 33).

CRITICAL ARCHITECTURAL AXIOMS:
1. FAIL-CLOSED COORDINATION:
   Any ambiguity, malformed message, state divergence, or replay ambiguity fails closed.
2. FEDERATION_COORDINATION != AUTHORITY:
   Errors in coordination abort synchronization; they never bypass local CapabilityGate.
"""


class FederationCoordinationError(Exception):
    """Base exception for all distributed federation coordination failures."""
    pass


class EngineIdentityError(FederationCoordinationError):
    """Raised when engine identity is malformed, invalid, or forged."""
    pass


class ProtocolMismatchError(FederationCoordinationError):
    """Raised when a remote engine presents an incompatible protocol version."""
    pass


class HandshakeError(FederationCoordinationError):
    """Raised when a federation handshake fails validation or is rejected."""
    pass


class StateVersionError(FederationCoordinationError):
    """Raised when a state version violation occurs (e.g., negative or regressing version)."""
    pass


class StaleStateError(FederationCoordinationError):
    """Raised when received state is stale or attempts an epoch/version regression."""
    pass


class StateDigestConflictError(FederationCoordinationError):
    """Raised when two states claim the same version but compute divergent cryptographic digests."""
    pass


class ReplaySyncError(FederationCoordinationError):
    """Raised when replay-state synchronization encounters conflicts or security violations."""
    pass


class TrustSyncError(FederationCoordinationError):
    """Raised when remote trust claims violate local policy or attempt self-escalation."""
    pass


class RevocationSyncError(FederationCoordinationError):
    """Raised when revocation propagation encounters malformed or inconsistent records."""
    pass


class CoordinationCapacityError(FederationCoordinationError):
    """Raised when coordination engine limits (e.g. max engines or message limits) are exceeded."""
    pass
