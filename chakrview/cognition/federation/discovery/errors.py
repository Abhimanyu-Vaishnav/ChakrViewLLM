"""
Exception Hierarchy for ChakrView Federation Discovery & Secure Membership (Step 35).

Enforces fail-closed semantics across discovery, authentication, membership,
quarantine, and heartbeat subsystems.
"""

from typing import Optional


class FederationDiscoveryError(Exception):
    """Base exception for all federation discovery and membership errors."""
    pass


class EndpointValidationError(FederationDiscoveryError):
    """Raised when a node endpoint format, address, port, or protocol is invalid."""
    pass


class CandidateRegistrationError(FederationDiscoveryError):
    """Raised when registering a candidate node violates discovery bounds or uniqueness."""
    pass


class MembershipError(FederationDiscoveryError):
    """Base exception for membership management failures."""
    pass


class MembershipStateTransitionError(MembershipError):
    """Raised when an illegal membership lifecycle transition is attempted."""
    pass


class MembershipCapacityError(MembershipError):
    """Raised when active membership exceeds maximum configured member capacity."""
    pass


class UnknownPeerError(MembershipError):
    """Raised when an unrecognized or unauthenticated peer attempts connection or operations."""
    pass


class QuarantineError(MembershipError):
    """Raised when an operation is attempted on or by a quarantined node."""
    pass


class MembershipRevocationError(MembershipError):
    """Raised when revocation constraints or terminal state invariants are violated."""
    pass


class AuthenticationGateError(FederationDiscoveryError):
    """Raised when a candidate fails cryptographic identity or TLS certificate authentication."""
    pass


class CertificateBindingMismatchError(AuthenticationGateError):
    """Raised when the TLS certificate does not match the expected peer binding."""
    pass


class HeartbeatError(FederationDiscoveryError):
    """Base exception for federation heartbeat operations."""
    pass


class HeartbeatTimeoutError(HeartbeatError):
    """Raised when a member node fails to respond within the heartbeat deadline."""
    pass


class HeartbeatValidationError(HeartbeatError):
    """Raised when a heartbeat response contains divergent digest or invalid credentials."""
    pass


class RejoinDeniedError(MembershipError):
    """Raised when a rejoining node fails membership, revocation, or replay validation."""
    pass


# Convenient aliases for testing and external consumers
DiscoveryError = FederationDiscoveryError
MalformedEndpointError = EndpointValidationError
UnsupportedProtocolError = EndpointValidationError
DuplicateCandidateError = CandidateRegistrationError
DiscoveryCapacityError = CandidateRegistrationError
InvalidMembershipTransitionError = MembershipStateTransitionError
RevocationError = MembershipRevocationError
HeartbeatSpoofingError = HeartbeatValidationError
