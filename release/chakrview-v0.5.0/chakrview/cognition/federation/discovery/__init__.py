"""
ChakrView Federation Networking, Node Discovery & Secure Membership Subsystem (Step 35).

Provides explicit endpoint modeling, controlled candidate discovery, authenticated
membership management, heartbeats, and fail-closed rejoin/quarantine policies.
"""

from chakrview.cognition.federation.discovery.errors import (
    FederationDiscoveryError,
    EndpointValidationError,
    CandidateRegistrationError,
    MembershipError,
    MembershipStateTransitionError,
    MembershipCapacityError,
    UnknownPeerError,
    QuarantineError,
    MembershipRevocationError,
    AuthenticationGateError,
    CertificateBindingMismatchError,
    HeartbeatError,
    HeartbeatTimeoutError,
    HeartbeatValidationError,
    RejoinDeniedError,
    DiscoveryError,
    MalformedEndpointError,
    UnsupportedProtocolError,
    DuplicateCandidateError,
    DiscoveryCapacityError,
    InvalidMembershipTransitionError,
    RevocationError,
    HeartbeatSpoofingError,
)
from chakrview.cognition.federation.discovery.models import (
    NodeAddress,
    NodeProtocol,
    NodeDiscoverySource,
    MembershipState,
    VALID_MEMBERSHIP_TRANSITIONS,
    FederationNodeEndpoint,
    FederationNodeCandidate,
    FederationNodeMembership,
    MAX_MEMBERSHIP_NODES,
    DEFAULT_HEARTBEAT_INTERVAL_SECONDS,
    DEFAULT_HEARTBEAT_TIMEOUT_SECONDS,
    DEFAULT_MAX_MISSED_HEARTBEATS,
)
from chakrview.cognition.federation.discovery.provider import (
    DiscoveryProvider,
    StaticConfigDiscoveryProvider,
    FileConfigDiscoveryProvider,
    InProcessAdvertisementDiscoveryProvider,
    CompositeDiscoveryService,
)
from chakrview.cognition.federation.discovery.membership import FederationMembershipManager
from chakrview.cognition.federation.discovery.heartbeat import (
    HeartbeatPayload,
    FederationHeartbeatMonitor,
)
from chakrview.cognition.federation.discovery.connection import FederationConnectionManager

__all__ = [
    # Errors
    "FederationDiscoveryError",
    "EndpointValidationError",
    "CandidateRegistrationError",
    "MembershipError",
    "MembershipStateTransitionError",
    "MembershipCapacityError",
    "UnknownPeerError",
    "QuarantineError",
    "MembershipRevocationError",
    "AuthenticationGateError",
    "CertificateBindingMismatchError",
    "HeartbeatError",
    "HeartbeatTimeoutError",
    "HeartbeatValidationError",
    "RejoinDeniedError",
    "DiscoveryError",
    "MalformedEndpointError",
    "UnsupportedProtocolError",
    "DuplicateCandidateError",
    "DiscoveryCapacityError",
    "InvalidMembershipTransitionError",
    "RevocationError",
    "HeartbeatSpoofingError",
    # Models
    "NodeAddress",
    "NodeProtocol",
    "NodeDiscoverySource",
    "MembershipState",
    "VALID_MEMBERSHIP_TRANSITIONS",
    "FederationNodeEndpoint",
    "FederationNodeCandidate",
    "FederationNodeMembership",
    "MAX_MEMBERSHIP_NODES",
    "DEFAULT_HEARTBEAT_INTERVAL_SECONDS",
    "DEFAULT_HEARTBEAT_TIMEOUT_SECONDS",
    "DEFAULT_MAX_MISSED_HEARTBEATS",
    # Discovery Providers
    "DiscoveryProvider",
    "StaticConfigDiscoveryProvider",
    "FileConfigDiscoveryProvider",
    "InProcessAdvertisementDiscoveryProvider",
    "CompositeDiscoveryService",
    # Managers & Monitors
    "FederationMembershipManager",
    "HeartbeatPayload",
    "FederationHeartbeatMonitor",
    "FederationConnectionManager",
]
