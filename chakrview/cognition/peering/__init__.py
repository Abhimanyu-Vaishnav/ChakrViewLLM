"""
Cross-Zone Peering & Trust Negotiation Subsystem for ChakrView (Step 29).

Provides governed architectural primitives for multi-zone cognitive federation:
- Deterministic Peer Identity & Fingerprinting (IDENTITY != CRYPTOGRAPHIC_AUTHENTICATION)
- Deterministic Discovery (DISCOVERY != TRUST)
- Structured Attestation Claims & Architectural Verifiers
- Bounded, Scoped, Expiring Trust Grants (PEER_TRUST != PEER_AUTHORITY)
- Default-Deny Federation Policies & Prohibited Scope Enforcement
- Deterministic Trust & Scope Negotiation
- Fail-Closed Revocation & Logical Epoch Expiration
- Cross-Tenant & Cross-Zone Boundary Isolation Guard
- Bounded & Sanitized Audit Telemetry
- Central CrossZoneFederationEngine mediating all capability execution via CapabilityGate
"""

from chakrview.cognition.peering.models import (
    PeerIdentity,
    PeerAttestation,
    PeerDeclaration,
    PeerRegistration,
    TrustGrant,
    TrustLevel,
    TrustStatus,
    DiscoveryStatus,
    AttestationStatus,
    FederationScope,
    ProhibitedScope,
    NegotiationAgreement,
    RevocationRecord,
    AuditEventType,
    AuditRecord,
    SafePublicPeeringTrace,
    MAX_PEERS_TOTAL,
    MAX_PEERS_PER_ZONE,
    MAX_ACTIVE_PEERS,
    MAX_DELEGATION_DEPTH,
    DEFAULT_TRUST_TTL_EPOCHS,
    MAX_AUDIT_LOG_ENTRIES,
)
from chakrview.cognition.peering.identity import (
    PeerIdentityProvider,
    IdentityValidationError,
    UnsupportedSecurityModeError,
)
from chakrview.cognition.peering.attestation import (
    PeerAttestationVerifier,
    AttestationVerificationError,
)
from chakrview.cognition.peering.policy import (
    CrossZoneFederationPolicy,
    PolicyViolationError,
)
from chakrview.cognition.peering.trust import TrustModel
from chakrview.cognition.peering.discovery import PeerDiscoveryManager
from chakrview.cognition.peering.negotiation import TrustNegotiator
from chakrview.cognition.peering.registry import (
    PeerRegistry,
    DuplicatePeerError,
    PeerRegistryCapacityError,
)
from chakrview.cognition.peering.revocation import RevocationManager
from chakrview.cognition.peering.isolation import (
    CrossZoneIsolationGuard,
    IsolationViolationError,
)
from chakrview.cognition.peering.audit import BoundedAuditLogger
from chakrview.cognition.peering.engine import (
    CrossZoneFederationEngine,
    CrossZoneAuthorizationError,
    WeightMutationDetectedError,
)

__all__ = [
    # Models & Ceilings
    "PeerIdentity",
    "PeerAttestation",
    "PeerDeclaration",
    "PeerRegistration",
    "TrustGrant",
    "TrustLevel",
    "TrustStatus",
    "DiscoveryStatus",
    "AttestationStatus",
    "FederationScope",
    "ProhibitedScope",
    "NegotiationAgreement",
    "RevocationRecord",
    "AuditEventType",
    "AuditRecord",
    "SafePublicPeeringTrace",
    "MAX_PEERS_TOTAL",
    "MAX_PEERS_PER_ZONE",
    "MAX_ACTIVE_PEERS",
    "MAX_DELEGATION_DEPTH",
    "DEFAULT_TRUST_TTL_EPOCHS",
    "MAX_AUDIT_LOG_ENTRIES",
    # Identity
    "PeerIdentityProvider",
    "IdentityValidationError",
    "UnsupportedSecurityModeError",
    # Attestation
    "PeerAttestationVerifier",
    "AttestationVerificationError",
    # Policy
    "CrossZoneFederationPolicy",
    "PolicyViolationError",
    # Trust
    "TrustModel",
    # Discovery
    "PeerDiscoveryManager",
    # Negotiation
    "TrustNegotiator",
    # Registry
    "PeerRegistry",
    "DuplicatePeerError",
    "PeerRegistryCapacityError",
    # Revocation
    "RevocationManager",
    # Isolation
    "CrossZoneIsolationGuard",
    "IsolationViolationError",
    # Audit
    "BoundedAuditLogger",
    # Central Engine
    "CrossZoneFederationEngine",
    "CrossZoneAuthorizationError",
    "WeightMutationDetectedError",
]
