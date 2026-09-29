"""
Secure Physical Transport & Wire Protocol Subsystem (Step 30).

CRITICAL ARCHITECTURAL AXIOMS:
1. TRANSPORT != AUTHORITY:
   Transports convey WireEnvelope containers; they never authorize actions or bypass policies.
2. TRANSPORT != TRUST:
   A reachable or connected transport endpoint confers zero trust.
3. CRYPTOGRAPHIC INTEGRITY & FAIL-CLOSED SECURITY:
   Envelopes are signed, digest-verified, framed with length prefixes, and protected against replay.
"""

from chakrview.cognition.transport.errors import (
    TransportError,
    TransportTimeoutError,
    TransportUnavailableError,
    TransportProtocolError,
    FrameError,
    OversizedPayloadError,
    ReplayAttackError,
    WireSecurityError,
)
from chakrview.cognition.transport.models import (
    MessageType,
    TransportStatus,
    TransportHealth,
    WireEnvelope,
    DEFAULT_MAX_PAYLOAD_BYTES,
    MAX_WIRE_FRAME_BYTES,
)
from chakrview.cognition.transport.base import Transport
from chakrview.cognition.transport.framing import LengthPrefixedFramer
from chakrview.cognition.transport.serialization import DeterministicWireSerializer
from chakrview.cognition.transport.loopback import LoopbackWireTransport
from chakrview.cognition.transport.tcp import TCPWireTransport
from chakrview.cognition.transport.http2 import HTTP2WireTransport, is_http2_available
from chakrview.cognition.transport.grpc import GRPCWireTransport, is_grpc_available
from chakrview.cognition.transport.registry import TransportRegistry
from chakrview.cognition.transport.security import (
    TLSMode,
    TLSProtocolVersion,
    CertificateLifecycleState,
    CertificateUsage,
    CertificateMetadata,
    PeerCertificateBinding,
    TransportSecurityError,
    TLSError,
    TLSConfigurationError,
    TLSHandshakeError,
    InsecureDowngradeError,
    CertificateError,
    CertificateValidationError,
    CertificateExpiredError,
    CertificateNotYetValidError,
    CertificateRevokedError,
    HostnameMismatchError,
    UntrustedCAError,
    ClientCertificateMissingError,
    PeerBindingMismatchError,
    SecureTransportPolicy,
    compute_certificate_fingerprint,
    parse_certificate_from_pem,
    parse_certificate_from_der,
    extract_certificate_metadata,
    inspect_certificate_lifecycle,
    CertificateRevocationRegistry,
    HermeticPKIBuilder,
    validate_certificate,
    validate_certificate_or_raise,
    TLSContextFactory,
    PeerCertificateBinder,
)

__all__ = [
    "TransportError",
    "TransportTimeoutError",
    "TransportUnavailableError",
    "TransportProtocolError",
    "FrameError",
    "OversizedPayloadError",
    "ReplayAttackError",
    "WireSecurityError",
    "MessageType",
    "TransportStatus",
    "TransportHealth",
    "WireEnvelope",
    "DEFAULT_MAX_PAYLOAD_BYTES",
    "MAX_WIRE_FRAME_BYTES",
    "Transport",
    "LengthPrefixedFramer",
    "DeterministicWireSerializer",
    "LoopbackWireTransport",
    "TCPWireTransport",
    "HTTP2WireTransport",
    "is_http2_available",
    "GRPCWireTransport",
    "is_grpc_available",
    "TransportRegistry",
    # Step 31 Transport Security
    "TLSMode",
    "TLSProtocolVersion",
    "CertificateLifecycleState",
    "CertificateUsage",
    "CertificateMetadata",
    "PeerCertificateBinding",
    "TransportSecurityError",
    "TLSError",
    "TLSConfigurationError",
    "TLSHandshakeError",
    "InsecureDowngradeError",
    "CertificateError",
    "CertificateValidationError",
    "CertificateExpiredError",
    "CertificateNotYetValidError",
    "CertificateRevokedError",
    "HostnameMismatchError",
    "UntrustedCAError",
    "ClientCertificateMissingError",
    "PeerBindingMismatchError",
    "SecureTransportPolicy",
    "compute_certificate_fingerprint",
    "parse_certificate_from_pem",
    "parse_certificate_from_der",
    "extract_certificate_metadata",
    "inspect_certificate_lifecycle",
    "CertificateRevocationRegistry",
    "HermeticPKIBuilder",
    "validate_certificate",
    "validate_certificate_or_raise",
    "TLSContextFactory",
    "PeerCertificateBinder",
]

