"""
Production Transport Security, TLS/mTLS & Certificate Lifecycle (Step 31).

Exports the transport security subsystem components:
- Models: TLSMode, TLSProtocolVersion, CertificateLifecycleState, CertificateMetadata, PeerCertificateBinding
- Errors: TLSError, CertificateValidationError, HostnameMismatchError, etc.
- Policy: SecureTransportPolicy
- Certificates: parse_certificate_from_pem, parse_certificate_from_der, compute_certificate_fingerprint,
                extract_certificate_metadata, CertificateRevocationRegistry, HermeticPKIBuilder
- Validation: validate_certificate, validate_certificate_or_raise
- TLS: TLSContextFactory
- Binding: PeerCertificateBinder
"""

from chakrview.cognition.transport.security.models import (
    TLSMode,
    TLSProtocolVersion,
    CertificateLifecycleState,
    CertificateUsage,
    CertificateMetadata,
    PeerCertificateBinding,
    DEFAULT_CERT_EXPIRY_WARNING_SECONDS,
    DEFAULT_TLS_HANDSHAKE_TIMEOUT_SECONDS,
)
from chakrview.cognition.transport.security.errors import (
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
)
from chakrview.cognition.transport.security.policy import (
    SecureTransportPolicy,
)
from chakrview.cognition.transport.security.certificates import (
    compute_certificate_fingerprint,
    parse_certificate_from_pem,
    parse_certificate_from_der,
    extract_certificate_metadata,
    inspect_certificate_lifecycle,
    CertificateRevocationRegistry,
    HermeticPKIBuilder,
)
from chakrview.cognition.transport.security.validation import (
    validate_certificate,
    validate_certificate_or_raise,
)
from chakrview.cognition.transport.security.tls import (
    TLSContextFactory,
)
from chakrview.cognition.transport.security.binding import (
    PeerCertificateBinder,
)

__all__ = [
    # Models
    "TLSMode",
    "TLSProtocolVersion",
    "CertificateLifecycleState",
    "CertificateUsage",
    "CertificateMetadata",
    "PeerCertificateBinding",
    "DEFAULT_CERT_EXPIRY_WARNING_SECONDS",
    "DEFAULT_TLS_HANDSHAKE_TIMEOUT_SECONDS",
    # Errors
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
    # Policy
    "SecureTransportPolicy",
    # Certificates
    "compute_certificate_fingerprint",
    "parse_certificate_from_pem",
    "parse_certificate_from_der",
    "extract_certificate_metadata",
    "inspect_certificate_lifecycle",
    "CertificateRevocationRegistry",
    "HermeticPKIBuilder",
    # Validation
    "validate_certificate",
    "validate_certificate_or_raise",
    # TLS
    "TLSContextFactory",
    # Binding
    "PeerCertificateBinder",
]
