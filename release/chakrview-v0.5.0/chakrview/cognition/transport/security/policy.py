"""
Secure Transport Policy (Step 31).

Defines configuration and safety constraints for TLS and mTLS transports.
Enforces strict fail-closed validation, TLS 1.3 preference, mandatory certificate
verification, and explicit containment of test-only plaintext modes.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set

from chakrview.cognition.transport.security.models import (
    TLSMode,
    TLSProtocolVersion,
    DEFAULT_CERT_EXPIRY_WARNING_SECONDS,
    DEFAULT_TLS_HANDSHAKE_TIMEOUT_SECONDS,
)
from chakrview.cognition.transport.security.errors import (
    TLSConfigurationError,
    InsecureDowngradeError,
)


@dataclass
class SecureTransportPolicy:
    """
    Immutable or validated policy governing TLS/mTLS parameters.
    Hard safety rules prevent remote peers or configuration errors from downgrading
    security in production.
    """
    tls_mode: TLSMode = TLSMode.TLS
    minimum_tls_version: TLSProtocolVersion = TLSProtocolVersion.TLS_1_3
    allow_tls_1_2: bool = False
    verify_peer_certificate: bool = True
    verify_hostname: bool = True
    trusted_ca_paths: List[str] = field(default_factory=list)
    trusted_ca_data: Optional[str] = None  # PEM-encoded CA bundle
    allowed_peer_fingerprints: Set[str] = field(default_factory=set)
    certificate_expiry_warning_window_seconds: float = DEFAULT_CERT_EXPIRY_WARNING_SECONDS
    maximum_handshake_timeout_seconds: float = DEFAULT_TLS_HANDSHAKE_TIMEOUT_SECONDS
    require_client_certificate: bool = False
    allow_insecure_test_downgrade: bool = False

    def __post_init__(self) -> None:
        if self.tls_mode == TLSMode.MTLS:
            self.require_client_certificate = True
        self.validate()

    def validate(self) -> None:
        """
        Validate policy against hard architectural security invariants.
        Fails closed if insecure configurations are detected.
        """
        # Invariant: Plaintext mode requires explicit test-only flag
        if self.tls_mode == TLSMode.PLAINTEXT_TEST_ONLY:
            if not self.allow_insecure_test_downgrade:
                raise InsecureDowngradeError(
                    "Plaintext mode is forbidden in production transport policy. "
                    "Must set allow_insecure_test_downgrade=True explicitly in test configurations."
                )

        # Invariant: Disabling certificate verification requires explicit test-only flag
        if not self.verify_peer_certificate:
            if not self.allow_insecure_test_downgrade:
                raise InsecureDowngradeError(
                    "Disabling peer certificate verification is strictly forbidden in production policy."
                )

        # Invariant: TLS 1.2 requires explicit permission flag
        if self.minimum_tls_version == TLSProtocolVersion.TLS_1_2 and not self.allow_tls_1_2:
            raise TLSConfigurationError(
                "TLS 1.2 is only permitted when allow_tls_1_2 is explicitly enabled in policy."
            )

        # Invariant: Handshake timeout must be positive and bounded
        if self.maximum_handshake_timeout_seconds <= 0 or self.maximum_handshake_timeout_seconds > 60.0:
            raise TLSConfigurationError(
                f"Handshake timeout must be between 0.1s and 60.0s, got {self.maximum_handshake_timeout_seconds}s."
            )

    def to_dict(self) -> Dict[str, Any]:
        """Safe dictionary representation with zero secrets."""
        return {
            "tls_mode": self.tls_mode.value,
            "minimum_tls_version": self.minimum_tls_version.value,
            "allow_tls_1_2": self.allow_tls_1_2,
            "verify_peer_certificate": self.verify_peer_certificate,
            "verify_hostname": self.verify_hostname,
            "has_trusted_ca_data": self.trusted_ca_data is not None,
            "trusted_ca_paths_count": len(self.trusted_ca_paths),
            "allowed_peer_fingerprints_count": len(self.allowed_peer_fingerprints),
            "certificate_expiry_warning_window_seconds": self.certificate_expiry_warning_window_seconds,
            "maximum_handshake_timeout_seconds": self.maximum_handshake_timeout_seconds,
            "require_client_certificate": self.require_client_certificate,
            "allow_insecure_test_downgrade": self.allow_insecure_test_downgrade,
        }
