"""
Transport Security Data Models and Contracts (Step 31).

Defines strongly typed primitives for TLS modes, protocol versions, certificate
metadata, certificate lifecycle states, and peer-certificate bindings.

CRITICAL INVARIANTS:
1. No private key material is ever stored or serialized in these data models.
2. Certificate metadata representation is strictly sanitized for logging and telemetry.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Set
import time


# ============================================================================
# Protocol & Mode Enumerations
# ============================================================================

class TLSMode(str, Enum):
    """Transport security execution modes."""
    PLAINTEXT_TEST_ONLY = "PLAINTEXT_TEST_ONLY"
    TLS = "TLS"
    MTLS = "MTLS"


class TLSProtocolVersion(str, Enum):
    """Allowed TLS protocol versions."""
    TLS_1_2 = "TLSv1.2"
    TLS_1_3 = "TLSv1.3"


class CertificateLifecycleState(str, Enum):
    """Deterministic certificate lifecycle states."""
    VALID = "VALID"
    EXPIRING = "EXPIRING"
    EXPIRED = "EXPIRED"
    REVOKED = "REVOKED"
    UNKNOWN = "UNKNOWN"
    INVALID = "INVALID"


class CertificateUsage(str, Enum):
    """Allowed certificate operational usages."""
    SERVER_AUTH = "SERVER_AUTH"
    CLIENT_AUTH = "CLIENT_AUTH"
    MUTUAL_AUTH = "MUTUAL_AUTH"


# ============================================================================
# Constants
# ============================================================================

DEFAULT_CERT_EXPIRY_WARNING_SECONDS = 86400.0 * 14.0  # 14 days
DEFAULT_TLS_HANDSHAKE_TIMEOUT_SECONDS = 5.0
DEFAULT_MAX_TLS_FRAME_BYTES = 1024 * 1024  # 1 MB


# ============================================================================
# Certificate Metadata Model
# ============================================================================

@dataclass(frozen=True)
class CertificateMetadata:
    """
    Sanitized metadata describing an X.509 certificate.
    Contains zero private key or secret material.
    """
    fingerprint: str  # Format: "SHA256:xx:xx:..."
    subject: Dict[str, str]
    issuer: Dict[str, str]
    serial_number: str
    not_before_epoch: float
    not_after_epoch: float
    san_dns_names: List[str] = field(default_factory=list)
    san_ip_addresses: List[str] = field(default_factory=list)
    key_usage: List[str] = field(default_factory=list)
    extended_key_usage: List[str] = field(default_factory=list)
    lifecycle_state: CertificateLifecycleState = CertificateLifecycleState.VALID
    is_ca: bool = False

    def is_valid_at(self, epoch_time: Optional[float] = None) -> bool:
        """Check if certificate is within validity window."""
        t = epoch_time if epoch_time is not None else time.time()
        return self.not_before_epoch <= t <= self.not_after_epoch

    def matches_hostname(self, hostname: str) -> bool:
        """Check if certificate matches expected hostname or IP."""
        if not hostname:
            return False
        h_lower = hostname.lower()
        if h_lower in [d.lower() for d in self.san_dns_names]:
            return True
        if h_lower in [ip.lower() for ip in self.san_ip_addresses]:
            return True
        # Check common name fallback if SAN is empty
        cn = self.subject.get("CN", "").lower()
        if cn and cn == h_lower:
            return True
        return False

    def to_dict(self) -> Dict[str, Any]:
        """Safe dictionary representation with zero secrets."""
        return {
            "fingerprint": self.fingerprint,
            "subject": self.subject,
            "issuer": self.issuer,
            "serial_number": self.serial_number,
            "not_before_epoch": self.not_before_epoch,
            "not_after_epoch": self.not_after_epoch,
            "san_dns_names": list(self.san_dns_names),
            "san_ip_addresses": list(self.san_ip_addresses),
            "key_usage": list(self.key_usage),
            "extended_key_usage": list(self.extended_key_usage),
            "lifecycle_state": self.lifecycle_state.value,
            "is_ca": self.is_ca,
        }

    def sanitized_repr(self) -> str:
        """Compact summary string for telemetry and logging."""
        cn = self.subject.get("CN", "UNKNOWN")
        return f"Cert(CN={cn}, FP={self.fingerprint[:24]}..., State={self.lifecycle_state.value})"


# ============================================================================
# Peer Certificate Binding Model
# ============================================================================

@dataclass(frozen=True)
class PeerCertificateBinding:
    """
    Explicit binding between a ChakrView CryptographicPeerIdentity and a TLS certificate.
    Prevents certificate substitution or unauthorized peer impersonation.
    """
    peer_id: str
    certificate_fingerprint: str
    expected_common_name: Optional[str] = None
    expected_san: Optional[str] = None
    binding_epoch: int = 1
    is_active: bool = True

    def matches_certificate(self, cert_meta: CertificateMetadata) -> bool:
        """Verify that a certificate matches this binding policy."""
        if not self.is_active:
            return False
        if self.certificate_fingerprint != cert_meta.fingerprint:
            return False
        if self.expected_common_name is not None:
            cn = cert_meta.subject.get("CN")
            if cn != self.expected_common_name:
                return False
        if self.expected_san is not None:
            if not cert_meta.matches_hostname(self.expected_san):
                return False
        return True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "peer_id": self.peer_id,
            "certificate_fingerprint": self.certificate_fingerprint,
            "expected_common_name": self.expected_common_name,
            "expected_san": self.expected_san,
            "binding_epoch": self.binding_epoch,
            "is_active": self.is_active,
        }
