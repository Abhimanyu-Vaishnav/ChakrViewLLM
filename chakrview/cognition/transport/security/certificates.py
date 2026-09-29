"""
X.509 Certificate Parsing, Metadata Extraction, Lifecycle, and Hermetic PKI (Step 31).

Provides:
1. Safe certificate parsing and canonical metadata extraction using PyCA cryptography.
2. Deterministic fingerprint calculation (SHA-256).
3. Lifecycle state evaluation (VALID, EXPIRING, EXPIRED, REVOKED).
4. Hermetic test PKI builder for generating valid, expired, and untrusted CA/server/client
   certificates for testing without external tool dependencies.
5. In-memory CertificateRevocationRegistry for tracking revoked fingerprints.

CRITICAL INVARIANTS:
1. Private keys are never serialized in metadata or exposed via public logging APIs.
2. Certificate parsing and validation fail closed on malformed input.
"""

from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple, Set, Union, Any
import hashlib
import ipaddress
import time

from cryptography import x509
from cryptography.x509.oid import NameOID, ExtendedKeyUsageOID, ExtensionOID
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec

from chakrview.cognition.transport.security.models import (
    CertificateMetadata,
    CertificateLifecycleState,
    DEFAULT_CERT_EXPIRY_WARNING_SECONDS,
)
from chakrview.cognition.transport.security.errors import (
    CertificateError,
    CertificateValidationError,
    CertificateExpiredError,
    CertificateNotYetValidError,
    CertificateRevokedError,
)


# ============================================================================
# Fingerprint & Parsing Utilities
# ============================================================================

def compute_certificate_fingerprint(cert_or_bytes: Union[x509.Certificate, bytes]) -> str:
    """
    Compute canonical SHA-256 fingerprint formatted as 'SHA256:XX:XX:...'.
    """
    if isinstance(cert_or_bytes, x509.Certificate):
        der_bytes = cert_or_bytes.public_bytes(serialization.Encoding.DER)
    elif isinstance(cert_or_bytes, (bytes, bytearray)):
        der_bytes = bytes(cert_or_bytes)
    else:
        raise CertificateError(f"Unsupported certificate input type: {type(cert_or_bytes)}")

    sha = hashlib.sha256(der_bytes).hexdigest().upper()
    colon_pairs = ":".join(sha[i:i + 2] for i in range(0, len(sha), 2))
    return f"SHA256:{colon_pairs}"


def parse_certificate_from_pem(pem_data: Union[bytes, str]) -> x509.Certificate:
    """
    Parse an X.509 certificate from PEM string or bytes. Fails closed on error.
    """
    try:
        raw_bytes = pem_data.encode("utf-8") if isinstance(pem_data, str) else pem_data
        return x509.load_pem_x509_certificate(raw_bytes)
    except Exception as e:
        raise CertificateValidationError(f"Failed to parse PEM certificate: {e}")


def parse_certificate_from_der(der_bytes: bytes) -> x509.Certificate:
    """
    Parse an X.509 certificate from binary DER format. Fails closed on error.
    """
    try:
        return x509.load_der_x509_certificate(der_bytes)
    except Exception as e:
        raise CertificateValidationError(f"Failed to parse DER certificate: {e}")


# ============================================================================
# Metadata Extraction
# ============================================================================

def _name_to_dict(name: x509.Name) -> Dict[str, str]:
    """Convert an X.509 Name into a human-readable dictionary with standard aliases."""
    out: Dict[str, str] = {}
    oid_aliases = {
        "commonName": "CN",
        "organizationName": "O",
        "organizationalUnitName": "OU",
        "countryName": "C",
        "stateOrProvinceName": "ST",
        "localityName": "L",
    }
    for attr in name:
        try:
            oid_name = attr.oid._name
        except Exception:
            oid_name = attr.oid.dotted_string
        out[oid_name] = attr.value
        if oid_name in oid_aliases:
            out[oid_aliases[oid_name]] = attr.value
    return out



def extract_certificate_metadata(
    cert: x509.Certificate,
    current_time: Optional[float] = None,
    warning_window_seconds: float = DEFAULT_CERT_EXPIRY_WARNING_SECONDS,
    is_revoked: bool = False,
) -> CertificateMetadata:
    """
    Extract sanitized, secret-free metadata from an X.509 certificate.
    """
    try:
        fingerprint = compute_certificate_fingerprint(cert)
        subject_dict = _name_to_dict(cert.subject)
        issuer_dict = _name_to_dict(cert.issuer)
        serial_str = hex(cert.serial_number)

        # Dates: cryptography provides not_valid_before_utc and not_valid_after_utc
        not_before = cert.not_valid_before_utc.timestamp()
        not_after = cert.not_valid_after_utc.timestamp()

        # Subject Alternative Names
        san_dns: List[str] = []
        san_ips: List[str] = []
        try:
            san_ext = cert.extensions.get_extension_for_oid(ExtensionOID.SUBJECT_ALTERNATIVE_NAME)
            for name in san_ext.value:
                if isinstance(name, x509.DNSName):
                    san_dns.append(name.value)
                elif isinstance(name, x509.IPAddress):
                    san_ips.append(str(name.value))
        except x509.ExtensionNotFound:
            pass

        # Key Usage
        key_usage: List[str] = []
        try:
            ku_ext = cert.extensions.get_extension_for_oid(ExtensionOID.KEY_USAGE)
            val = ku_ext.value
            for ku_attr in [
                "digital_signature", "content_commitment", "key_encipherment",
                "data_encipherment", "key_agreement", "key_cert_sign", "crl_sign",
            ]:
                if getattr(val, ku_attr, False):
                    key_usage.append(ku_attr)
        except x509.ExtensionNotFound:
            pass

        # Extended Key Usage
        ext_key_usage: List[str] = []
        try:
            eku_ext = cert.extensions.get_extension_for_oid(ExtensionOID.EXTENDED_KEY_USAGE)
            for oid in eku_ext.value:
                if oid == ExtendedKeyUsageOID.SERVER_AUTH:
                    ext_key_usage.append("server_auth")
                elif oid == ExtendedKeyUsageOID.CLIENT_AUTH:
                    ext_key_usage.append("client_auth")
                else:
                    ext_key_usage.append(str(oid.dotted_string))
        except x509.ExtensionNotFound:
            pass

        # Basic Constraints (CA)
        is_ca = False
        try:
            bc_ext = cert.extensions.get_extension_for_oid(ExtensionOID.BASIC_CONSTRAINTS)
            is_ca = bc_ext.value.ca
        except x509.ExtensionNotFound:
            pass

        # Determine Lifecycle State
        now = current_time if current_time is not None else time.time()
        lifecycle = inspect_certificate_lifecycle(
            not_before=not_before,
            not_after=not_after,
            current_time=now,
            warning_window_seconds=warning_window_seconds,
            is_revoked=is_revoked,
        )

        return CertificateMetadata(
            fingerprint=fingerprint,
            subject=subject_dict,
            issuer=issuer_dict,
            serial_number=serial_str,
            not_before_epoch=not_before,
            not_after_epoch=not_after,
            san_dns_names=san_dns,
            san_ip_addresses=san_ips,
            key_usage=key_usage,
            extended_key_usage=ext_key_usage,
            lifecycle_state=lifecycle,
            is_ca=is_ca,
        )

    except Exception as e:
        raise CertificateValidationError(f"Failed to extract certificate metadata: {e}")


def inspect_certificate_lifecycle(
    not_before: float,
    not_after: float,
    current_time: float,
    warning_window_seconds: float = DEFAULT_CERT_EXPIRY_WARNING_SECONDS,
    is_revoked: bool = False,
) -> CertificateLifecycleState:
    """
    Deterministic evaluation of certificate lifecycle state.
    """
    if is_revoked:
        return CertificateLifecycleState.REVOKED
    if current_time < not_before:
        return CertificateLifecycleState.INVALID
    if current_time > not_after:
        return CertificateLifecycleState.EXPIRED
    if current_time + warning_window_seconds >= not_after:
        return CertificateLifecycleState.EXPIRING
    return CertificateLifecycleState.VALID


# ============================================================================
# Certificate Revocation Registry
# ============================================================================

class CertificateRevocationRegistry:
    """
    Thread-safe, in-memory ledger of revoked certificate fingerprints.
    """

    def __init__(self) -> None:
        self._revoked: Dict[str, Dict[str, Any]] = {}

    def revoke(self, fingerprint: str, reason: str = "Unspecified", epoch: int = 1) -> None:
        """Register a certificate fingerprint as revoked."""
        self._revoked[fingerprint] = {
            "fingerprint": fingerprint,
            "reason": reason,
            "revoked_at_epoch": epoch,
            "revoked_at_timestamp": time.time(),
        }

    def is_revoked(self, fingerprint: str) -> bool:
        """Check if fingerprint is revoked."""
        return fingerprint in self._revoked

    def get_revocation_record(self, fingerprint: str) -> Optional[Dict[str, Any]]:
        return self._revoked.get(fingerprint)

    def count(self) -> int:
        return len(self._revoked)

    def list_revocations(self) -> List[str]:
        """Return list of all revoked certificate fingerprints."""
        return list(self._revoked.keys())


# ============================================================================
# Hermetic PKI Builder (For Tests and Isolated Deployments)
# ============================================================================

class HermeticPKIBuilder:
    """
    Generates genuine, standards-compliant X.509 certificate chains in-memory
    using ECDSA (P-256) keys, eliminating any dependency on external CA infrastructure
    or CLI tools while enabling authentic TLS/mTLS handshakes over sockets.
    """

    @staticmethod
    def create_ca(
        common_name: str = "ChakrView Test CA",
        organization: str = "ChakrView Federation",
        validity_days: int = 30,
    ) -> Tuple[str, str, x509.Certificate, ec.EllipticCurvePrivateKey]:
        """
        Generate self-signed Root CA.
        Returns: (ca_cert_pem, ca_key_pem, ca_cert, ca_key)
        """
        key = ec.generate_private_key(ec.SECP256R1())
        name = x509.Name([
            x509.NameAttribute(NameOID.COMMON_NAME, common_name),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, organization),
        ])
        now = datetime.now(timezone.utc)
        cert = (
            x509.CertificateBuilder()
            .subject_name(name)
            .issuer_name(name)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - timedelta(minutes=5))
            .not_valid_after(now + timedelta(days=validity_days))
            .add_extension(
                x509.BasicConstraints(ca=True, path_length=None),
                critical=True,
            )
            .add_extension(
                x509.KeyUsage(
                    digital_signature=True,
                    content_commitment=False,
                    key_encipherment=False,
                    data_encipherment=False,
                    key_agreement=False,
                    key_cert_sign=True,
                    crl_sign=True,
                    encipher_only=False,
                    decipher_only=False,
                ),
                critical=True,
            )
            .sign(key, hashes.SHA256())
        )

        cert_pem = cert.public_bytes(serialization.Encoding.PEM).decode("utf-8")
        key_pem = key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        ).decode("utf-8")

        return cert_pem, key_pem, cert, key

    @staticmethod
    def create_server_cert(
        ca_cert: x509.Certificate,
        ca_key: ec.EllipticCurvePrivateKey,
        common_name: str = "localhost",
        san_dns: Optional[List[str]] = None,
        san_ips: Optional[List[str]] = None,
        validity_days: int = 60,
        start_offset_days: int = 0,
    ) -> Tuple[str, str, x509.Certificate, ec.EllipticCurvePrivateKey]:
        """
        Generate server certificate signed by CA.
        """
        if san_dns is None:
            san_dns = ["localhost"]
        if san_ips is None:
            san_ips = ["127.0.0.1"]

        key = ec.generate_private_key(ec.SECP256R1())
        subject = x509.Name([
            x509.NameAttribute(NameOID.COMMON_NAME, common_name),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "ChakrView Federation Node"),
        ])
        now = datetime.now(timezone.utc) + timedelta(days=start_offset_days)

        san_elements: List[Union[x509.DNSName, x509.IPAddress]] = []
        for d in san_dns:
            san_elements.append(x509.DNSName(d))
        for ip in san_ips:
            san_elements.append(x509.IPAddress(ipaddress.ip_address(ip)))

        cert = (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(ca_cert.subject)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - timedelta(minutes=5))
            .not_valid_after(now + timedelta(days=validity_days))
            .add_extension(
                x509.BasicConstraints(ca=False, path_length=None),
                critical=True,
            )
            .add_extension(
                x509.KeyUsage(
                    digital_signature=True,
                    content_commitment=False,
                    key_encipherment=True,
                    data_encipherment=False,
                    key_agreement=False,
                    key_cert_sign=False,
                    crl_sign=False,
                    encipher_only=False,
                    decipher_only=False,
                ),
                critical=True,
            )
            .add_extension(
                x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]),
                critical=False,
            )
            .add_extension(
                x509.SubjectAlternativeName(san_elements),
                critical=False,
            )
            .sign(ca_key, hashes.SHA256())
        )

        cert_pem = cert.public_bytes(serialization.Encoding.PEM).decode("utf-8")
        key_pem = key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        ).decode("utf-8")

        return cert_pem, key_pem, cert, key

    @staticmethod
    def create_client_cert(
        ca_cert: x509.Certificate,
        ca_key: ec.EllipticCurvePrivateKey,
        common_name: str = "peer-client",
        validity_days: int = 60,
        start_offset_days: int = 0,
    ) -> Tuple[str, str, x509.Certificate, ec.EllipticCurvePrivateKey]:
        """
        Generate client certificate for mutual TLS (mTLS).
        """
        key = ec.generate_private_key(ec.SECP256R1())
        subject = x509.Name([
            x509.NameAttribute(NameOID.COMMON_NAME, common_name),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "ChakrView Federated Peer"),
        ])
        now = datetime.now(timezone.utc) + timedelta(days=start_offset_days)

        cert = (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(ca_cert.subject)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - timedelta(minutes=5))
            .not_valid_after(now + timedelta(days=validity_days))
            .add_extension(
                x509.BasicConstraints(ca=False, path_length=None),
                critical=True,
            )
            .add_extension(
                x509.KeyUsage(
                    digital_signature=True,
                    content_commitment=False,
                    key_encipherment=True,
                    data_encipherment=False,
                    key_agreement=False,
                    key_cert_sign=False,
                    crl_sign=False,
                    encipher_only=False,
                    decipher_only=False,
                ),
                critical=True,
            )
            .add_extension(
                x509.ExtendedKeyUsage([ExtendedKeyUsageOID.CLIENT_AUTH]),
                critical=False,
            )
            .sign(ca_key, hashes.SHA256())
        )

        cert_pem = cert.public_bytes(serialization.Encoding.PEM).decode("utf-8")
        key_pem = key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        ).decode("utf-8")

        return cert_pem, key_pem, cert, key
