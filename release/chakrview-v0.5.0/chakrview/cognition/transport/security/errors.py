"""
Transport Security Exception Hierarchy (Step 31).

All transport security, TLS, certificate, and binding failures fail closed.
Never silently downgrade or continue execution after a security failure.
"""

from chakrview.cognition.transport.errors import TransportError


class TransportSecurityError(TransportError):
    """Base class for all transport security and TLS failures."""
    pass


class TLSError(TransportSecurityError):
    """Base exception for TLS protocol, context, and handshake failures."""
    pass


class TLSConfigurationError(TLSError):
    """Raised when TLS configuration or policy is invalid or insecure."""
    pass


class TLSHandshakeError(TLSError):
    """Raised when a TLS/mTLS handshake fails or times out."""
    pass


class InsecureDowngradeError(TLSError):
    """Raised when an insecure downgrade (e.g., plaintext or CERT_NONE) is attempted."""
    pass


class CertificateError(TransportSecurityError):
    """Base exception for certificate parsing, verification, or lifecycle failures."""
    pass


class CertificateValidationError(CertificateError):
    """Raised when a certificate fails validation (chain, structure, or policy)."""
    pass


class CertificateExpiredError(CertificateValidationError):
    """Raised when a certificate is expired."""
    pass


class CertificateNotYetValidError(CertificateValidationError):
    """Raised when a certificate's validity start date is in the future."""
    pass


class CertificateRevokedError(CertificateValidationError):
    """Raised when a certificate has been revoked."""
    pass


class HostnameMismatchError(CertificateValidationError):
    """Raised when a certificate Subject Alternative Name / CN does not match expected hostname."""
    pass


class UntrustedCAError(CertificateValidationError):
    """Raised when a certificate issuer is not signed by a trusted CA."""
    pass


class ClientCertificateMissingError(TLSError):
    """Raised during mTLS when client does not present a mandatory certificate."""
    pass


class PeerBindingMismatchError(TransportSecurityError):
    """Raised when a TLS certificate identity does not match the bound ChakrView peer identity."""
    pass
