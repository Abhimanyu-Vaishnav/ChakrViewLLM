"""
Deterministic Certificate Validation Functions (Step 31).

Enforces strict, fail-closed validation of parsed certificate metadata:
1. Temporal validity (not-before and not-after checks)
2. Revocation status verification
3. Fingerprint pinning policy enforcement
4. Hostname and Subject Alternative Name (SAN) verification
5. KeyUsage and ExtendedKeyUsage compliance
"""

from typing import Dict, List, Optional, Set, Tuple
import time

from chakrview.cognition.transport.security.models import (
    CertificateMetadata,
    CertificateLifecycleState,
)
from chakrview.cognition.transport.security.certificates import CertificateRevocationRegistry
from chakrview.cognition.transport.security.errors import (
    CertificateValidationError,
    CertificateExpiredError,
    CertificateNotYetValidError,
    CertificateRevokedError,
    HostnameMismatchError,
)


def validate_certificate(
    metadata: CertificateMetadata,
    expected_hostname: Optional[str] = None,
    allowed_fingerprints: Optional[Set[str]] = None,
    revocation_registry: Optional[CertificateRevocationRegistry] = None,
    current_time: Optional[float] = None,
    required_key_usages: Optional[List[str]] = None,
    required_ext_key_usages: Optional[List[str]] = None,
) -> Tuple[bool, Optional[str]]:
    """
    Validate certificate metadata against explicit policy requirements.
    Returns (is_valid, failure_reason).
    """
    now = current_time if current_time is not None else time.time()

    # 1. Revocation Check
    if revocation_registry and revocation_registry.is_revoked(metadata.fingerprint):
        rec = revocation_registry.get_revocation_record(metadata.fingerprint)
        reason = rec.get("reason", "Revoked") if rec else "Revoked"
        return False, f"Certificate has been revoked: {reason} (FP: {metadata.fingerprint})"

    # 2. Date Bounds Checks
    if now < metadata.not_before_epoch:
        return False, f"Certificate is not yet valid (valid from epoch {metadata.not_before_epoch}, current {now})"

    if now > metadata.not_after_epoch:
        return False, f"Certificate expired at epoch {metadata.not_after_epoch} (current {now})"

    # 3. Fingerprint Pinning Policy Check
    if allowed_fingerprints and len(allowed_fingerprints) > 0:
        if metadata.fingerprint not in allowed_fingerprints:
            return False, f"Certificate fingerprint '{metadata.fingerprint}' is not in allowed pinning set"

    # 4. Hostname / SAN Verification
    if expected_hostname:
        if not metadata.matches_hostname(expected_hostname):
            return False, (
                f"Certificate hostname mismatch: expected '{expected_hostname}', "
                f"found SAN DNS={metadata.san_dns_names}, SAN IP={metadata.san_ip_addresses}, "
                f"CN={metadata.subject.get('CN')}"
            )

    # 5. Key Usage Verification
    if required_key_usages:
        for ku in required_key_usages:
            if ku not in metadata.key_usage:
                return False, f"Certificate missing required key usage: '{ku}'"

    # 6. Extended Key Usage Verification
    if required_ext_key_usages:
        for eku in required_ext_key_usages:
            if eku not in metadata.extended_key_usage:
                return False, f"Certificate missing required extended key usage: '{eku}'"

    return True, None


def validate_certificate_or_raise(
    metadata: CertificateMetadata,
    expected_hostname: Optional[str] = None,
    allowed_fingerprints: Optional[Set[str]] = None,
    revocation_registry: Optional[CertificateRevocationRegistry] = None,
    current_time: Optional[float] = None,
    required_key_usages: Optional[List[str]] = None,
    required_ext_key_usages: Optional[List[str]] = None,
) -> None:
    """
    Validate certificate and raise a strongly typed exception on failure.
    """
    now = current_time if current_time is not None else time.time()

    # 1. Revocation
    if revocation_registry and revocation_registry.is_revoked(metadata.fingerprint):
        rec = revocation_registry.get_revocation_record(metadata.fingerprint)
        reason = rec.get("reason", "Revoked") if rec else "Revoked"
        raise CertificateRevokedError(f"Certificate revoked: {reason} (FP: {metadata.fingerprint})")

    # 2. Date Bounds
    if now < metadata.not_before_epoch:
        raise CertificateNotYetValidError(
            f"Certificate not yet valid (valid from {metadata.not_before_epoch}, now {now})"
        )
    if now > metadata.not_after_epoch:
        raise CertificateExpiredError(
            f"Certificate expired at epoch {metadata.not_after_epoch} (current {now})"
        )

    # 3. Fingerprint Pinning
    if allowed_fingerprints and len(allowed_fingerprints) > 0:
        if metadata.fingerprint not in allowed_fingerprints:
            raise CertificateValidationError(
                f"Certificate fingerprint '{metadata.fingerprint}' not in allowed pinning set"
            )

    # 4. Hostname
    if expected_hostname:
        if not metadata.matches_hostname(expected_hostname):
            raise HostnameMismatchError(
                f"Hostname mismatch: expected '{expected_hostname}', "
                f"SAN DNS={metadata.san_dns_names}, SAN IP={metadata.san_ip_addresses}"
            )

    # 5. Key Usages
    if required_key_usages:
        for ku in required_key_usages:
            if ku not in metadata.key_usage:
                raise CertificateValidationError(f"Missing required key usage '{ku}'")

    # 6. Extended Key Usages
    if required_ext_key_usages:
        for eku in required_ext_key_usages:
            if eku not in metadata.extended_key_usage:
                raise CertificateValidationError(f"Missing required extended key usage '{eku}'")
