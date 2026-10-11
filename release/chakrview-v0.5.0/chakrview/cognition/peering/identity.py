"""
Deterministic Peer Identity Provider for Cross-Zone Federation (Step 29).

CRITICAL ARCHITECTURAL AXIOM:
IDENTITY != CRYPTOGRAPHIC_AUTHENTICATION:
Step 29 implements deterministic local identity representation and fingerprinting.
Production cryptographic authentication (Ed25519, HSM, X.509 PKI) is explicitly
deferred to future steps. Claiming cryptographic authentication raises UnsupportedSecurityModeError.
"""

import re
from typing import Dict, List, Optional, Any

from chakrview.cognition.peering.models import PeerIdentity


class IdentityValidationError(ValueError):
    """Raised when peer identity fields violate architectural format or safety rules."""
    pass


class UnsupportedSecurityModeError(NotImplementedError):
    """Raised when an unsupported production cryptographic mode is requested."""
    pass


class PeerIdentityProvider:
    """
    Deterministic identity generator and validator for federated peers.
    """

    ID_PATTERN = re.compile(r"^[a-zA-Z0-9_\-\.]{2,64}$")

    @classmethod
    def validate_identifier(cls, name: str, value: str) -> None:
        """Enforce strict naming and anti-injection rules on identifier strings."""
        if not value or not isinstance(value, str):
            raise IdentityValidationError(f"{name} must be a non-empty string.")
        if not cls.ID_PATTERN.match(value):
            raise IdentityValidationError(
                f"{name} '{value}' contains invalid characters. Must match {cls.ID_PATTERN.pattern}."
            )
        if ".." in value or "/" in value or "\\" in value:
            raise IdentityValidationError(f"{name} '{value}' contains forbidden path traversal sequences.")

    @classmethod
    def create_identity(
        cls,
        peer_id: str,
        zone_id: str,
        organization_id: str,
        capability_profile: Optional[Dict[str, Any]] = None,
        supported_features: Optional[List[str]] = None,
        protocol_version: str = "29.0",
        architecture_version: str = "0.1",
        created_epoch: int = 1,
    ) -> PeerIdentity:
        """
        Create a validated, deterministically fingerprinted PeerIdentity.
        """
        cls.validate_identifier("peer_id", peer_id)
        cls.validate_identifier("zone_id", zone_id)
        cls.validate_identifier("organization_id", organization_id)

        features = list(supported_features or [])
        profile = dict(capability_profile or {})

        # Construct un-fingerprinted identity to compute canonical digest
        temp_identity = PeerIdentity(
            peer_id=peer_id,
            zone_id=zone_id,
            organization_id=organization_id,
            protocol_version=protocol_version,
            architecture_version=architecture_version,
            capability_profile=profile,
            supported_features=features,
            created_epoch=created_epoch,
            fingerprint="",
        )

        fingerprint = temp_identity.compute_fingerprint()

        return PeerIdentity(
            peer_id=peer_id,
            zone_id=zone_id,
            organization_id=organization_id,
            protocol_version=protocol_version,
            architecture_version=architecture_version,
            capability_profile=profile,
            supported_features=features,
            created_epoch=created_epoch,
            fingerprint=fingerprint,
        )

    @classmethod
    def verify_fingerprint(cls, identity: PeerIdentity) -> bool:
        """Verify that identity fingerprint matches canonical SHA-256 digest."""
        if not identity.fingerprint:
            return False
        expected = identity.compute_fingerprint()
        return identity.fingerprint == expected

    @classmethod
    def authenticate_cryptographically(cls, identity: PeerIdentity, signature: Any) -> bool:
        """
        Explicitly fail closed: Production PKI / Ed25519 authentication is deferred.
        """
        raise UnsupportedSecurityModeError(
            "Production cryptographic authentication (PKI/Ed25519/HSM) is explicitly deferred. "
            "Step 29 enforces IDENTITY != CRYPTOGRAPHIC_AUTHENTICATION."
        )
