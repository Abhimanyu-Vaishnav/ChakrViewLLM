"""
Federation Engine Identity Provider and Verification (Step 33).

CRITICAL ARCHITECTURAL AXIOMS:
1. ENGINE_IDENTITY != PEER_IDENTITY:
   Engine identity identifies a federation coordinator node; it does not replace
   or conflate with peer identity.
2. ENGINE_IDENTITY != AUTHORITY:
   A remote engine having a valid identity confers ZERO authority over local capabilities,
   model weights, private sessions, or tenant data.
3. IMMUTABILITY & DETERMINISM:
   Engine identities are deterministic, cryptographically fingerprinted, and frozen.
"""

from typing import Optional, Tuple
import re

from chakrview.cognition.federation.models import (
    FederationEngineIdentity,
    FEDERATION_PROTOCOL_VERSION,
)
from chakrview.cognition.federation.errors import EngineIdentityError


class FederationEngineIdentityProvider:
    """
    Factory and deterministic validator for FederationEngineIdentity instances.
    """

    # Identifiers must be alphanumeric with underscores/hyphens
    ID_PATTERN = re.compile(r"^[a-zA-Z0-9_\-\.]{3,64}$")

    @classmethod
    def create_identity(
        cls,
        engine_id: str,
        zone_id: str,
        created_epoch: int = 1,
        protocol_version: str = FEDERATION_PROTOCOL_VERSION,
        architecture_version: str = "0.1",
        public_key_fingerprint: Optional[str] = None,
    ) -> FederationEngineIdentity:
        """
        Derive an immutable, fingerprinted FederationEngineIdentity.
        """
        if not engine_id or not cls.ID_PATTERN.match(engine_id):
            raise EngineIdentityError(
                f"Invalid engine_id '{engine_id}': must match pattern '{cls.ID_PATTERN.pattern}'."
            )
        if not zone_id or not cls.ID_PATTERN.match(zone_id):
            raise EngineIdentityError(
                f"Invalid zone_id '{zone_id}': must match pattern '{cls.ID_PATTERN.pattern}'."
            )
        if created_epoch < 1:
            raise EngineIdentityError(f"Created epoch must be >= 1, got {created_epoch}.")

        temp_id = FederationEngineIdentity(
            engine_id=engine_id,
            zone_id=zone_id,
            created_epoch=created_epoch,
            protocol_version=protocol_version,
            architecture_version=architecture_version,
            public_key_fingerprint=public_key_fingerprint,
        )
        fingerprint = temp_id.compute_fingerprint()

        return FederationEngineIdentity(
            engine_id=engine_id,
            zone_id=zone_id,
            created_epoch=created_epoch,
            identity_fingerprint=fingerprint,
            protocol_version=protocol_version,
            architecture_version=architecture_version,
            public_key_fingerprint=public_key_fingerprint,
        )

    @classmethod
    def validate_identity(cls, identity: FederationEngineIdentity) -> Tuple[bool, str]:
        """
        Validate structural integrity and cryptographic fingerprint of an engine identity.
        Returns (is_valid, reason).
        """
        if not isinstance(identity, FederationEngineIdentity):
            return False, f"Expected FederationEngineIdentity, got {type(identity)}"

        if not identity.engine_id or not cls.ID_PATTERN.match(identity.engine_id):
            return False, f"Invalid engine_id: '{identity.engine_id}'"

        if not identity.zone_id or not cls.ID_PATTERN.match(identity.zone_id):
            return False, f"Invalid zone_id: '{identity.zone_id}'"

        if identity.created_epoch < 1:
            return False, f"Invalid created_epoch: {identity.created_epoch}"

        expected_fingerprint = identity.compute_fingerprint()
        if identity.identity_fingerprint != expected_fingerprint:
            return False, (
                f"Fingerprint mismatch: expected '{expected_fingerprint}', "
                f"got '{identity.identity_fingerprint}'"
            )

        return True, "Valid engine identity"
