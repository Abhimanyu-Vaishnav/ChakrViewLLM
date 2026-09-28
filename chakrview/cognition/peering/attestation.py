"""
Deterministic Peer Attestation and Claim Verification for Cross-Zone Federation (Step 29).

CRITICAL ARCHITECTURAL AXIOM:
Attestation is a structured, deterministic claim about peer architecture and runtime
characteristics. It is NOT production cryptographic proof.
Claiming status=CRYPTOGRAPHICALLY_AUTHENTICATED raises UnsupportedSecurityModeError.
"""

from typing import Dict, Tuple, Any, Optional

from chakrview.cognition.peering.models import (
    PeerAttestation,
    AttestationStatus,
)
from chakrview.cognition.peering.identity import UnsupportedSecurityModeError


class AttestationVerificationError(ValueError):
    """Raised when an attestation claim is malformed or invalid."""
    pass


class PeerAttestationVerifier:
    """
    Deterministic structural and architectural claim verifier for peer attestations.
    """

    EXPECTED_ARCHITECTURE_VERSION = "0.1"
    EXPECTED_PROTOCOL_VERSION = "29.0"

    @classmethod
    def create_attestation(
        cls,
        attestation_id: str,
        peer_id: str,
        zone_id: str,
        capability_manifest: Optional[Dict[str, Any]] = None,
        policy_version: str = "29.0",
        runtime_integrity_hash: str = "0" * 64,
        declared_epoch: int = 1,
    ) -> PeerAttestation:
        """Create a structured attestation claim."""
        manifest = dict(capability_manifest or {
            "supported_roles": ["ANALYST", "RESEARCHER", "CRITIC", "SYNTHESIZER", "VERIFIER"],
            "max_concurrency": 4,
            "token_ceiling": 512,
        })
        return PeerAttestation(
            attestation_id=attestation_id,
            peer_id=peer_id,
            zone_id=zone_id,
            architecture_version=cls.EXPECTED_ARCHITECTURE_VERSION,
            protocol_version=cls.EXPECTED_PROTOCOL_VERSION,
            capability_manifest=manifest,
            policy_version=policy_version,
            runtime_integrity_hash=runtime_integrity_hash,
            declared_epoch=declared_epoch,
            status=AttestationStatus.DECLARED,
        )

    @classmethod
    def verify_attestation(
        cls,
        attestation: PeerAttestation,
        expected_peer_id: Optional[str] = None,
        expected_zone_id: Optional[str] = None,
    ) -> Tuple[bool, str]:
        """
        Verify peer attestation claims against expected architectural parameters.
        Returns (is_valid, reason).
        """
        # Guard against unsupported cryptographic claim
        if attestation.status == AttestationStatus.CRYPTOGRAPHICALLY_AUTHENTICATED:
            raise UnsupportedSecurityModeError(
                "Attestation status CRYPTOGRAPHICALLY_AUTHENTICATED is unsupported in Step 29."
            )

        if expected_peer_id and attestation.peer_id != expected_peer_id:
            return False, f"Attestation peer_id '{attestation.peer_id}' does not match expected '{expected_peer_id}'."

        if expected_zone_id and attestation.zone_id != expected_zone_id:
            return False, f"Attestation zone_id '{attestation.zone_id}' does not match expected '{expected_zone_id}'."

        if attestation.architecture_version != cls.EXPECTED_ARCHITECTURE_VERSION:
            return False, (
                f"Incompatible architecture version '{attestation.architecture_version}'. "
                f"Expected '{cls.EXPECTED_ARCHITECTURE_VERSION}' (ChakrMicro v0.1)."
            )

        if attestation.protocol_version != cls.EXPECTED_PROTOCOL_VERSION:
            return False, (
                f"Incompatible protocol version '{attestation.protocol_version}'. "
                f"Expected '{cls.EXPECTED_PROTOCOL_VERSION}'."
            )

        if not attestation.capability_manifest or not isinstance(attestation.capability_manifest, dict):
            return False, "Attestation capability_manifest must be a non-empty dictionary."

        token_ceiling = attestation.capability_manifest.get("token_ceiling", 0)
        if token_ceiling > 512:
            return False, f"Capability token_ceiling {token_ceiling} exceeds ChakrMicro maximum context (512)."

        if not attestation.runtime_integrity_hash or len(attestation.runtime_integrity_hash) < 32:
            return False, "Runtime integrity hash missing or invalid."

        return True, "Attestation verified successfully."
