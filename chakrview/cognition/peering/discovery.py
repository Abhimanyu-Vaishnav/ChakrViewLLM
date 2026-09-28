"""
Peer Discovery Abstraction for Cross-Zone Federation (Step 29).

CRITICAL ARCHITECTURAL AXIOM:
DISCOVERY != TRUST:
A discovered peer is merely observed or declared. Discovery NEVER establishes trust,
authorizes capability execution, or permits task delegation.
The required lifecycle is strictly:
DISCOVER -> IDENTIFY -> ATTEST -> POLICY CHECK -> TRUST NEGOTIATION -> LIMITED FEDERATION.
"""

from typing import Dict, List, Optional, Tuple

from chakrview.cognition.peering.models import (
    PeerDeclaration,
    PeerRegistration,
    DiscoveryStatus,
    TrustLevel,
)
from chakrview.cognition.peering.identity import PeerIdentityProvider
from chakrview.cognition.peering.attestation import PeerAttestationVerifier
from chakrview.cognition.peering.policy import CrossZoneFederationPolicy


class PeerDiscoveryManager:
    """
    Manages deterministic discovery of external cognitive zones without implicit trust.
    """

    def __init__(self, policy: Optional[CrossZoneFederationPolicy] = None) -> None:
        self.policy = policy or CrossZoneFederationPolicy()
        self._discovered: Dict[str, PeerRegistration] = {}

    def discover(self, declaration: PeerDeclaration, current_epoch: int = 1) -> PeerRegistration:
        """
        Record a newly discovered peer declaration.
        Initial status is DISCOVERED, trust is strictly NONE (trust_grant is None).
        """
        # Validate identity formatting and canonical fingerprint
        PeerIdentityProvider.validate_identifier("peer_id", declaration.identity.peer_id)
        PeerIdentityProvider.validate_identifier("zone_id", declaration.identity.zone_id)
        PeerIdentityProvider.validate_identifier("organization_id", declaration.identity.organization_id)

        # Enforce policy admission check
        admissible, reason = self.policy.is_peer_admissible(declaration.identity)
        status = DiscoveryStatus.DISCOVERED if admissible else DiscoveryStatus.REJECTED

        registration = PeerRegistration(
            identity=declaration.identity,
            discovery_status=status,
            trust_grant=None,  # Strict: DISCOVER != TRUST
            latest_attestation=declaration.attestation,
            registered_epoch=current_epoch,
            last_seen_epoch=current_epoch,
            revocation_record=None,
        )

        self._discovered[declaration.identity.peer_id] = registration
        return registration

    def get_registration(self, peer_id: str) -> Optional[PeerRegistration]:
        return self._discovered.get(peer_id)

    def list_discovered(self, status: Optional[DiscoveryStatus] = None) -> List[PeerRegistration]:
        if status is None:
            return list(self._discovered.values())
        return [r for r in self._discovered.values() if r.discovery_status == status]

    def update_status(self, peer_id: str, new_status: DiscoveryStatus, epoch: int) -> bool:
        if peer_id not in self._discovered:
            return False
        reg = self._discovered[peer_id]
        reg.discovery_status = new_status
        reg.last_seen_epoch = epoch
        return True
