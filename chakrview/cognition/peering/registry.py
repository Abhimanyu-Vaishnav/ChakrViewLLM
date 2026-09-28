"""
Deterministic Peer Registry for Cross-Zone Federation (Step 29).

CRITICAL ARCHITECTURAL AXIOMS:
1. HARD CAPACITY CEILINGS:
   Total peers <= 32, peers per zone <= 16, active peers <= 8.
2. UNIQUE PEER IDENTITY:
   Duplicate peer IDs are rejected fail-closed.
3. ZERO SECRET STORAGE:
   Registry records public identity, attestation, and grant metadata only.
   No cryptographic keys, tokens, or credentials may be stored.
"""

from typing import Dict, List, Optional, Set
import threading

from chakrview.cognition.peering.models import (
    PeerRegistration,
    PeerIdentity,
    TrustGrant,
    TrustStatus,
    DiscoveryStatus,
    RevocationRecord,
    MAX_PEERS_TOTAL,
    MAX_PEERS_PER_ZONE,
    MAX_ACTIVE_PEERS,
)
from chakrview.cognition.peering.revocation import RevocationManager


class PeerRegistryCapacityError(RuntimeError):
    """Raised when peer registry exceeds hard architectural capacity limits."""
    pass


class DuplicatePeerError(ValueError):
    """Raised when registering a peer with an already existing peer_id."""
    pass


class PeerRegistry:
    """
    Deterministic in-memory registry of peer registrations, trust grants, and status.
    """

    def __init__(self, revocation_manager: Optional[RevocationManager] = None) -> None:
        self._lock = threading.Lock()
        self._peers: Dict[str, PeerRegistration] = {}
        self.revocation_manager = revocation_manager or RevocationManager()

    def register_peer(self, registration: PeerRegistration) -> PeerRegistration:
        """Register a new peer with ceiling and uniqueness validation."""
        with self._lock:
            peer_id = registration.identity.peer_id
            zone_id = registration.identity.zone_id

            if peer_id in self._peers:
                raise DuplicatePeerError(f"Peer '{peer_id}' is already registered in registry.")

            if len(self._peers) >= MAX_PEERS_TOTAL:
                raise PeerRegistryCapacityError(
                    f"PeerRegistry total capacity limit ({MAX_PEERS_TOTAL}) reached."
                )

            zone_peers = [p for p in self._peers.values() if p.identity.zone_id == zone_id]
            if len(zone_peers) >= MAX_PEERS_PER_ZONE:
                raise PeerRegistryCapacityError(
                    f"Zone '{zone_id}' peer capacity limit ({MAX_PEERS_PER_ZONE}) reached."
                )

            self._peers[peer_id] = registration
            return registration

    def get_peer(self, peer_id: str) -> Optional[PeerRegistration]:
        with self._lock:
            return self._peers.get(peer_id)

    def list_peers(
        self,
        zone_id: Optional[str] = None,
        discovery_status: Optional[DiscoveryStatus] = None,
    ) -> List[PeerRegistration]:
        with self._lock:
            peers = list(self._peers.values())
            if zone_id:
                peers = [p for p in peers if p.identity.zone_id == zone_id]
            if discovery_status:
                peers = [p for p in peers if p.discovery_status == discovery_status]
            return peers

    def update_trust_grant(self, peer_id: str, grant: TrustGrant) -> bool:
        """Attach or update a negotiated trust grant on a peer."""
        with self._lock:
            if peer_id not in self._peers:
                return False

            active_peers = [p for p in self._peers.values() if p.trust_grant and p.trust_grant.status == TrustStatus.ACTIVE]
            if len(active_peers) >= MAX_ACTIVE_PEERS and grant.status == TrustStatus.ACTIVE:
                # If this peer is not already active, enforce ceiling
                current = self._peers[peer_id]
                if not (current.trust_grant and current.trust_grant.status == TrustStatus.ACTIVE):
                    raise PeerRegistryCapacityError(
                        f"Active peers ceiling ({MAX_ACTIVE_PEERS}) reached."
                    )

            reg = self._peers[peer_id]
            reg.trust_grant = grant
            reg.discovery_status = DiscoveryStatus.VERIFIED
            return True

    def revoke_peer(
        self,
        peer_id: str,
        reason: str,
        revoked_epoch: int,
        revoked_by: str = "registry_admin",
    ) -> RevocationRecord:
        """Revoke a peer, invalidating grants and recording revocation."""
        with self._lock:
            if peer_id not in self._peers:
                raise KeyError(f"Peer '{peer_id}' not found in registry.")

            reg = self._peers[peer_id]
            reg.discovery_status = DiscoveryStatus.REVOKED
            active_grant = reg.trust_grant

            rec = self.revocation_manager.revoke(
                peer_id=peer_id,
                zone_id=reg.identity.zone_id,
                reason=reason,
                revoked_epoch=revoked_epoch,
                revoked_by=revoked_by,
                active_grant=active_grant,
            )
            reg.revocation_record = rec
            return rec

    def expire_peers(self, current_epoch: int) -> List[str]:
        """Check all active grants and mark expired ones."""
        expired_peer_ids: List[str] = []
        with self._lock:
            for peer_id, reg in self._peers.items():
                if reg.trust_grant and reg.trust_grant.status == TrustStatus.ACTIVE:
                    if current_epoch > reg.trust_grant.expires_epoch:
                        reg.trust_grant.status = TrustStatus.EXPIRED
                        reg.discovery_status = DiscoveryStatus.EXPIRED
                        expired_peer_ids.append(peer_id)
        return expired_peer_ids

    def remove_peer(self, peer_id: str) -> bool:
        """Remove a peer entry from registry."""
        with self._lock:
            if peer_id in self._peers:
                del self._peers[peer_id]
                return True
            return False
