"""
Secure Peer Session Model for Cross-Zone Federation (Step 30).

CRITICAL ARCHITECTURAL AXIOMS:
1. SESSION != AUTHORITY:
   An active secure session enables authenticated message exchange; it confers
   zero local capability authority.
2. ZERO SECRET STORAGE:
   Sessions contain cryptographic public keys, fingerprints, and replay state,
   never private keys or sensitive reasoning activations.
3. BOUNDED DETERMINISTIC LIFETIME:
   Every session has an explicit expires_at_epoch and bounded replay protection window.
"""

from dataclasses import dataclass, field
from enum import Enum
import time
from typing import Dict, List, Optional, Any, Set, Tuple

from chakrview.cognition.peering.crypto import Ed25519PublicKeyWrapper
from chakrview.cognition.peering.authentication import PeerAuthenticationState


class SessionStatus(str, Enum):
    """Lifecycle operational state of a secure peer session."""
    INITIATED = "INITIATED"
    AUTHENTICATING = "AUTHENTICATING"
    ACTIVE = "ACTIVE"
    EXPIRED = "EXPIRED"
    TERMINATED = "TERMINATED"
    REVOKED = "REVOKED"


@dataclass
class SecurePeerSession:
    """
    Logical secure session established between local zone and an authenticated remote peer.
    """
    session_id: str
    local_peer_id: str
    remote_peer_id: str
    local_zone_id: str
    remote_zone_id: str
    protocol_version: str = "30.0"
    status: SessionStatus = SessionStatus.INITIATED
    auth_state: PeerAuthenticationState = PeerAuthenticationState.UNAUTHENTICATED
    created_epoch: int = 1
    authenticated_at_epoch: Optional[int] = None
    expires_at_epoch: int = 50
    remote_public_key: Optional[Ed25519PublicKeyWrapper] = None
    trust_grant_id: Optional[str] = None
    transport_type: str = "loopback"
    max_message_history: int = 1000

    # Bounded replay cache: message_id set + FIFO list
    _seen_message_ids: Set[str] = field(default_factory=set, repr=False)
    _message_id_fifo: List[str] = field(default_factory=list, repr=False)

    def is_active(self, current_epoch: int) -> Tuple[bool, str]:
        """
        Verify if session is in ACTIVE state and within validity epoch.
        """
        if self.status == SessionStatus.REVOKED:
            return False, "Session has been explicitly REVOKED"
        if self.status == SessionStatus.TERMINATED:
            return False, "Session has been TERMINATED"
        if self.status != SessionStatus.ACTIVE:
            return False, f"Session is not active (current status: {self.status.value})"
        if current_epoch > self.expires_at_epoch:
            self.status = SessionStatus.EXPIRED
            return False, f"Session EXPIRED at epoch {self.expires_at_epoch} (current {current_epoch})"
        return True, "Session is active and valid"

    def mark_authenticated(
        self,
        remote_public_key: Ed25519PublicKeyWrapper,
        authenticated_epoch: int,
        trust_grant_id: Optional[str] = None,
    ) -> None:
        """Transition session to authenticated active state."""
        self.remote_public_key = remote_public_key
        self.authenticated_at_epoch = authenticated_epoch
        self.auth_state = PeerAuthenticationState.CRYPTOGRAPHICALLY_AUTHENTICATED
        self.status = SessionStatus.ACTIVE
        self.trust_grant_id = trust_grant_id

    def record_and_check_message_id(self, message_id: str) -> bool:
        """
        Verify message ID freshness and record in replay cache.
        Returns True if fresh (not replayed), False if already seen (replay detected).
        """
        if not message_id or message_id in self._seen_message_ids:
            return False

        self._seen_message_ids.add(message_id)
        self._message_id_fifo.append(message_id)

        # Enforce memory bounding
        if len(self._message_id_fifo) > self.max_message_history:
            oldest = self._message_id_fifo.pop(0)
            self._seen_message_ids.discard(oldest)

        return True

    def terminate(self, reason: str = "Administrative termination") -> None:
        """Cleanly terminate the session."""
        self.status = SessionStatus.TERMINATED

    def revoke(self, reason: str = "Security policy violation") -> None:
        """Revoke the session immediately (fail closed)."""
        self.status = SessionStatus.REVOKED

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "local_peer_id": self.local_peer_id,
            "remote_peer_id": self.remote_peer_id,
            "local_zone_id": self.local_zone_id,
            "remote_zone_id": self.remote_zone_id,
            "protocol_version": self.protocol_version,
            "status": self.status.value,
            "auth_state": self.auth_state.value,
            "created_epoch": self.created_epoch,
            "authenticated_at_epoch": self.authenticated_at_epoch,
            "expires_at_epoch": self.expires_at_epoch,
            "remote_public_fingerprint": self.remote_public_key.fingerprint if self.remote_public_key else None,
            "trust_grant_id": self.trust_grant_id,
            "transport_type": self.transport_type,
            "replay_cache_size": len(self._seen_message_ids),
        }
