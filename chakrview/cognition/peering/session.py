"""
Secure Peer Session Model for Cross-Zone Federation (Step 30 & 32).

CRITICAL ARCHITECTURAL AXIOMS:
1. SESSION != AUTHORITY & SESSION != TRUST:
   An active secure session enables authenticated message exchange; it confers
   zero local capability authority or federation trust.
2. ZERO SECRET STORAGE:
   Sessions contain cryptographic public keys, fingerprints, and replay state,
   never private keys, raw tokens, or sensitive reasoning activations.
3. BOUNDED DETERMINISTIC LIFETIME & FRESHNESS:
   Every session has an explicit created_epoch, expires_at_epoch, max_lifetime_epochs,
   bounded renewal limits, and bounded replay protection window.
4. FAIL-CLOSED STATE MACHINE:
   Illegal transitions raise SessionTransitionError. Terminal states (TERMINATED,
   REVOKED, FAILED) can never transition back to ACTIVE. Expired sessions fail closed.
"""

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import secrets
import time
from typing import Dict, List, Optional, Any, Set, Tuple

from chakrview.cognition.peering.crypto import Ed25519PublicKeyWrapper
from chakrview.cognition.peering.authentication import PeerAuthenticationState


class SessionStatus(str, Enum):
    """Lifecycle operational state of a secure peer session."""
    INITIATED = "INITIATED"
    AUTHENTICATING = "AUTHENTICATING"
    ACTIVE = "ACTIVE"
    RENEWING = "RENEWING"
    EXPIRED = "EXPIRED"
    TERMINATED = "TERMINATED"
    REVOKED = "REVOKED"
    FAILED = "FAILED"


class SessionKeyState(str, Enum):
    """Lifecycle state of session key metadata."""
    CREATED = "CREATED"
    ACTIVE = "ACTIVE"
    ROTATING = "ROTATING"
    EXPIRED = "EXPIRED"
    REVOKED = "REVOKED"


class SessionTransitionError(PermissionError):
    """Raised when an illegal session state transition is attempted."""
    pass


VALID_SESSION_TRANSITIONS: Dict[SessionStatus, Set[SessionStatus]] = {
    SessionStatus.INITIATED: {
        SessionStatus.AUTHENTICATING,
        SessionStatus.FAILED,
        SessionStatus.TERMINATED,
        SessionStatus.REVOKED,
    },
    SessionStatus.AUTHENTICATING: {
        SessionStatus.ACTIVE,
        SessionStatus.FAILED,
        SessionStatus.TERMINATED,
        SessionStatus.REVOKED,
    },
    SessionStatus.ACTIVE: {
        SessionStatus.RENEWING,
        SessionStatus.EXPIRED,
        SessionStatus.TERMINATED,
        SessionStatus.REVOKED,
    },
    SessionStatus.RENEWING: {
        SessionStatus.ACTIVE,
        SessionStatus.FAILED,
        SessionStatus.EXPIRED,
        SessionStatus.TERMINATED,
        SessionStatus.REVOKED,
    },
    SessionStatus.EXPIRED: {
        SessionStatus.TERMINATED,
        SessionStatus.REVOKED,
    },
    SessionStatus.TERMINATED: set(),
    SessionStatus.REVOKED: set(),
    SessionStatus.FAILED: {
        SessionStatus.TERMINATED,
        SessionStatus.REVOKED,
    },
}


@dataclass
class SessionKeyMetadata:
    """
    Metadata representation of a session key lifecycle.
    Never exposes raw symmetric or asymmetric key secrets.
    """
    key_id: str
    session_id: str = ""
    key_state: SessionKeyState = SessionKeyState.CREATED
    created_epoch: int = 1
    expires_epoch: int = 50
    key_fingerprint: str = ""
    rotation_count: int = 0

    def __repr__(self) -> str:
        return (
            f"<SessionKeyMetadata key_id={self.key_id} "
            f"fingerprint={self.key_fingerprint[:12]}... state={self.key_state.value}>"
        )

    def __str__(self) -> str:
        return self.__repr__()

    def to_dict(self) -> Dict[str, Any]:
        """Safe non-secret dictionary export."""
        return {
            "key_id": self.key_id,
            "session_id": self.session_id,
            "key_state": self.key_state.value,
            "created_epoch": self.created_epoch,
            "expires_epoch": self.expires_epoch,
            "key_fingerprint": self.key_fingerprint,
            "rotation_count": self.rotation_count,
        }

    def invalidate(self, revoked: bool = False) -> None:
        """Invalidate session key upon session expiration or revocation."""
        self.key_state = SessionKeyState.REVOKED if revoked else SessionKeyState.EXPIRED

    def activate(self, activated_epoch: int) -> None:
        """Activate session key."""
        self.key_state = SessionKeyState.ACTIVE
        self.created_epoch = activated_epoch

    def is_valid(self, current_epoch: int) -> bool:
        """Check if session key is active and unexpired."""
        if self.key_state != SessionKeyState.ACTIVE:
            return False
        if current_epoch > self.expires_epoch:
            return False
        return True

    @classmethod
    def create(cls, session_id: str, created_epoch: int = 1, expires_epoch: int = 50) -> "SessionKeyMetadata":
        """Factory for new session key in CREATED state."""
        return cls(
            key_id=f"sk_{session_id}_{created_epoch}",
            session_id=session_id,
            key_state=SessionKeyState.CREATED,
            created_epoch=created_epoch,
            expires_epoch=expires_epoch,
        )


@dataclass
class SecurePeerSession:
    """
    Logical secure session established between local zone and an authenticated remote peer.
    Governed by strict state transitions, freshness ceilings, and bounded replay defenses.
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

    # Step 32 Freshness & Bounded Lifetime Ceilings
    max_lifetime_epochs: int = 200
    renewal_count: int = 0
    max_renewals: int = 5
    last_renewed_epoch: Optional[int] = None
    session_key_metadata: Optional[SessionKeyMetadata] = None
    last_seen_sequence_number: int = 0

    # Bounded replay cache: message_id set + FIFO list
    _seen_message_ids: Set[str] = field(default_factory=set, repr=False)
    _message_id_fifo: List[str] = field(default_factory=list, repr=False)

    @property
    def seen_message_ids(self) -> Set[str]:
        return self._seen_message_ids

    @property
    def last_sequence_number(self) -> int:
        return self.last_seen_sequence_number

    def __post_init__(self) -> None:
        if self.session_key_metadata is None:
            self.session_key_metadata = SessionKeyMetadata.create(
                session_id=self.session_id,
                created_epoch=self.created_epoch,
                expires_epoch=self.expires_at_epoch,
            )

    def transition_to(self, target_status: SessionStatus, reason: Optional[str] = None) -> None:
        """
        Enforce valid state machine transitions.
        Terminal states (TERMINATED, REVOKED, FAILED) can never transition back to ACTIVE.
        """
        if self.status == target_status:
            return

        valid_targets = VALID_SESSION_TRANSITIONS.get(self.status, set())
        if target_status not in valid_targets:
            raise SessionTransitionError(
                f"Illegal session transition from {self.status.value} to {target_status.value} "
                f"for session '{self.session_id}' (reason: {reason or 'none'})."
            )

        self.status = target_status

    def is_active(self, current_epoch: int) -> Tuple[bool, str]:
        """
        Verify if session is in ACTIVE state and within validity epoch.
        Fails closed on terminal or expired states.
        """
        if self.status == SessionStatus.REVOKED:
            return False, "Session has been explicitly REVOKED"
        if self.status == SessionStatus.TERMINATED:
            return False, "Session has been TERMINATED"
        if self.status == SessionStatus.FAILED:
            return False, "Session has FAILED"
        if self.status != SessionStatus.ACTIVE and self.status != SessionStatus.RENEWING:
            return False, f"Session is not active (current status: {self.status.value})"

        # Check epoch freshness
        if current_epoch > self.expires_at_epoch:
            if self.status in (SessionStatus.ACTIVE, SessionStatus.RENEWING):
                self.transition_to(SessionStatus.EXPIRED, reason=f"Epoch {current_epoch} > {self.expires_at_epoch}")
            if self.session_key_metadata:
                self.session_key_metadata.invalidate(revoked=False)
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
        self.trust_grant_id = trust_grant_id

        # Formal state transition
        if self.status == SessionStatus.INITIATED:
            self.transition_to(SessionStatus.AUTHENTICATING, reason="Authenticating peer identity")
            self.transition_to(SessionStatus.ACTIVE, reason="Cryptographic authentication succeeded")
        elif self.status == SessionStatus.AUTHENTICATING:
            self.transition_to(SessionStatus.ACTIVE, reason="Cryptographic authentication succeeded")
        elif self.status != SessionStatus.ACTIVE:
            self.transition_to(SessionStatus.ACTIVE, reason="Re-authentication succeeded")

        # Initialize session key metadata
        key_id = f"sk_{self.session_id}_{authenticated_epoch}"
        fingerprint = hashlib.sha256(f"{self.session_id}:{remote_public_key.fingerprint}".encode("utf-8")).hexdigest()
        self.session_key_metadata = SessionKeyMetadata(
            key_id=key_id,
            key_state=SessionKeyState.ACTIVE,
            created_epoch=authenticated_epoch,
            expires_epoch=self.expires_at_epoch,
            key_fingerprint=fingerprint,
        )

    def can_renew(self, current_epoch: int) -> Tuple[bool, str]:
        """
        Check whether the session is eligible for renewal.
        Enforces freshness, maximum lifetime ceiling, and renewal count limit.
        """
        if self.status == SessionStatus.REVOKED:
            return False, "Cannot renew REVOKED session"
        if self.status == SessionStatus.TERMINATED:
            return False, "Cannot renew TERMINATED session"
        if self.status == SessionStatus.FAILED:
            return False, "Cannot renew FAILED session"
        if self.status == SessionStatus.EXPIRED or current_epoch > self.expires_at_epoch:
            return False, f"Cannot renew EXPIRED session (expired at epoch {self.expires_at_epoch}, current {current_epoch})"

        if self.renewal_count >= self.max_renewals:
            return False, f"Maximum renewal count reached ({self.max_renewals})"

        current_lifetime = self.expires_at_epoch - self.created_epoch
        if current_lifetime >= self.max_lifetime_epochs:
            return False, f"Session has reached maximum lifetime ceiling ({self.max_lifetime_epochs} epochs)"

        return True, "Session is eligible for renewal"

    def renew(self, extension_epochs: int, current_epoch: int) -> None:
        """
        Extend session lifetime under strict bounds.
        Renewal never escalates capability scopes or authority.
        """
        eligible, reason = self.can_renew(current_epoch)
        if not eligible:
            raise SessionTransitionError(f"Session renewal rejected: {reason}")

        if extension_epochs < 1:
            raise ValueError(f"Extension epochs must be >= 1, got {extension_epochs}.")

        # Enforce maximum lifetime ceiling
        max_possible_expiry = self.created_epoch + self.max_lifetime_epochs
        if (self.expires_at_epoch + extension_epochs) > max_possible_expiry:
            raise SessionTransitionError(
                f"Session renewal rejected: extension to {self.expires_at_epoch + extension_epochs} "
                f"exceeds maximum session lifetime of {self.max_lifetime_epochs} epochs (max expiry {max_possible_expiry})."
            )
        new_expiry = self.expires_at_epoch + extension_epochs

        # Transition through RENEWING to ACTIVE
        self.transition_to(SessionStatus.RENEWING, reason="Renewal initiated")
        self.expires_at_epoch = new_expiry
        self.renewal_count += 1
        self.last_renewed_epoch = current_epoch

        if self.session_key_metadata:
            self.session_key_metadata.expires_epoch = new_expiry
            self.session_key_metadata.rotation_count += 1

        self.transition_to(SessionStatus.ACTIVE, reason="Renewal completed")

    def record_and_check_message_id(self, message_id: str) -> bool:
        """
        Verify message ID freshness and record in bounded replay cache.
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

    def record_and_check_sequence(self, sequence_number: Optional[int]) -> bool:
        """
        Optional monotonic sequence validation.
        Rejects out-of-order or duplicate sequence numbers if provided.
        """
        if sequence_number is None:
            return True

        if sequence_number <= self.last_seen_sequence_number:
            return False

        self.last_seen_sequence_number = sequence_number
        return True

    def validate_sequence_number(self, sequence_number: int) -> bool:
        """Validate sequence number against replay floor without advancing it."""
        return sequence_number > self.last_seen_sequence_number

    def terminate(self, reason: str = "Administrative termination") -> None:
        """Cleanly terminate the session and invalidate key material."""
        if self.status not in (SessionStatus.TERMINATED, SessionStatus.REVOKED):
            self.transition_to(SessionStatus.TERMINATED, reason=reason)
        if self.session_key_metadata:
            self.session_key_metadata.invalidate(revoked=False)

    def revoke(self, reason: str = "Security policy violation") -> None:
        """Revoke the session immediately (fail closed) and invalidate key material."""
        if self.status != SessionStatus.REVOKED:
            self.transition_to(SessionStatus.REVOKED, reason=reason)
        if self.session_key_metadata:
            self.session_key_metadata.invalidate(revoked=True)

    def to_dict(self) -> Dict[str, Any]:
        """Safe telemetry dictionary export with zero secrets."""
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
            "max_lifetime_epochs": self.max_lifetime_epochs,
            "renewal_count": self.renewal_count,
            "max_renewals": self.max_renewals,
            "last_renewed_epoch": self.last_renewed_epoch,
            "remote_public_fingerprint": self.remote_public_key.fingerprint if self.remote_public_key else None,
            "trust_grant_id": self.trust_grant_id,
            "transport_type": self.transport_type,
            "replay_cache_size": len(self._seen_message_ids),
            "last_seen_sequence_number": self.last_seen_sequence_number,
            "session_key": self.session_key_metadata.to_dict() if self.session_key_metadata else None,
        }
