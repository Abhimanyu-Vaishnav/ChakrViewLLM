"""
Bounded Challenge-Response Peer Authentication Protocol (Step 30).

CRITICAL ARCHITECTURAL AXIOMS:
1. AUTHENTICATION != AUTHORIZATION:
   Proof of private key possession confirms peer identity ownership, NOT authority.
2. AUTHENTICATION != TRUST:
   A successfully authenticated peer is in state CRYPTOGRAPHICALLY_AUTHENTICATED;
   trust requires subsequent bounded negotiation under local policy.
3. FAIL-CLOSED REPLAY PROTECTION:
   Challenges contain cryptographically random 256-bit nonces, bounded epoch TTLs,
   strict session binding, and are strictly consumed upon first presentation.
"""

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import secrets
import time
from typing import Dict, List, Optional, Any, Set, Tuple

from chakrview.cognition.peering.crypto import (
    Ed25519PrivateKeyWrapper,
    Ed25519PublicKeyWrapper,
    CryptographicPeerIdentity,
    KeyLifecycleState,
    SignatureVerificationError,
)


class PeerAuthenticationState(str, Enum):
    """Lifecycle state of peer cryptographic authentication."""
    UNAUTHENTICATED = "UNAUTHENTICATED"
    CHALLENGE_ISSUED = "CHALLENGE_ISSUED"
    CRYPTOGRAPHICALLY_AUTHENTICATED = "CRYPTOGRAPHICALLY_AUTHENTICATED"
    AUTHENTICATION_FAILED = "AUTHENTICATION_FAILED"
    EXPIRED = "EXPIRED"
    REVOKED = "REVOKED"


class AuthenticationError(Exception):
    """Base exception for authentication failures."""
    pass


class ReplayedChallengeError(AuthenticationError):
    """Raised when a replayed challenge or nonce is presented."""
    pass


class ChallengeExpiredError(AuthenticationError):
    """Raised when a challenge has passed its validity epoch."""
    pass


@dataclass(frozen=True)
class AuthChallenge:
    """
    Cryptographic challenge issued to verify peer private-key possession.
    """
    challenge_id: str
    session_id: str
    challenger_peer_id: str
    target_peer_id: str
    nonce: str  # 64 hex characters (32 bytes cryptographically secure random)
    created_epoch: int
    expires_epoch: int
    protocol_version: str = "30.0"

    def canonical_payload(self) -> bytes:
        """
        Deterministic canonical payload bound to protocol, session, peer identities, and epoch.
        """
        doc = {
            "protocol_version": self.protocol_version,
            "challenge_id": self.challenge_id,
            "session_id": self.session_id,
            "challenger_peer_id": self.challenger_peer_id,
            "target_peer_id": self.target_peer_id,
            "nonce": self.nonce,
            "created_epoch": self.created_epoch,
            "expires_epoch": self.expires_epoch,
        }
        canonical_json = json.dumps(doc, sort_keys=True, separators=(",", ":"))
        return canonical_json.encode("utf-8")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "challenge_id": self.challenge_id,
            "session_id": self.session_id,
            "challenger_peer_id": self.challenger_peer_id,
            "target_peer_id": self.target_peer_id,
            "nonce": self.nonce,
            "created_epoch": self.created_epoch,
            "expires_epoch": self.expires_epoch,
            "protocol_version": self.protocol_version,
        }


@dataclass(frozen=True)
class AuthChallengeResponse:
    """
    Signed response to an AuthChallenge proving private-key possession.
    """
    challenge_id: str
    session_id: str
    signer_peer_id: str
    signature_hex: str
    response_epoch: int
    protocol_version: str = "30.0"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "challenge_id": self.challenge_id,
            "session_id": self.session_id,
            "signer_peer_id": self.signer_peer_id,
            "signature_hex": self.signature_hex,
            "response_epoch": self.response_epoch,
            "protocol_version": self.protocol_version,
        }


class ChallengeResponseAuthenticator:
    """
    Orchestrates bounded challenge issuance, signing, and verification.
    Guarantees replay protection and one-time consumption.
    """

    def __init__(self, max_consumed_history: int = 1000) -> None:
        self.max_consumed_history = max_consumed_history
        # Active issued challenges: challenge_id -> AuthChallenge
        self._active_challenges: Dict[str, AuthChallenge] = {}
        # Set of consumed nonces to prevent replay
        self._consumed_nonces: Set[str] = set()
        # FIFO tracking of nonces to keep memory strictly bounded
        self._nonce_fifo: List[str] = []

    def issue_challenge(
        self,
        session_id: str,
        challenger_peer_id: str,
        target_peer_id: str,
        current_epoch: int,
        ttl_epochs: int = 2,
        protocol_version: str = "30.0",
    ) -> AuthChallenge:
        """
        Generate a cryptographically fresh challenge with secure random nonce.
        """
        nonce = secrets.token_bytes(32).hex()
        challenge_id = f"chal_{secrets.token_hex(8)}"
        challenge = AuthChallenge(
            challenge_id=challenge_id,
            session_id=session_id,
            challenger_peer_id=challenger_peer_id,
            target_peer_id=target_peer_id,
            nonce=nonce,
            created_epoch=current_epoch,
            expires_epoch=current_epoch + ttl_epochs,
            protocol_version=protocol_version,
        )
        self._active_challenges[challenge_id] = challenge
        return challenge

    @staticmethod
    def sign_challenge(
        challenge: AuthChallenge,
        private_key: Ed25519PrivateKeyWrapper,
        signer_peer_id: str,
        current_epoch: int,
    ) -> AuthChallengeResponse:
        """
        Sign the canonical challenge payload using peer private key.
        """
        payload = challenge.canonical_payload()
        sig_hex = private_key.sign_hex(payload)
        return AuthChallengeResponse(
            challenge_id=challenge.challenge_id,
            session_id=challenge.session_id,
            signer_peer_id=signer_peer_id,
            signature_hex=sig_hex,
            response_epoch=current_epoch,
            protocol_version=challenge.protocol_version,
        )

    def verify_response(
        self,
        response: AuthChallengeResponse,
        peer_identity: CryptographicPeerIdentity,
        current_epoch: int,
    ) -> Tuple[bool, str]:
        """
        Verify peer challenge response against the issued challenge.
        Enforces one-time consumption, expiry, signature correctness, and key validity.
        """
        challenge = self._active_challenges.get(response.challenge_id)
        if not challenge:
            return False, f"Challenge '{response.challenge_id}' not found or already consumed."

        # Verify key state
        key_valid, key_reason = peer_identity.is_valid(current_epoch)
        if not key_valid:
            return False, f"Peer key validation failed: {key_reason}"

        # Verify session binding
        if challenge.session_id != response.session_id:
            return False, f"Session mismatch: challenge bound to {challenge.session_id}, got {response.session_id}."

        # Verify peer identifier binding
        if challenge.target_peer_id != response.signer_peer_id or challenge.target_peer_id != peer_identity.peer_id:
            return False, f"Peer identity mismatch: expected {challenge.target_peer_id}, got {response.signer_peer_id}."

        # Check nonce replay
        if challenge.nonce in self._consumed_nonces:
            return False, f"Replay attack detected: Nonce '{challenge.nonce[:12]}' was already consumed."

        # Check challenge expiration
        if current_epoch > challenge.expires_epoch:
            # Clean up expired challenge
            del self._active_challenges[response.challenge_id]
            return False, f"Challenge expired at epoch {challenge.expires_epoch} (current {current_epoch})."

        # Verify cryptographic signature
        try:
            sig_bytes = bytes.fromhex(response.signature_hex)
        except ValueError:
            return False, "Malformed signature hex encoding."

        canonical_bytes = challenge.canonical_payload()
        if not peer_identity.public_key.verify(sig_bytes, canonical_bytes):
            return False, "Ed25519 signature verification failed mathematically."

        # Mark challenge as consumed (one-time consumption)
        del self._active_challenges[response.challenge_id]
        self._record_consumed_nonce(challenge.nonce)

        return True, "Challenge successfully authenticated."

    def _record_consumed_nonce(self, nonce: str) -> None:
        """Record consumed nonce with strict FIFO bounded memory."""
        self._consumed_nonces.add(nonce)
        self._nonce_fifo.append(nonce)
        if len(self._nonce_fifo) > self.max_consumed_history:
            oldest = self._nonce_fifo.pop(0)
            self._consumed_nonces.discard(oldest)

    def prune_expired(self, current_epoch: int) -> int:
        """Prune active challenges that have exceeded their expires_epoch."""
        expired_ids = [
            cid for cid, chal in self._active_challenges.items()
            if current_epoch > chal.expires_epoch
        ]
        for cid in expired_ids:
            del self._active_challenges[cid]
        return len(expired_ids)
