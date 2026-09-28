"""
Cryptographic Identity, Authentication, and Replay Protection (Step 27).

Establishes the security foundation for distributed agent transport:
- ReplayProtectionTracker: Bounded memory tracker rejecting replayed, expired, or excessively hopped messages.
- MessageSigner & MessageVerifier: Deterministic cryptographic signing and attestation.
- Identity Providers: Node and Agent identity separation.
- Authentication vs Authorization: Strict decoupling where authentication NEVER automatically grants capability authorization.

CRITICAL ARCHITECTURAL AXIOMS:
1. AUTHENTICATION != AUTHORIZATION:
   Cryptographically proving node or agent identity does NOT grant capability permissions.
   All external capabilities remain strictly gated by CapabilityGate.
2. BOUNDED REPLAY STATE:
   Replay memory is strictly capped to prevent memory exhaustion attacks.
"""

from abc import ABC, abstractmethod
import collections
import hashlib
import hmac
import time
from typing import Dict, Optional, Set, Tuple, Any
import uuid

from chakrview.cognition.distributed.models import (
    DistributedMessageEnvelope,
    MessageHeader,
    MessageRoute,
    MessageIntegrity,
    NodeIdentity,
    DEFAULT_MESSAGE_TTL_SECONDS,
    MAX_MESSAGE_HOPS,
)
from chakrview.cognition.federated.models import AgentIdentity, MessageType


class SecurityError(PermissionError):
    """Base exception for security boundary failures."""
    pass


class ReplayAttackError(SecurityError):
    """Raised when duplicate message ID or nonce is replayed."""
    pass


class MessageExpiredError(SecurityError):
    """Raised when message TTL has elapsed."""
    pass


class ExcessiveHopsError(SecurityError):
    """Raised when hop count exceeds maximum allowable ceiling."""
    pass


class TenantRoutingError(SecurityError):
    """Raised when cross-tenant message exchange is attempted."""
    pass


class MessageTamperingError(SecurityError):
    """Raised when cryptographic fingerprint or signature is invalid."""
    pass


# ============================================================================
# 1. Replay Protection Tracker
# ============================================================================

class ReplayProtectionTracker:
    """
    Deterministic bounded-memory replay protection tracker.
    Guarantees that replayed messages, duplicate nonces, expired TTLs,
    and excessive hops fail closed.
    """

    def __init__(self, max_tracked: int = 1024, default_ttl: float = DEFAULT_MESSAGE_TTL_SECONDS) -> None:
        self.max_tracked = max_tracked
        self.default_ttl = default_ttl
        # message_id -> expiration_timestamp
        self._processed_message_ids: collections.OrderedDict[str, float] = collections.OrderedDict()
        # (sender_node_id, nonce) -> expiration_timestamp
        self._processed_nonces: collections.OrderedDict[Tuple[str, str], float] = collections.OrderedDict()
        # (sender_node_id, sender_agent_id) -> last_seen_logical_clock
        self._sender_clocks: Dict[Tuple[str, str], int] = {}

    def _prune_expired(self, current_time: float) -> None:
        """Evict expired entries and enforce capacity ceiling."""
        # Prune message IDs
        expired_ids = [mid for mid, exp in self._processed_message_ids.items() if exp < current_time]
        for mid in expired_ids:
            del self._processed_message_ids[mid]

        while len(self._processed_message_ids) >= self.max_tracked:
            self._processed_message_ids.popitem(last=False)

        # Prune nonces
        expired_nonces = [n for n, exp in self._processed_nonces.items() if exp < current_time]
        for n in expired_nonces:
            del self._processed_nonces[n]

        while len(self._processed_nonces) >= self.max_tracked:
            self._processed_nonces.popitem(last=False)

    def validate_and_record(
        self,
        envelope: DistributedMessageEnvelope,
        current_time: Optional[float] = None,
        expected_tenant_id: Optional[str] = None,
    ) -> bool:
        """
        Validate envelope against replay, expiration, hops, and tenant boundaries.
        Records validated message to prevent subsequent replays.
        """
        now = current_time if current_time is not None else time.time()
        self._prune_expired(now)

        header = envelope.header
        route = envelope.route
        integrity = envelope.integrity

        # 1. Tenant boundary enforcement
        if expected_tenant_id and route.tenant_id != expected_tenant_id:
            raise TenantRoutingError(
                f"Tenant boundary violation: envelope tenant '{route.tenant_id}' "
                f"does not match expected tenant '{expected_tenant_id}'."
            )

        # 2. Expiration check
        if (now - header.timestamp) > header.ttl_seconds:
            raise MessageExpiredError(
                f"Message '{header.message_id}' expired. Age: {now - header.timestamp:.2f}s, "
                f"TTL: {header.ttl_seconds}s."
            )

        # 3. Hop count check
        if header.hop_count > header.max_hops:
            raise ExcessiveHopsError(
                f"Message '{header.message_id}' exceeded max hops ({header.hop_count} > {header.max_hops})."
            )

        # 4. Message ID uniqueness
        if header.message_id in self._processed_message_ids:
            raise ReplayAttackError(f"Replay detected: duplicate message ID '{header.message_id}'.")

        # 5. Nonce uniqueness per sender node
        nonce_key = (route.sender_node_id, integrity.nonce)
        if nonce_key in self._processed_nonces:
            raise ReplayAttackError(
                f"Replay detected: duplicate nonce '{integrity.nonce}' from node '{route.sender_node_id}'."
            )

        # 6. Logical clock progression check (optional monotonically non-decreasing)
        sender_key = (route.sender_node_id, route.sender_agent_id)
        last_clock = self._sender_clocks.get(sender_key, -1)
        if header.logical_clock < last_clock:
            raise ReplayAttackError(
                f"Invalid logical clock progression for sender {sender_key}: "
                f"{header.logical_clock} < {last_clock}."
            )

        # Record validation
        expires_at = header.timestamp + header.ttl_seconds
        self._processed_message_ids[header.message_id] = expires_at
        self._processed_nonces[nonce_key] = expires_at
        self._sender_clocks[sender_key] = max(last_clock, header.logical_clock)

        return True


# ============================================================================
# 2. Cryptographic Signing & Attestation
# ============================================================================

class MessageSigner(ABC):
    """Abstract interface for cryptographic envelope signing."""

    @abstractmethod
    def sign(self, envelope_bytes: bytes) -> str:
        """Produce a signature or authentication code over canonical bytes."""
        pass


class MessageVerifier(ABC):
    """Abstract interface for cryptographic signature verification."""

    @abstractmethod
    def verify(self, envelope_bytes: bytes, signature: str, signer_identity: str) -> bool:
        """Verify the signature against canonical bytes."""
        pass


class DeterministicHmacMessageSigner(MessageSigner):
    """Deterministic HMAC-SHA256 signer using node secret keys."""

    def __init__(self, secret_key: bytes) -> None:
        self.secret_key = secret_key

    def sign(self, envelope_bytes: bytes) -> str:
        return hmac.new(self.secret_key, envelope_bytes, hashlib.sha256).hexdigest()


class DeterministicHmacMessageVerifier(MessageVerifier):
    """Deterministic HMAC-SHA256 verifier with key registry."""

    def __init__(self, key_registry: Optional[Dict[str, bytes]] = None) -> None:
        self._key_registry = dict(key_registry or {})

    def register_key(self, identity_id: str, secret_key: bytes) -> None:
        self._key_registry[identity_id] = secret_key

    def verify(self, envelope_bytes: bytes, signature: str, signer_identity: str) -> bool:
        secret_key = self._key_registry.get(signer_identity)
        if not secret_key:
            return False
        expected_sig = hmac.new(secret_key, envelope_bytes, hashlib.sha256).hexdigest()
        return hmac.compare_digest(signature, expected_sig)


# ============================================================================
# 3. Identity & Attestation Providers
# ============================================================================

class NodeIdentityProvider(ABC):
    """Provider managing local node credentials and identity."""

    @abstractmethod
    def get_identity(self) -> NodeIdentity:
        pass


class LocalNodeIdentityProvider(NodeIdentityProvider):
    """In-memory identity provider for local testing and single/multi-node execution."""

    def __init__(self, identity: NodeIdentity) -> None:
        self._identity = identity

    def get_identity(self) -> NodeIdentity:
        return self._identity


class AttestationProvider(ABC):
    """Pluggable attestation engine for node trust verification."""

    @abstractmethod
    def attest_node(self, node_identity: NodeIdentity) -> bool:
        pass


class LocalAttestationProvider(AttestationProvider):
    """Deterministic local attestation validating matching cluster ID and version."""

    def __init__(self, expected_cluster_id: str = "chakrview_cluster_local") -> None:
        self.expected_cluster_id = expected_cluster_id

    def attest_node(self, node_identity: NodeIdentity) -> bool:
        return (
            node_identity.cluster_id == self.expected_cluster_id
            and bool(node_identity.node_id)
            and bool(node_identity.tenant_id)
        )


# ============================================================================
# 4. Envelope Factory Helper
# ============================================================================

def create_distributed_envelope(
    sender_node_id: str,
    sender_agent_id: str,
    receiver_node_id: str,
    receiver_agent_id: str,
    tenant_id: str,
    session_id: str,
    message_type: MessageType,
    payload: Dict[str, Any],
    correlation_id: Optional[str] = None,
    causation_id: Optional[str] = None,
    signer: Optional[MessageSigner] = None,
    signer_identity: Optional[str] = None,
    ttl_seconds: float = DEFAULT_MESSAGE_TTL_SECONDS,
    hop_count: int = 0,
    logical_clock: int = 0,
) -> DistributedMessageEnvelope:
    """
    Factory function creating a well-formed, tamper-evident DistributedMessageEnvelope.
    """
    msg_id = f"dmsg_{uuid.uuid4().hex[:12]}"
    corr_id = correlation_id or f"corr_{uuid.uuid4().hex[:10]}"
    nonce = f"nonce_{uuid.uuid4().hex[:8]}"

    header = MessageHeader(
        message_id=msg_id,
        correlation_id=corr_id,
        causation_id=causation_id,
        message_type=message_type,
        timestamp=time.time(),
        logical_clock=logical_clock,
        ttl_seconds=ttl_seconds,
        hop_count=hop_count,
        max_hops=MAX_MESSAGE_HOPS,
    )

    route = MessageRoute(
        sender_node_id=sender_node_id,
        sender_agent_id=sender_agent_id,
        receiver_node_id=receiver_node_id,
        receiver_agent_id=receiver_agent_id,
        tenant_id=tenant_id,
        session_id=session_id,
    )

    # Temporary envelope to compute canonical digest
    temp_envelope = DistributedMessageEnvelope(
        envelope_id=f"env_{msg_id}",
        header=header,
        route=route,
        integrity=MessageIntegrity(nonce=nonce),
        payload=payload,
    )

    fingerprint = temp_envelope.compute_integrity_hash()
    signature = None
    if signer and signer_identity:
        signature = signer.sign(temp_envelope.canonical_serialize())

    integrity = MessageIntegrity(
        fingerprint=fingerprint,
        nonce=nonce,
        signature=signature,
        signer_identity=signer_identity,
    )

    return DistributedMessageEnvelope(
        envelope_id=temp_envelope.envelope_id,
        header=header,
        route=route,
        integrity=integrity,
        payload=payload,
    )
