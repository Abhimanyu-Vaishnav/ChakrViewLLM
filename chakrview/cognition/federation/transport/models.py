"""
Data Models for Production Federation Message Transport (Step 36).

Defines strongly typed, bounded representations for:
- Channel lifecycle states & transitions
- Federation message types and envelopes
- Channel metrics and reconnection policies
"""

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import time
from typing import Dict, List, Optional, Any, Set

from chakrview.cognition.peering.crypto import (
    Ed25519PrivateKeyWrapper,
    Ed25519PublicKeyWrapper,
)
from chakrview.cognition.transport.models import (
    WireEnvelope,
    MessageType,
    DEFAULT_MAX_PAYLOAD_BYTES,
    MAX_WIRE_FRAME_BYTES,
)
from chakrview.cognition.federation.transport.errors import (
    ChannelStateError,
    OversizedFrameError,
    EnvelopeIntegrityError,
    UnknownMessageTypeError,
    ProhibitedPayloadError,
)

FEDERATION_PROTOCOL_VERSION = "36.0"
DEFAULT_MAX_FRAME_SIZE = MAX_WIRE_FRAME_BYTES  # 1 MB
DEFAULT_MAX_PAYLOAD_SIZE = DEFAULT_MAX_PAYLOAD_BYTES  # 64 KB
DEFAULT_CHANNEL_TIMEOUT_SECONDS = 5.0
DEFAULT_MAX_DISPATCHER_QUEUE_SIZE = 100


class ChannelState(str, Enum):
    """
    Explicit operational states of a secure federation channel.
    
    States:
    - DISCONNECTED: Channel has no active network socket.
    - CONNECTING: Transport socket connection in progress.
    - AUTHENTICATING: mTLS handshake & certificate binding validation active.
    - ESTABLISHED: Authenticated, session-bound channel ready for message transmission.
    - DEGRADED: Missed heartbeats or transient network impairment detected.
    - CLOSING: Orderly shutdown handshake initiated.
    - CLOSED: Cleanly decommissioned channel.
    - QUARANTINED: Channel isolated due to security violation or anomaly.
    - REVOKED: Permanently revoked node or channel; terminal absorbing state.
    """
    DISCONNECTED = "DISCONNECTED"
    CONNECTING = "CONNECTING"
    AUTHENTICATING = "AUTHENTICATING"
    ESTABLISHED = "ESTABLISHED"
    DEGRADED = "DEGRADED"
    CLOSING = "CLOSING"
    CLOSED = "CLOSED"
    QUARANTINED = "QUARANTINED"
    REVOKED = "REVOKED"


VALID_CHANNEL_TRANSITIONS: Dict[ChannelState, Set[ChannelState]] = {
    ChannelState.DISCONNECTED: {
        ChannelState.CONNECTING,
        ChannelState.CLOSED,
        ChannelState.QUARANTINED,
        ChannelState.REVOKED,
    },
    ChannelState.CONNECTING: {
        ChannelState.AUTHENTICATING,
        ChannelState.DISCONNECTED,
        ChannelState.CLOSED,
        ChannelState.QUARANTINED,
        ChannelState.REVOKED,
    },
    ChannelState.AUTHENTICATING: {
        ChannelState.ESTABLISHED,
        ChannelState.DISCONNECTED,
        ChannelState.CLOSED,
        ChannelState.QUARANTINED,
        ChannelState.REVOKED,
    },
    ChannelState.ESTABLISHED: {
        ChannelState.DEGRADED,
        ChannelState.CLOSING,
        ChannelState.CLOSED,
        ChannelState.QUARANTINED,
        ChannelState.REVOKED,
    },
    ChannelState.DEGRADED: {
        ChannelState.ESTABLISHED,
        ChannelState.CLOSING,
        ChannelState.CLOSED,
        ChannelState.QUARANTINED,
        ChannelState.REVOKED,
    },
    ChannelState.CLOSING: {
        ChannelState.CLOSED,
        ChannelState.QUARANTINED,
        ChannelState.REVOKED,
    },
    ChannelState.CLOSED: {
        ChannelState.CONNECTING,
        ChannelState.QUARANTINED,
        ChannelState.REVOKED,
    },
    ChannelState.QUARANTINED: {
        ChannelState.CLOSING,
        ChannelState.CLOSED,
        ChannelState.REVOKED,
    },
    ChannelState.REVOKED: set(),  # Absorbing terminal state
}


class FederationMessageType(str, Enum):
    """Taxonomy of typed messages carried over secure federation channels."""
    CHALLENGE = "CHALLENGE"
    CHALLENGE_RESPONSE = "CHALLENGE_RESPONSE"
    CAPABILITY_REQUEST = "CAPABILITY_REQUEST"
    CAPABILITY_RESPONSE = "CAPABILITY_RESPONSE"
    EVIDENCE_EXCHANGE = "EVIDENCE_EXCHANGE"
    HEARTBEAT = "HEARTBEAT"
    HEARTBEAT_ACK = "HEARTBEAT_ACK"
    REVOCATION = "REVOCATION"
    POLICY_SYNC = "POLICY_SYNC"
    STATE_SYNC = "STATE_SYNC"
    ERROR_RESPONSE = "ERROR_RESPONSE"
    RESOURCE_ADVERTISEMENT = "RESOURCE_ADVERTISEMENT"
    RESOURCE_QUERY = "RESOURCE_QUERY"
    RESOURCE_QUERY_RESPONSE = "RESOURCE_QUERY_RESPONSE"

    # Step 38 Distributed Task Orchestration Message Types
    TASK_ASSIGNMENT = "TASK_ASSIGNMENT"
    TASK_ASSIGNMENT_ACK = "TASK_ASSIGNMENT_ACK"
    TASK_STATUS_UPDATE = "TASK_STATUS_UPDATE"
    TASK_CHECKPOINT = "TASK_CHECKPOINT"
    TASK_CHECKPOINT_ACK = "TASK_CHECKPOINT_ACK"
    TASK_RESULT = "TASK_RESULT"
    TASK_RESULT_ACK = "TASK_RESULT_ACK"
    TASK_CANCEL = "TASK_CANCEL"
    TASK_CANCEL_ACK = "TASK_CANCEL_ACK"


@dataclass
class ChannelMetrics:
    """Diagnostic operational metrics for an active or closed channel."""
    bytes_sent: int = 0
    bytes_received: int = 0
    messages_sent: int = 0
    messages_received: int = 0
    heartbeats_sent: int = 0
    heartbeats_acknowledged: int = 0
    errors_count: int = 0
    last_activity_time: float = field(default_factory=time.time)
    connected_time: Optional[float] = None
    round_trip_latency_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bytes_sent": self.bytes_sent,
            "bytes_received": self.bytes_received,
            "messages_sent": self.messages_sent,
            "messages_received": self.messages_received,
            "heartbeats_sent": self.heartbeats_sent,
            "heartbeats_acknowledged": self.heartbeats_acknowledged,
            "errors_count": self.errors_count,
            "last_activity_time": self.last_activity_time,
            "connected_time": self.connected_time,
            "round_trip_latency_ms": self.round_trip_latency_ms,
        }


@dataclass
class ReconnectPolicy:
    """Exponential backoff configuration for channel reconnection."""
    initial_backoff_seconds: float = 0.05
    max_backoff_seconds: float = 2.0
    backoff_multiplier: float = 2.0
    max_attempts: int = 5
    jitter: bool = False

    def compute_backoff(self, attempt: int) -> float:
        """Compute bounded exponential backoff delay."""
        delay = self.initial_backoff_seconds * (self.backoff_multiplier ** (attempt - 1))
        return min(delay, self.max_backoff_seconds)


@dataclass
class FederationMessageEnvelope:
    """
    Authenticated, sequence-checked message envelope for inter-node communication.
    Builds upon and extends WireEnvelope with explicit engine identities,
    monotonic sequencing, and tenant isolation tags.
    """
    message_type: FederationMessageType
    message_id: str
    session_id: str
    sender_engine_id: str
    receiver_engine_id: str
    sender_peer_id: str
    receiver_peer_id: str
    sequence_number: int
    epoch: int
    payload: Dict[str, Any]
    protocol_version: str = FEDERATION_PROTOCOL_VERSION
    timestamp: float = field(default_factory=time.time)
    payload_digest: str = ""
    signature: str = ""
    tenant_id: Optional[str] = None

    def __post_init__(self) -> None:
        if isinstance(self.message_type, str):
            try:
                self.message_type = FederationMessageType(self.message_type)
            except ValueError as e:
                raise UnknownMessageTypeError(f"Unknown message type '{self.message_type}': {e}") from e
        if not self.payload_digest:
            self.payload_digest = self.compute_payload_digest()

    def compute_payload_digest(self) -> str:
        """Compute deterministic SHA-256 digest of canonical JSON payload."""
        try:
            canonical_json = json.dumps(self.payload, sort_keys=True, separators=(",", ":"))
        except TypeError as e:
            raise ProhibitedPayloadError(f"Payload contains non-serializable or non-primitive types: {e}") from e
        return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()

    def signing_bytes(self) -> bytes:
        """
        Produce canonical byte sequence over all envelope fields except signature.
        """
        digest = self.payload_digest or self.compute_payload_digest()
        doc = {
            "protocol_version": self.protocol_version,
            "message_type": self.message_type.value,
            "message_id": self.message_id,
            "session_id": self.session_id,
            "sender_engine_id": self.sender_engine_id,
            "receiver_engine_id": self.receiver_engine_id,
            "sender_peer_id": self.sender_peer_id,
            "receiver_peer_id": self.receiver_peer_id,
            "sequence_number": self.sequence_number,
            "epoch": self.epoch,
            "timestamp": self.timestamp,
            "payload_digest": digest,
            "tenant_id": self.tenant_id or "",
        }
        canonical_json = json.dumps(doc, sort_keys=True, separators=(",", ":"))
        return canonical_json.encode("utf-8")

    def sign(self, private_key: Ed25519PrivateKeyWrapper) -> "FederationMessageEnvelope":
        """Compute Ed25519 signature over envelope header + payload digest."""
        sig_hex = private_key.sign_hex(self.signing_bytes())
        self.signature = sig_hex
        return self

    def verify_signature(self, public_key: Ed25519PublicKeyWrapper) -> bool:
        """Verify Ed25519 signature against envelope signing bytes."""
        if not self.signature:
            return False
        try:
            sig_bytes = bytes.fromhex(self.signature)
        except ValueError:
            return False
        return public_key.verify(sig_bytes, self.signing_bytes())

    def verify_digest(self) -> bool:
        """Verify that payload_digest matches current payload."""
        expected = self.compute_payload_digest()
        return self.payload_digest == expected

    def validate(self, max_payload_bytes: int = DEFAULT_MAX_PAYLOAD_SIZE) -> None:
        """Validate envelope constraints, byte ceilings, and digest integrity."""
        if not self.protocol_version:
            raise EnvelopeIntegrityError("FederationMessageEnvelope missing protocol_version.")
        if not self.message_id:
            raise EnvelopeIntegrityError("FederationMessageEnvelope missing message_id.")
        if not self.sender_engine_id:
            raise EnvelopeIntegrityError("FederationMessageEnvelope missing sender_engine_id.")
        if not self.receiver_engine_id:
            raise EnvelopeIntegrityError("FederationMessageEnvelope missing receiver_engine_id.")
        if not isinstance(self.payload, dict):
            raise EnvelopeIntegrityError("FederationMessageEnvelope payload must be a dict.")

        raw_payload_len = len(json.dumps(self.payload))
        if raw_payload_len > max_payload_bytes:
            raise OversizedFrameError(
                f"Payload size {raw_payload_len} bytes exceeds maximum limit {max_payload_bytes} bytes."
            )

        if not self.verify_digest():
            raise EnvelopeIntegrityError("FederationMessageEnvelope payload_digest mismatch (tampered payload).")

    def to_dict(self) -> Dict[str, Any]:
        """Convert envelope to canonical dictionary."""
        return {
            "protocol_version": self.protocol_version,
            "message_type": self.message_type.value,
            "message_id": self.message_id,
            "session_id": self.session_id,
            "sender_engine_id": self.sender_engine_id,
            "receiver_engine_id": self.receiver_engine_id,
            "sender_peer_id": self.sender_peer_id,
            "receiver_peer_id": self.receiver_peer_id,
            "sequence_number": self.sequence_number,
            "epoch": self.epoch,
            "timestamp": self.timestamp,
            "payload": self.payload,
            "payload_digest": self.payload_digest,
            "signature": self.signature,
            "tenant_id": self.tenant_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FederationMessageEnvelope":
        """Instantiate envelope from dictionary with type validation."""
        d = dict(data)
        d["message_type"] = FederationMessageType(d["message_type"])
        return cls(**d)

    def to_wire_envelope(self) -> WireEnvelope:
        """Convert to legacy WireEnvelope for compatibility where needed."""
        mapped_type = MessageType.CAPABILITY_REQUEST
        if self.message_type == FederationMessageType.HEARTBEAT:
            mapped_type = MessageType.HEARTBEAT
        elif self.message_type == FederationMessageType.CHALLENGE:
            mapped_type = MessageType.CHALLENGE
        elif self.message_type == FederationMessageType.CHALLENGE_RESPONSE:
            mapped_type = MessageType.CHALLENGE_RESPONSE
        elif self.message_type == FederationMessageType.REVOCATION:
            mapped_type = MessageType.REVOCATION

        env = WireEnvelope(
            protocol_version=self.protocol_version,
            message_type=mapped_type,
            message_id=self.message_id,
            session_id=self.session_id,
            sender_peer_id=self.sender_peer_id,
            receiver_peer_id=self.receiver_peer_id,
            created_epoch=self.epoch,
            expires_epoch=self.epoch + 10,
            payload=self.payload,
            payload_digest=self.payload_digest,
            signature=self.signature,
        )
        return env

    @classmethod
    def from_wire_envelope(
        cls,
        wire: WireEnvelope,
        sender_engine_id: str,
        receiver_engine_id: str,
        sequence_number: int = 1,
        tenant_id: Optional[str] = None,
    ) -> "FederationMessageEnvelope":
        """Construct FederationMessageEnvelope from a WireEnvelope."""
        return cls(
            protocol_version=wire.protocol_version,
            message_type=FederationMessageType(wire.message_type.value),
            message_id=wire.message_id,
            session_id=wire.session_id,
            sender_engine_id=sender_engine_id,
            receiver_engine_id=receiver_engine_id,
            sender_peer_id=wire.sender_peer_id,
            receiver_peer_id=wire.receiver_peer_id,
            sequence_number=sequence_number,
            epoch=wire.created_epoch,
            payload=wire.payload,
            payload_digest=wire.payload_digest,
            signature=wire.signature,
            tenant_id=tenant_id,
        )
