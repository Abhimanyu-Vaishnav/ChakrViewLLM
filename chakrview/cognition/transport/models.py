"""
Strongly Typed Transport and Wire Envelope Models (Step 30).

CRITICAL ARCHITECTURAL AXIOMS:
1. TRANSPORT != AUTHORITY:
   The transport layer moves validated envelopes; it never makes cognitive
   decisions or authorizes capabilities.
2. ENVELOPE DIGEST & SIGNATURE INTEGRITY:
   Every envelope carries a SHA-256 payload digest and an Ed25519 detached signature
   over header + digest before trusted processing.
3. BOUNDED PAYLOAD SIZE:
   Hard payload ceilings strictly enforced to prevent memory exhaustion.
"""

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import secrets
import time
from typing import Dict, List, Optional, Any, Tuple

from chakrview.cognition.peering.crypto import (
    Ed25519PrivateKeyWrapper,
    Ed25519PublicKeyWrapper,
)
from chakrview.cognition.transport.errors import (
    TransportProtocolError,
    OversizedPayloadError,
)

# Hard wire limits
DEFAULT_MAX_PAYLOAD_BYTES = 65536      # 64 KB default
MAX_WIRE_FRAME_BYTES = 1048576         # 1 MB absolute ceiling


class MessageType(str, Enum):
    """Taxonomy of typed messages transported over cross-zone federation wire."""
    CHALLENGE = "CHALLENGE"
    CHALLENGE_RESPONSE = "CHALLENGE_RESPONSE"
    CAPABILITY_REQUEST = "CAPABILITY_REQUEST"
    CAPABILITY_RESPONSE = "CAPABILITY_RESPONSE"
    EVIDENCE_EXCHANGE = "EVIDENCE_EXCHANGE"
    HEARTBEAT = "HEARTBEAT"
    REVOCATION = "REVOCATION"
    POLICY_SYNC = "POLICY_SYNC"
    ERROR_RESPONSE = "ERROR_RESPONSE"


class TransportStatus(str, Enum):
    """Operational status of a physical or logical transport endpoint."""
    DISCONNECTED = "DISCONNECTED"
    CONNECTING = "CONNECTING"
    CONNECTED = "CONNECTED"
    LISTENING = "LISTENING"
    ERROR = "ERROR"


@dataclass
class TransportHealth:
    """Diagnostic health snapshot of a transport adapter."""
    is_healthy: bool
    endpoint: str
    transport_type: str
    latency_ms: float = 0.0
    active_connections: int = 0
    messages_sent: int = 0
    messages_received: int = 0
    error_count: int = 0
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_healthy": self.is_healthy,
            "endpoint": self.endpoint,
            "transport_type": self.transport_type,
            "latency_ms": self.latency_ms,
            "active_connections": self.active_connections,
            "messages_sent": self.messages_sent,
            "messages_received": self.messages_received,
            "error_count": self.error_count,
            "details": dict(self.details),
        }


@dataclass
class WireEnvelope:
    """
    Standard secure envelope for cross-zone message transport.
    """
    protocol_version: str
    message_type: MessageType
    message_id: str
    session_id: str
    sender_peer_id: str
    receiver_peer_id: str
    created_epoch: int
    expires_epoch: int
    payload: Dict[str, Any]
    payload_digest: str = ""
    signature: str = ""

    def __post_init__(self) -> None:
        if isinstance(self.message_type, str):
            self.message_type = MessageType(self.message_type)
        if not self.payload_digest:
            self.payload_digest = self.compute_payload_digest()

    def compute_payload_digest(self) -> str:
        """Compute deterministic SHA-256 digest of canonical JSON payload."""
        canonical_json = json.dumps(self.payload, sort_keys=True, separators=(",", ":"))
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
            "sender_peer_id": self.sender_peer_id,
            "receiver_peer_id": self.receiver_peer_id,
            "created_epoch": self.created_epoch,
            "expires_epoch": self.expires_epoch,
            "payload_digest": digest,
        }
        canonical_json = json.dumps(doc, sort_keys=True, separators=(",", ":"))
        return canonical_json.encode("utf-8")

    def sign(self, private_key: Ed25519PrivateKeyWrapper) -> "WireEnvelope":
        """Compute Ed25519 signature over envelope header + payload digest."""
        sig_hex = private_key.sign_hex(self.signing_bytes())
        self.signature = sig_hex
        return self

    def verify_signature(self, public_key: Ed25519PublicKeyWrapper) -> bool:
        """
        Mathematically verify Ed25519 signature against signing bytes.
        """
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

    def validate(self, max_bytes: int = DEFAULT_MAX_PAYLOAD_BYTES) -> None:
        """
        Validate schema constraints, byte ceilings, and digest integrity.
        """
        if not self.protocol_version or not isinstance(self.protocol_version, str):
            raise TransportProtocolError("WireEnvelope missing valid protocol_version.")
        if not self.message_id or not isinstance(self.message_id, str):
            raise TransportProtocolError("WireEnvelope missing valid message_id.")
        if not self.session_id or not isinstance(self.session_id, str):
            raise TransportProtocolError("WireEnvelope missing valid session_id.")
        if not self.sender_peer_id or not isinstance(self.sender_peer_id, str):
            raise TransportProtocolError("WireEnvelope missing valid sender_peer_id.")
        if not self.receiver_peer_id or not isinstance(self.receiver_peer_id, str):
            raise TransportProtocolError("WireEnvelope missing valid receiver_peer_id.")
        if not isinstance(self.payload, dict):
            raise TransportProtocolError("WireEnvelope payload must be a dict.")

        # Check payload size ceiling
        raw_payload_len = len(json.dumps(self.payload))
        if raw_payload_len > max_bytes:
            raise OversizedPayloadError(
                f"Payload size {raw_payload_len} bytes exceeds maximum limit {max_bytes} bytes."
            )

        if not self.verify_digest():
            raise TransportProtocolError("WireEnvelope payload_digest mismatch (tampered payload).")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "protocol_version": self.protocol_version,
            "message_type": self.message_type.value,
            "message_id": self.message_id,
            "session_id": self.session_id,
            "sender_peer_id": self.sender_peer_id,
            "receiver_peer_id": self.receiver_peer_id,
            "created_epoch": self.created_epoch,
            "expires_epoch": self.expires_epoch,
            "payload": self.payload,
            "payload_digest": self.payload_digest,
            "signature": self.signature,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WireEnvelope":
        d = dict(data)
        d["message_type"] = MessageType(d["message_type"])
        return cls(**d)
