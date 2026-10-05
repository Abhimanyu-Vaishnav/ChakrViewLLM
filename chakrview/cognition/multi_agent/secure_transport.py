"""
ChakrView Step 122: Secure Distributed Transport & Governance.

Builds tamper-proof network envelopes, HMAC message authentication,
payload checksum verification, replay protection, and GovernedToolGate trust boundaries:
- SecureEnvelope: HMAC-SHA256 signature, nonce, timestamp, payload hash
- DistributedSecurityPolicy: Nonce caching for replay defense, size limits
- NetworkToolGateBridge: Remote workers remain bound to local GovernedToolGate
"""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Set

from chakrview.cognition.multi_agent.transport import RequestEnvelope, ResponseEnvelope


DEFAULT_SHARED_FEDERATION_KEY = b"chakrview_secure_federation_key_v1"
MAX_PAYLOAD_SIZE_BYTES = 1024 * 1024  # 1 MB boundary
MAX_TIMESTAMP_DRIFT_SECONDS = 300.0   # 5 minutes replay window


@dataclass
class SecureEnvelope:
    """
    Authenticated network envelope wrapping task requests or responses.
    """
    inner_payload: Dict[str, Any]
    sender_node_id: str
    recipient_node_id: str
    nonce: str
    timestamp: float = field(default_factory=time.time)
    signature: str = ""

    def sign(self, secret_key: bytes = DEFAULT_SHARED_FEDERATION_KEY) -> None:
        raw = f"{self.sender_node_id}:{self.recipient_node_id}:{self.nonce}:{self.timestamp}:{json.dumps(self.inner_payload, sort_keys=True)}"
        self.signature = hmac.new(secret_key, raw.encode("utf-8"), hashlib.sha256).hexdigest()

    def verify(self, secret_key: bytes = DEFAULT_SHARED_FEDERATION_KEY) -> bool:
        raw = f"{self.sender_node_id}:{self.recipient_node_id}:{self.nonce}:{self.timestamp}:{json.dumps(self.inner_payload, sort_keys=True)}"
        expected = hmac.new(secret_key, raw.encode("utf-8"), hashlib.sha256).hexdigest()
        return hmac.compare_digest(self.signature, expected)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SecureEnvelope:
        return cls(**data)


class DistributedSecurityValidator:
    """
    Enforces replay protection, payload size boundaries, and authentication.
    """

    def __init__(self, secret_key: bytes = DEFAULT_SHARED_FEDERATION_KEY) -> None:
        self.secret_key = secret_key
        self._seen_nonces: Set[str] = set()

    def validate_incoming_envelope(self, envelope: SecureEnvelope) -> None:
        # Check signature
        if not envelope.verify(self.secret_key):
            raise PermissionError("Security violation: Invalid envelope signature / HMAC mismatch")

        # Check timestamp drift
        drift = abs(time.time() - envelope.timestamp)
        if drift > MAX_TIMESTAMP_DRIFT_SECONDS:
            raise ValueError(f"Security violation: Envelope timestamp drift too large ({drift:.1f}s)")

        # Check replay nonce
        if envelope.nonce in self._seen_nonces:
            raise ValueError(f"Security violation: Replay detected for nonce {envelope.nonce}")
        self._seen_nonces.add(envelope.nonce)

        # Check size limits
        serialized_size = len(json.dumps(envelope.inner_payload).encode("utf-8"))
        if serialized_size > MAX_PAYLOAD_SIZE_BYTES:
            raise ValueError(f"Security violation: Payload exceeds size ceiling ({serialized_size} > {MAX_PAYLOAD_SIZE_BYTES})")
