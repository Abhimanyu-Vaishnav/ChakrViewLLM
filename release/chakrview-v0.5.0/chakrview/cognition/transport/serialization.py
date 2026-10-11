"""
Deterministic Wire Serializer and Deserializer (Step 30).

CRITICAL ARCHITECTURAL AXIOMS:
1. DETERMINISTIC CANONICAL ENCODING:
   Serializes strictly to UTF-8 encoded canonical JSON with sorted keys and compact separators.
2. NO UNSAFE DESERIALIZATION:
   Arbitrary object deserialization (such as pickle or yaml) is strictly forbidden.
   Deserializes solely into validated primitive dict/list types.
3. FAIL-CLOSED VALIDATION:
   Oversized payloads, schema violations, or malformed JSON raise immediate typed errors.
"""

import json
from typing import Dict, Any

from chakrview.cognition.transport.models import WireEnvelope, DEFAULT_MAX_PAYLOAD_BYTES, MAX_WIRE_FRAME_BYTES
from chakrview.cognition.transport.errors import (
    TransportProtocolError,
    OversizedPayloadError,
)


class DeterministicWireSerializer:
    """
    Encodes and decodes WireEnvelope instances to and from canonical byte sequences.
    """

    @staticmethod
    def serialize(envelope: WireEnvelope) -> bytes:
        """
        Serialize a WireEnvelope to deterministic canonical JSON bytes.
        """
        envelope.validate()
        data_dict = envelope.to_dict()
        canonical_str = json.dumps(data_dict, sort_keys=True, separators=(",", ":"))
        encoded = canonical_str.encode("utf-8")
        if len(encoded) > MAX_WIRE_FRAME_BYTES:
            raise OversizedPayloadError(
                f"Serialized wire envelope {len(encoded)} bytes exceeds hard ceiling {MAX_WIRE_FRAME_BYTES} bytes."
            )
        return encoded

    @staticmethod
    def deserialize(raw_bytes: bytes, max_bytes: int = MAX_WIRE_FRAME_BYTES) -> WireEnvelope:
        """
        Parse and validate a WireEnvelope from raw wire bytes.
        """
        if not raw_bytes:
            raise TransportProtocolError("Cannot deserialize empty wire payload.")
        if len(raw_bytes) > max_bytes:
            raise OversizedPayloadError(
                f"Incoming wire payload {len(raw_bytes)} bytes exceeds limit {max_bytes} bytes."
            )

        try:
            decoded_str = raw_bytes.decode("utf-8")
            data = json.loads(decoded_str)
        except UnicodeDecodeError as e:
            raise TransportProtocolError(f"UTF-8 decoding failed for wire envelope: {e}")
        except json.JSONDecodeError as e:
            raise TransportProtocolError(f"Malformed JSON in wire envelope: {e}")

        if not isinstance(data, dict):
            raise TransportProtocolError("Wire payload root element must be a JSON object.")

        required_keys = {
            "protocol_version", "message_type", "message_id", "session_id",
            "sender_peer_id", "receiver_peer_id", "created_epoch",
            "expires_epoch", "payload", "payload_digest", "signature"
        }
        missing = required_keys - set(data.keys())
        if missing:
            raise TransportProtocolError(f"Wire envelope missing required keys: {sorted(missing)}")

        try:
            envelope = WireEnvelope.from_dict(data)
            envelope.validate()
            return envelope
        except Exception as e:
            if isinstance(e, TransportProtocolError):
                raise
            raise TransportProtocolError(f"Failed to instantiate WireEnvelope: {e}")
