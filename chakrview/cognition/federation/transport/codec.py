"""
Canonical Deterministic Codec for Federation Message Envelopes (Step 36).

CRITICAL ARCHITECTURAL AXIOMS:
1. DETERMINISTIC CANONICAL ENCODING:
   All envelopes serialize strictly to UTF-8 encoded canonical JSON with
   alphabetically sorted keys and compact separators (',', ':').
2. PROHIBITED DATA EXCLUSION:
   Private keys, session secrets, model weights (tensors), memory addresses,
   and arbitrary executable Python objects (pickle, eval) are strictly rejected.
3. FAIL-CLOSED INTEGRITY VERIFICATION:
   Envelopes with invalid schemas, corrupt JSON, unknown message types, or
   mismatched payload digests are rejected with explicit typed errors.
"""

import json
from typing import Dict, Any, Set

from chakrview.cognition.federation.transport.models import (
    FederationMessageEnvelope,
    FederationMessageType,
    DEFAULT_MAX_FRAME_SIZE,
    DEFAULT_MAX_PAYLOAD_SIZE,
)
from chakrview.cognition.federation.transport.errors import (
    CodecError,
    ProhibitedPayloadError,
    UnknownMessageTypeError,
    EnvelopeIntegrityError,
    OversizedFrameError,
)

# Prohibited keys in payload dictionaries
PROHIBITED_KEY_PATTERNS = {
    "private_key",
    "private_bytes",
    "secret_key",
    "session_secret",
    "shared_secret",
    "model_weights",
    "weight_tensor",
    "tensor_data",
    "state_dict",
    "pickle_data",
    "__reduce__",
    "__class__",
    "raw_pointer",
}


def _assert_no_prohibited_content(obj: Any, path: str = "payload") -> None:
    """Recursively scan payload data structure for forbidden keys or non-primitive types."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if not isinstance(k, str):
                raise ProhibitedPayloadError(f"Dictionary keys must be strings, got {type(k)} at '{path}'.")
            lower_k = k.lower()
            for pattern in PROHIBITED_KEY_PATTERNS:
                if pattern in lower_k:
                    raise ProhibitedPayloadError(
                        f"Prohibited security/neural keyword '{k}' detected at '{path}'. "
                        "Private keys, secrets, and model weights cannot be transported over federation channels."
                    )
            _assert_no_prohibited_content(v, f"{path}.{k}")
    elif isinstance(obj, (list, tuple)):
        for idx, item in enumerate(obj):
            _assert_no_prohibited_content(item, f"{path}[{idx}]")
    elif obj is None or isinstance(obj, (bool, int, float, str)):
        return
    else:
        # Rejects PyTorch tensors, objects, arbitrary classes, functions, bytes
        raise ProhibitedPayloadError(
            f"Prohibited non-primitive data type {type(obj).__name__} at '{path}'. "
            "Arbitrary Python objects, tensors, and binary blobs cannot be serialized in federation payloads."
        )


class FederationMessageCodec:
    """
    Canonical deterministic serializer and deserializer for FederationMessageEnvelope.
    """

    @staticmethod
    def serialize(envelope: FederationMessageEnvelope) -> bytes:
        """
        Serialize a FederationMessageEnvelope to deterministic canonical JSON bytes.
        
        Args:
            envelope: FederationMessageEnvelope instance to serialize.
            
        Returns:
            Canonical UTF-8 encoded bytes.
        """
        envelope.validate()

        # Enforce strict prohibition of secrets, weights, and non-primitives
        _assert_no_prohibited_content(envelope.payload)

        data = envelope.to_dict()
        canonical_str = json.dumps(data, sort_keys=True, separators=(",", ":"))
        encoded = canonical_str.encode("utf-8")

        if len(encoded) > DEFAULT_MAX_FRAME_SIZE:
            raise OversizedFrameError(
                f"Serialized envelope {len(encoded)} bytes exceeds maximum frame ceiling {DEFAULT_MAX_FRAME_SIZE} bytes."
            )

        return encoded

    @staticmethod
    def deserialize(
        raw_bytes: bytes,
        max_bytes: int = DEFAULT_MAX_FRAME_SIZE,
    ) -> FederationMessageEnvelope:
        """
        Parse and validate a FederationMessageEnvelope from raw wire bytes.
        
        Args:
            raw_bytes: Raw wire bytes to deserialize.
            max_bytes: Maximum allowed byte length ceiling.
            
        Returns:
            Validated FederationMessageEnvelope instance.
        """
        if not raw_bytes:
            raise CodecError("Cannot deserialize empty wire payload.")

        if len(raw_bytes) > max_bytes:
            raise OversizedFrameError(
                f"Incoming payload {len(raw_bytes)} bytes exceeds ceiling {max_bytes} bytes."
            )

        try:
            decoded_str = raw_bytes.decode("utf-8")
            data = json.loads(decoded_str)
        except UnicodeDecodeError as e:
            raise CodecError(f"UTF-8 decoding failed for federation envelope: {e}") from e
        except json.JSONDecodeError as e:
            raise CodecError(f"Malformed JSON in federation envelope: {e}") from e

        if not isinstance(data, dict):
            raise CodecError("Federation envelope root element must be a JSON object.")

        required_keys = {
            "protocol_version",
            "message_type",
            "message_id",
            "session_id",
            "sender_engine_id",
            "receiver_engine_id",
            "sender_peer_id",
            "receiver_peer_id",
            "sequence_number",
            "epoch",
            "payload",
            "payload_digest",
        }
        missing = required_keys - set(data.keys())
        if missing:
            raise EnvelopeIntegrityError(f"Federation envelope missing required keys: {sorted(missing)}")

        # Validate message type
        raw_type = data.get("message_type")
        try:
            msg_type = FederationMessageType(raw_type)
        except ValueError as e:
            raise UnknownMessageTypeError(f"Unknown federation message type '{raw_type}': {e}") from e

        # Validate payload content
        payload = data.get("payload")
        if not isinstance(payload, dict):
            raise CodecError("Envelope payload must be a JSON object.")
        _assert_no_prohibited_content(payload)

        try:
            envelope = FederationMessageEnvelope.from_dict(data)
            envelope.validate()
            return envelope
        except Exception as e:
            if isinstance(e, (CodecError, EnvelopeIntegrityError, OversizedFrameError, UnknownMessageTypeError)):
                raise
            raise EnvelopeIntegrityError(f"Failed to instantiate FederationMessageEnvelope: {e}") from e
