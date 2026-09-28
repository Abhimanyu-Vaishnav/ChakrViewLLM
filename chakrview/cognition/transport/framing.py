"""
Length-Prefixed Wire Framing for Stream Transports (Step 30).

Frame Layout:
[4-byte unsigned big-endian integer length (N)][N bytes of payload data]
"""

import struct
from typing import Optional, Tuple

from chakrview.cognition.transport.models import MAX_WIRE_FRAME_BYTES
from chakrview.cognition.transport.errors import FrameError, OversizedPayloadError

HEADER_FORMAT = ">I"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)  # 4 bytes


class LengthPrefixedFramer:
    """
    Encodes and decodes length-prefixed frames for stream transports (TCP).
    """

    @staticmethod
    def encode_frame(payload: bytes, max_frame_size: int = MAX_WIRE_FRAME_BYTES) -> bytes:
        """
        Encode payload with 4-byte big-endian length prefix.
        """
        payload_len = len(payload)
        if payload_len == 0:
            raise FrameError("Cannot frame empty payload.")
        if payload_len > max_frame_size:
            raise OversizedPayloadError(
                f"Payload size {payload_len} exceeds maximum frame limit {max_frame_size}."
            )
        header = struct.pack(HEADER_FORMAT, payload_len)
        return header + payload

    @staticmethod
    def decode_frame(buffer: bytearray, max_frame_size: int = MAX_WIRE_FRAME_BYTES) -> Optional[bytes]:
        """
        Extract complete frame from buffer if available.
        Mutates buffer by removing consumed bytes.
        Returns payload bytes if complete frame found, None if partial/incomplete.
        """
        if len(buffer) < HEADER_SIZE:
            return None

        # Peek frame length
        (payload_len,) = struct.unpack_from(HEADER_FORMAT, buffer, 0)

        if payload_len == 0:
            raise FrameError("Encountered zero-length frame header.")
        if payload_len > max_frame_size:
            raise OversizedPayloadError(
                f"Frame header indicates size {payload_len}, which exceeds maximum ceiling {max_frame_size}."
            )

        total_frame_len = HEADER_SIZE + payload_len
        if len(buffer) < total_frame_len:
            # Need more data from socket
            return None

        # Extract full payload
        payload = bytes(buffer[HEADER_SIZE:total_frame_len])
        del buffer[:total_frame_len]
        return payload
