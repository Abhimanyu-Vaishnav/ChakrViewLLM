"""
Deterministic Length-Prefixed Binary Framing for Federation Message Transport (Step 36).

Frame Layout:
[4-byte unsigned big-endian integer length (N)][N bytes of canonical payload data]

Guarantees:
- Bounded memory allocation: Oversized frame headers are rejected before allocating payload memory.
- Malformed header detection: Zero-length or negative lengths fail closed.
- Truncation detection: Incomplete payloads yield None without consuming bytes prematurely.
- Independence: Pure stream framing completely decoupled from authorization or cryptography.
"""

import struct
from typing import Optional, List, Tuple

from chakrview.cognition.federation.transport.models import DEFAULT_MAX_FRAME_SIZE
from chakrview.cognition.federation.transport.errors import (
    OversizedFrameError,
    MalformedFrameError,
    TruncatedFrameError,
)

HEADER_FORMAT = ">I"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)  # 4 bytes


class FederationMessageFramer:
    """
    Encodes and decodes bounded length-prefixed frames for federation transport streams.
    """

    @staticmethod
    def encode_frame(payload: bytes, max_frame_size: int = DEFAULT_MAX_FRAME_SIZE) -> bytes:
        """
        Encode raw payload bytes into a length-prefixed binary frame.
        
        Args:
            payload: Binary payload bytes to frame.
            max_frame_size: Maximum allowable frame payload length.
            
        Returns:
            Prefixed bytes ready for wire transmission.
        """
        if not payload:
            raise MalformedFrameError("Cannot encode empty frame payload.")

        payload_len = len(payload)
        if payload_len > max_frame_size:
            raise OversizedFrameError(
                f"Payload size {payload_len} bytes exceeds maximum allowed frame limit {max_frame_size} bytes."
            )

        header = struct.pack(HEADER_FORMAT, payload_len)
        return header + payload

    @staticmethod
    def decode_frame(
        buffer: bytearray,
        max_frame_size: int = DEFAULT_MAX_FRAME_SIZE,
    ) -> Optional[bytes]:
        """
        Extract a single complete frame from a stream buffer if available.
        Mutates the buffer by removing consumed frame bytes.
        
        Args:
            buffer: Mutable stream byte buffer.
            max_frame_size: Maximum allowable payload size ceiling.
            
        Returns:
            Extracted frame payload bytes if a complete frame is present, or None if partial.
        """
        if len(buffer) < HEADER_SIZE:
            return None

        # Peek frame length without consuming
        (payload_len,) = struct.unpack_from(HEADER_FORMAT, buffer, 0)

        if payload_len == 0:
            raise MalformedFrameError("Encountered invalid zero-length frame header.")

        # Enforce ceiling check BEFORE any payload allocation
        if payload_len > max_frame_size:
            raise OversizedFrameError(
                f"Frame header specifies {payload_len} bytes, exceeding maximum ceiling {max_frame_size} bytes."
            )

        total_frame_len = HEADER_SIZE + payload_len
        if len(buffer) < total_frame_len:
            # Frame payload is incomplete; wait for additional stream bytes
            return None

        # Extract complete frame and remove from buffer
        payload = bytes(buffer[HEADER_SIZE:total_frame_len])
        del buffer[:total_frame_len]
        return payload

    @staticmethod
    def decode_all_frames(
        raw_stream: bytes,
        max_frame_size: int = DEFAULT_MAX_FRAME_SIZE,
        strict_eof: bool = True,
    ) -> List[bytes]:
        """
        Decode all complete frames from a static byte sequence.
        
        Args:
            raw_stream: Raw bytes containing one or more concatenated frames.
            max_frame_size: Maximum allowable payload size ceiling.
            strict_eof: If True, raises TruncatedFrameError if trailing incomplete bytes remain.
            
        Returns:
            List of decoded payload byte sequences.
        """
        buf = bytearray(raw_stream)
        frames: List[bytes] = []

        while len(buf) >= HEADER_SIZE:
            frame = FederationMessageFramer.decode_frame(buf, max_frame_size=max_frame_size)
            if frame is None:
                break
            frames.append(frame)

        if strict_eof and len(buf) > 0:
            raise TruncatedFrameError(
                f"Stream ended with {len(buf)} incomplete trailing bytes."
            )

        return frames
