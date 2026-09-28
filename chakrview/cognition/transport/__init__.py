"""
Secure Physical Transport & Wire Protocol Subsystem (Step 30).

CRITICAL ARCHITECTURAL AXIOMS:
1. TRANSPORT != AUTHORITY:
   Transports convey WireEnvelope containers; they never authorize actions or bypass policies.
2. TRANSPORT != TRUST:
   A reachable or connected transport endpoint confers zero trust.
3. CRYPTOGRAPHIC INTEGRITY & FAIL-CLOSED SECURITY:
   Envelopes are signed, digest-verified, framed with length prefixes, and protected against replay.
"""

from chakrview.cognition.transport.errors import (
    TransportError,
    TransportTimeoutError,
    TransportUnavailableError,
    TransportProtocolError,
    FrameError,
    OversizedPayloadError,
    ReplayAttackError,
    WireSecurityError,
)
from chakrview.cognition.transport.models import (
    MessageType,
    TransportStatus,
    TransportHealth,
    WireEnvelope,
    DEFAULT_MAX_PAYLOAD_BYTES,
    MAX_WIRE_FRAME_BYTES,
)
from chakrview.cognition.transport.base import Transport
from chakrview.cognition.transport.framing import LengthPrefixedFramer
from chakrview.cognition.transport.serialization import DeterministicWireSerializer
from chakrview.cognition.transport.loopback import LoopbackWireTransport
from chakrview.cognition.transport.tcp import TCPWireTransport
from chakrview.cognition.transport.http2 import HTTP2WireTransport, is_http2_available
from chakrview.cognition.transport.grpc import GRPCWireTransport, is_grpc_available
from chakrview.cognition.transport.registry import TransportRegistry

__all__ = [
    "TransportError",
    "TransportTimeoutError",
    "TransportUnavailableError",
    "TransportProtocolError",
    "FrameError",
    "OversizedPayloadError",
    "ReplayAttackError",
    "WireSecurityError",
    "MessageType",
    "TransportStatus",
    "TransportHealth",
    "WireEnvelope",
    "DEFAULT_MAX_PAYLOAD_BYTES",
    "MAX_WIRE_FRAME_BYTES",
    "Transport",
    "LengthPrefixedFramer",
    "DeterministicWireSerializer",
    "LoopbackWireTransport",
    "TCPWireTransport",
    "HTTP2WireTransport",
    "is_http2_available",
    "GRPCWireTransport",
    "is_grpc_available",
    "TransportRegistry",
]
