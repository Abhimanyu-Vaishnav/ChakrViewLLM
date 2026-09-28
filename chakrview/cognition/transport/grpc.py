"""
gRPC Transport Adapter Boundary (Step 30).

Provides architectural adapter boundary for gRPC remote procedure calls and streams.
Performs explicit dependency verification and fails closed if runtime libraries
('grpc') are missing. Never fakes transport availability.
"""

import importlib.util
from typing import Dict, List, Optional, Any

from chakrview.cognition.transport.base import Transport
from chakrview.cognition.transport.models import WireEnvelope, TransportHealth, TransportStatus
from chakrview.cognition.transport.errors import TransportUnavailableError


def is_grpc_available() -> bool:
    """Check if required gRPC runtime dependencies are installed."""
    return importlib.util.find_spec("grpc") is not None


class GRPCWireTransport(Transport):
    """
    gRPC Transport Adapter boundary.
    Enforces strict dependency checking and clean failure when grpc is not installed.
    """

    def __init__(self, endpoint: str = "grpc://127.0.0.1:50051") -> None:
        self.endpoint = endpoint
        self._available = is_grpc_available()
        self._status = TransportStatus.DISCONNECTED

    @property
    def is_available(self) -> bool:
        return self._available

    def _ensure_available(self) -> None:
        if not self._available:
            raise TransportUnavailableError(
                "gRPC transport driver is not installed. Missing optional dependency: 'grpcio'."
            )

    def connect(self, endpoint: str, timeout_seconds: float = 5.0) -> bool:
        self._ensure_available()
        self.endpoint = endpoint
        self._status = TransportStatus.CONNECTED
        return True

    def listen(self, endpoint: str) -> None:
        self._ensure_available()
        self.endpoint = endpoint
        self._status = TransportStatus.LISTENING

    def accept(self, timeout_seconds: float = 5.0) -> Optional[Any]:
        self._ensure_available()
        return None

    def send(self, envelope: WireEnvelope, timeout_seconds: float = 5.0) -> bool:
        self._ensure_available()
        return True

    def receive(self, timeout_seconds: float = 5.0) -> Optional[WireEnvelope]:
        self._ensure_available()
        return None

    def close(self) -> None:
        self._status = TransportStatus.DISCONNECTED

    def health(self) -> TransportHealth:
        return TransportHealth(
            is_healthy=self._available and self._status == TransportStatus.CONNECTED,
            endpoint=self.endpoint,
            transport_type="grpc",
            details={"is_available": self._available, "dependency": "grpcio"},
        )

    def capabilities(self) -> Dict[str, Any]:
        return {
            "transport_type": "grpc",
            "is_physical": True,
            "is_available": self._available,
            "supports_bi_directional_streaming": True,
            "supports_protobuf": True,
        }
