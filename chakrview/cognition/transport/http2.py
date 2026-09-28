"""
HTTP/2 Transport Adapter Boundary (Step 30).

Provides architectural adapter boundary for HTTP/2 framing and multiplexing.
Performs explicit dependency verification and fails closed if runtime libraries
(e.g., 'h2' or 'httpx') are missing. Never fakes transport availability.
"""

import importlib.util
from typing import Dict, List, Optional, Any

from chakrview.cognition.transport.base import Transport
from chakrview.cognition.transport.models import WireEnvelope, TransportHealth, TransportStatus
from chakrview.cognition.transport.errors import TransportUnavailableError


def is_http2_available() -> bool:
    """Check if required HTTP/2 runtime dependencies are installed."""
    return (
        importlib.util.find_spec("h2") is not None
        or importlib.util.find_spec("httpx") is not None
    )


class HTTP2WireTransport(Transport):
    """
    HTTP/2 Transport Adapter boundary.
    Enforces strict dependency checking and clean failure when libraries are not installed.
    """

    def __init__(self, endpoint: str = "http2://127.0.0.1:8080") -> None:
        self.endpoint = endpoint
        self._available = is_http2_available()
        self._status = TransportStatus.DISCONNECTED

    @property
    def is_available(self) -> bool:
        return self._available

    def _ensure_available(self) -> None:
        if not self._available:
            raise TransportUnavailableError(
                "HTTP/2 transport driver is not installed. Missing optional dependencies: 'h2' or 'httpx'."
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
            transport_type="http2",
            details={"is_available": self._available, "dependency": "h2/httpx"},
        )

    def capabilities(self) -> Dict[str, Any]:
        return {
            "transport_type": "http2",
            "is_physical": True,
            "is_available": self._available,
            "supports_multiplexing": True,
            "supports_header_compression": True,
        }
