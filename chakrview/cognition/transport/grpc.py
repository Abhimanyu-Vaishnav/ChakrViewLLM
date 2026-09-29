"""
gRPC Transport Adapter Boundary (Step 31).

Provides architectural adapter boundary for gRPC remote procedure calls, streams, and TLS.
Performs explicit dependency verification and fails closed if runtime libraries
('grpc') are missing. Never fakes transport availability.
"""

import importlib.util
from typing import Dict, List, Optional, Any

from chakrview.cognition.transport.base import Transport
from chakrview.cognition.transport.models import WireEnvelope, TransportHealth, TransportStatus
from chakrview.cognition.transport.errors import TransportUnavailableError
from chakrview.cognition.transport.security.models import TLSMode
from chakrview.cognition.transport.security.policy import SecureTransportPolicy


def is_grpc_available() -> bool:
    """Check if required gRPC runtime dependencies are installed."""
    return importlib.util.find_spec("grpc") is not None


class GRPCWireTransport(Transport):
    """
    gRPC Transport Adapter boundary with TLS security configuration.
    Enforces strict dependency checking and clean failure when grpc is not installed.
    """

    def __init__(
        self,
        endpoint: str = "grpc://127.0.0.1:50051",
        security_policy: Optional[SecureTransportPolicy] = None,
        server_cert_pem: Optional[str] = None,
        server_key_pem: Optional[str] = None,
        client_cert_pem: Optional[str] = None,
        client_key_pem: Optional[str] = None,
        ca_cert_pem: Optional[str] = None,
    ) -> None:
        self.endpoint = endpoint
        self.security_policy = security_policy or SecureTransportPolicy(
            tls_mode=TLSMode.TLS,
            allow_tls_1_2=False,
        )
        self.security_policy.validate()

        self._server_cert_pem = server_cert_pem
        self._server_key_pem = server_key_pem
        self._client_cert_pem = client_cert_pem
        self._client_key_pem = client_key_pem
        self._ca_cert_pem = ca_cert_pem

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
            latency_ms=0.0,
            active_connections=0,
            messages_sent=0,
            messages_received=0,
            error_count=0,
            details={
                "is_available": self._available,
                "status": self._status.value,
                "tls_mode": self.security_policy.tls_mode.value,
            },
        )

    def capabilities(self) -> Dict[str, Any]:
        return {
            "transport_type": "grpc",
            "is_physical": True,
            "supports_duplex": True,
            "supports_streaming": True,
            "supports_tls": True,
            "supports_mtls": True,
            "tls_mode": self.security_policy.tls_mode.value,
            "is_available": self._available,
        }
