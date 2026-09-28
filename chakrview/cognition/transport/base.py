"""
Abstract Base Transport Interface (Step 30).

CRITICAL ARCHITECTURAL AXIOMS:
1. TRANSPORT != AUTHORITY:
   Transports convey WireEnvelope containers; they never authorize actions or bypass policies.
2. TRANSPORT != TRUST:
   A reachable or connected transport endpoint confers zero trust.
3. MODULAR PLUGGABILITY:
   The cross-zone federation engine depends solely on this abstraction, enabling
   seamless interchange of Loopback, TCP, HTTP/2, and gRPC adapters.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any

from chakrview.cognition.transport.models import WireEnvelope, TransportHealth, TransportStatus


class Transport(ABC):
    """
    Abstract interface for physical and logical cross-zone message transport.
    """

    @abstractmethod
    def connect(self, endpoint: str, timeout_seconds: float = 5.0) -> bool:
        """
        Establish connection to a remote transport endpoint.
        """
        pass

    @abstractmethod
    def listen(self, endpoint: str) -> None:
        """
        Bind and start listening for incoming connections on endpoint.
        """
        pass

    @abstractmethod
    def accept(self, timeout_seconds: float = 5.0) -> Optional[Any]:
        """
        Accept an incoming connection (server mode). Returns connection context.
        """
        pass

    @abstractmethod
    def send(self, envelope: WireEnvelope, timeout_seconds: float = 5.0) -> bool:
        """
        Send a validated WireEnvelope over the transport.
        """
        pass

    @abstractmethod
    def receive(self, timeout_seconds: float = 5.0) -> Optional[WireEnvelope]:
        """
        Receive the next incoming WireEnvelope from the transport.
        Returns None if timed out without data.
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """
        Gracefully release transport sockets and resources.
        """
        pass

    @abstractmethod
    def health(self) -> TransportHealth:
        """
        Query current operational health and telemetry.
        """
        pass

    @abstractmethod
    def capabilities(self) -> Dict[str, Any]:
        """
        Report supported features of this transport adapter.
        """
        pass
