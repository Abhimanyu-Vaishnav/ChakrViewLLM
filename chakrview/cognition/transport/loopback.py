"""
Deterministic In-Process Loopback Transport for Cross-Zone Federation (Step 30).

Provides thread-safe in-memory queues simulating wire transports on CPU without
operating-system network sockets. Preserves Step 29 testability.
"""

import collections
import queue
import time
from typing import Dict, List, Optional, Any

from chakrview.cognition.transport.base import Transport
from chakrview.cognition.transport.models import WireEnvelope, TransportHealth, TransportStatus
from chakrview.cognition.transport.errors import TransportError, TransportTimeoutError


class LoopbackWireTransport(Transport):
    """
    In-memory simulated network transport routing envelopes between endpoints.
    Shared router registry allows endpoints to discover and communicate with each other.
    """

    # Global in-process endpoint directory: endpoint_name -> queue.Queue[WireEnvelope]
    _ENDPOINTS: Dict[str, queue.Queue] = {}

    def __init__(self, endpoint: str = "loopback://default") -> None:
        self.endpoint = endpoint
        self._queue: queue.Queue[WireEnvelope] = queue.Queue()
        self._target_endpoint: Optional[str] = None
        self._status = TransportStatus.DISCONNECTED
        self._messages_sent = 0
        self._messages_received = 0
        self._error_count = 0
        self._is_listening = False

        # Register self
        LoopbackWireTransport._ENDPOINTS[self.endpoint] = self._queue

    def connect(self, endpoint: str, timeout_seconds: float = 5.0) -> bool:
        """
        Connect local transport to target endpoint.
        """
        if endpoint not in LoopbackWireTransport._ENDPOINTS:
            # If target doesn't exist yet, create queue for it
            LoopbackWireTransport._ENDPOINTS[endpoint] = queue.Queue()
        self._target_endpoint = endpoint
        self._status = TransportStatus.CONNECTED
        return True

    def listen(self, endpoint: str) -> None:
        """
        Bind local endpoint as listening server.
        """
        self.endpoint = endpoint
        LoopbackWireTransport._ENDPOINTS[endpoint] = self._queue
        self._is_listening = True
        self._status = TransportStatus.LISTENING

    def accept(self, timeout_seconds: float = 5.0) -> Optional[Any]:
        """
        Accept incoming loopback connection.
        """
        if not self._is_listening:
            return None
        return {"endpoint": self.endpoint, "status": "accepted"}

    def send(self, envelope: WireEnvelope, timeout_seconds: float = 5.0) -> bool:
        """
        Push envelope into target endpoint's queue.
        """
        if not self._target_endpoint:
            raise TransportError("Cannot send: loopback transport is not connected to a target endpoint.")

        target_q = LoopbackWireTransport._ENDPOINTS.get(self._target_endpoint)
        if target_q is None:
            raise TransportError(f"Target endpoint '{self._target_endpoint}' unreachable.")

        try:
            target_q.put(envelope, timeout=timeout_seconds)
            self._messages_sent += 1
            return True
        except queue.Full:
            self._error_count += 1
            raise TransportTimeoutError(f"Send to '{self._target_endpoint}' timed out (queue full).")

    def receive(self, timeout_seconds: float = 5.0) -> Optional[WireEnvelope]:
        """
        Pop next envelope from local queue.
        """
        try:
            envelope = self._queue.get(timeout=timeout_seconds)
            self._messages_received += 1
            return envelope
        except queue.Empty:
            return None

    def close(self) -> None:
        """
        Deregister endpoint and reset state.
        """
        self._status = TransportStatus.DISCONNECTED
        self._is_listening = False
        LoopbackWireTransport._ENDPOINTS.pop(self.endpoint, None)

    def health(self) -> TransportHealth:
        return TransportHealth(
            is_healthy=self._status in (TransportStatus.CONNECTED, TransportStatus.LISTENING),
            endpoint=self.endpoint,
            transport_type="loopback",
            latency_ms=0.01,
            active_connections=1 if self._target_endpoint else 0,
            messages_sent=self._messages_sent,
            messages_received=self._messages_received,
            error_count=self._error_count,
            details={"is_listening": self._is_listening, "queue_size": self._queue.qsize()},
        )

    def capabilities(self) -> Dict[str, Any]:
        return {
            "transport_type": "loopback",
            "is_physical": False,
            "supports_duplex": True,
            "supports_streaming": True,
            "max_bandwidth_mbps": 10000.0,
        }

    @classmethod
    def reset_all(cls) -> None:
        """Helper for test cleanup."""
        cls._ENDPOINTS.clear()
