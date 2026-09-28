"""
Transport Abstraction and Deterministic Loopback Implementation (Step 27).

Provides pluggable transport interfaces for distributed agent communication:
- Transport (Abstract Interface)
- LoopbackTransport (Deterministic in-memory test/local transport)
- TransportResponse, TransportStatus & Custom Transport Exceptions

CRITICAL ARCHITECTURAL AXIOMS:
1. TRANSPORT != AUTHORITY:
   The transport layer moves validated envelopes; it never makes cognitive
   decisions or authorizes capabilities.
2. NO MANDATORY EXTERNAL NETWORKING:
   Step 27 operates completely in-process using LoopbackTransport. Future network
   transports (gRPC, WebSockets, ZeroMQ) plug in seamlessly without touching core logic.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import time
from typing import Dict, List, Optional, Callable, Set, Any
import collections

from chakrview.cognition.distributed.models import (
    DistributedMessageEnvelope,
    TransportStatus,
)


class TransportError(Exception):
    """Base exception for transport failure."""
    pass


class TransportTimeout(TransportError):
    """Message delivery or request timed out."""
    pass


class TransportUnavailable(TransportError):
    """Destination node or transport endpoint is unreachable."""
    pass


class TransportProtocolError(TransportError):
    """Envelope formatting or protocol validation error."""
    pass


@dataclass
class TransportResponse:
    """Synchronous or asynchronous transport request-response result."""
    status: TransportStatus
    response_envelope: Optional[DistributedMessageEnvelope] = None
    latency_ms: float = 0.0
    error_message: Optional[str] = None

    def is_success(self) -> bool:
        return self.status == TransportStatus.SUCCESS and self.response_envelope is not None


class Transport(ABC):
    """
    Abstract interface for distributed message transport across nodes.
    """

    @abstractmethod
    def send(self, message: DistributedMessageEnvelope) -> bool:
        """Asynchronously send an envelope to its destination node."""
        pass

    @abstractmethod
    def receive(self, receiver_node_id: str, timeout_ms: float = 1000.0) -> Optional[DistributedMessageEnvelope]:
        """Fetch the next incoming message for a specific node."""
        pass

    @abstractmethod
    def request(self, message: DistributedMessageEnvelope, timeout_ms: float = 1000.0) -> TransportResponse:
        """Synchronously send a request envelope and wait for response."""
        pass

    @abstractmethod
    def broadcast(self, message: DistributedMessageEnvelope) -> int:
        """Broadcast an envelope to all reachable nodes. Returns number delivered."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Release transport resources."""
        pass

    @abstractmethod
    def health_check(self, node_id: str) -> bool:
        """Verify reachability of a given node."""
        pass


class LoopbackTransport(Transport):
    """
    Deterministic in-memory loopback transport for multi-node simulation on CPU.
    Requires zero network sockets or external dependencies.
    """

    def __init__(self) -> None:
        self._mailboxes: Dict[str, collections.deque] = {}
        self._handlers: Dict[str, Callable[[DistributedMessageEnvelope], Optional[DistributedMessageEnvelope]]] = {}
        self._registered_nodes: Set[str] = set()
        self._simulated_drop_nodes: Set[str] = set()
        self._simulated_timeout_nodes: Set[str] = set()
        self._simulated_error_nodes: Set[str] = set()
        self._closed: bool = False

    def register_node_endpoint(
        self,
        node_id: str,
        handler: Optional[Callable[[DistributedMessageEnvelope], Optional[DistributedMessageEnvelope]]] = None,
    ) -> None:
        """Register a node mailbox and optional dispatch handler."""
        if self._closed:
            raise TransportError("Transport is closed.")
        self._registered_nodes.add(node_id)
        if node_id not in self._mailboxes:
            self._mailboxes[node_id] = collections.deque()
        if handler is not None:
            self._handlers[node_id] = handler

    def unregister_node_endpoint(self, node_id: str) -> None:
        """Remove a node mailbox and handlers."""
        self._registered_nodes.discard(node_id)
        self._handlers.pop(node_id, None)
        self._mailboxes.pop(node_id, None)

    def set_fault_injection(
        self,
        drop_nodes: Optional[Set[str]] = None,
        timeout_nodes: Optional[Set[str]] = None,
        error_nodes: Optional[Set[str]] = None,
    ) -> None:
        """Inject simulated network faults for deterministic testing."""
        self._simulated_drop_nodes = set(drop_nodes or set())
        self._simulated_timeout_nodes = set(timeout_nodes or set())
        self._simulated_error_nodes = set(error_nodes or set())

    def send(self, message: DistributedMessageEnvelope) -> bool:
        if self._closed:
            raise TransportError("Transport is closed.")
        target_node = message.route.receiver_node_id

        if target_node in self._simulated_drop_nodes:
            # Silently drop for simulation
            return False

        if target_node in self._simulated_error_nodes:
            raise TransportProtocolError(f"Simulated transport protocol error delivering to {target_node}")

        if target_node not in self._registered_nodes and target_node != "BROADCAST":
            raise TransportUnavailable(f"Target node '{target_node}' is not registered on loopback transport.")

        if target_node not in self._mailboxes:
            self._mailboxes[target_node] = collections.deque()

        self._mailboxes[target_node].append(message)
        return True

    def receive(self, receiver_node_id: str, timeout_ms: float = 1000.0) -> Optional[DistributedMessageEnvelope]:
        if self._closed:
            raise TransportError("Transport is closed.")
        if receiver_node_id not in self._mailboxes or not self._mailboxes[receiver_node_id]:
            return None
        return self._mailboxes[receiver_node_id].popleft()

    def request(self, message: DistributedMessageEnvelope, timeout_ms: float = 1000.0) -> TransportResponse:
        start_t = time.time()
        if self._closed:
            raise TransportError("Transport is closed.")

        target_node = message.route.receiver_node_id

        if target_node in self._simulated_drop_nodes:
            return TransportResponse(
                status=TransportStatus.TIMEOUT,
                latency_ms=(time.time() - start_t) * 1000.0,
                error_message="Message dropped by simulated drop policy.",
            )

        if target_node in self._simulated_timeout_nodes:
            return TransportResponse(
                status=TransportStatus.TIMEOUT,
                latency_ms=(time.time() - start_t) * 1000.0,
                error_message=f"Request to node '{target_node}' timed out.",
            )

        if target_node in self._simulated_error_nodes:
            return TransportResponse(
                status=TransportStatus.PROTOCOL_ERROR,
                latency_ms=(time.time() - start_t) * 1000.0,
                error_message=f"Simulated protocol failure on node '{target_node}'.",
            )

        if target_node not in self._registered_nodes:
            return TransportResponse(
                status=TransportStatus.UNAVAILABLE,
                latency_ms=(time.time() - start_t) * 1000.0,
                error_message=f"Node '{target_node}' is unreachable.",
            )

        handler = self._handlers.get(target_node)
        if handler is None:
            # Fallback to queuing
            self.send(message)
            return TransportResponse(
                status=TransportStatus.SUCCESS,
                latency_ms=(time.time() - start_t) * 1000.0,
            )

        try:
            resp_envelope = handler(message)
            latency = (time.time() - start_t) * 1000.0
            if resp_envelope is None:
                return TransportResponse(
                    status=TransportStatus.TIMEOUT,
                    latency_ms=latency,
                    error_message="Handler returned no response.",
                )
            return TransportResponse(
                status=TransportStatus.SUCCESS,
                response_envelope=resp_envelope,
                latency_ms=latency,
            )
        except Exception as e:
            latency = (time.time() - start_t) * 1000.0
            return TransportResponse(
                status=TransportStatus.PROTOCOL_ERROR,
                latency_ms=latency,
                error_message=f"Handler execution failed: {str(e)}",
            )

    def broadcast(self, message: DistributedMessageEnvelope) -> int:
        if self._closed:
            raise TransportError("Transport is closed.")
        delivered = 0
        for node_id in list(self._registered_nodes):
            if node_id != message.route.sender_node_id and node_id not in self._simulated_drop_nodes:
                if node_id not in self._mailboxes:
                    self._mailboxes[node_id] = collections.deque()
                self._mailboxes[node_id].append(message)
                delivered += 1
        return delivered

    def health_check(self, node_id: str) -> bool:
        if self._closed:
            return False
        return (
            node_id in self._registered_nodes
            and node_id not in self._simulated_drop_nodes
            and node_id not in self._simulated_timeout_nodes
        )

    def close(self) -> None:
        self._closed = True
        self._mailboxes.clear()
        self._handlers.clear()
        self._registered_nodes.clear()
