"""
Secure TCP Wire Transport Adapter (Step 30).

Implements physical socket transport using standard library socket, length-prefixed
binary framing, strict byte ceilings, and fail-closed disconnection semantics.
"""

import socket
import time
from typing import Dict, List, Optional, Any, Tuple
from urllib.parse import urlparse

from chakrview.cognition.transport.base import Transport
from chakrview.cognition.transport.models import (
    WireEnvelope,
    TransportHealth,
    TransportStatus,
    MAX_WIRE_FRAME_BYTES,
)
from chakrview.cognition.transport.framing import LengthPrefixedFramer
from chakrview.cognition.transport.serialization import DeterministicWireSerializer
from chakrview.cognition.transport.errors import (
    TransportError,
    TransportTimeoutError,
    TransportUnavailableError,
    TransportProtocolError,
)


class TCPWireTransport(Transport):
    """
    Physical TCP transport adapter using stream sockets and length-prefixed wire envelopes.
    Can operate in client mode (connecting out) or server mode (listening and accepting).
    """

    def __init__(self, endpoint: str = "tcp://127.0.0.1:0") -> None:
        self.endpoint = endpoint
        self._server_sock: Optional[socket.socket] = None
        self._client_sock: Optional[socket.socket] = None
        self._status = TransportStatus.DISCONNECTED
        self._rx_buffer = bytearray()
        self._messages_sent = 0
        self._messages_received = 0
        self._error_count = 0
        self._bound_port: Optional[int] = None

    @staticmethod
    def _parse_endpoint(endpoint: str) -> Tuple[str, int]:
        """Parse host and port from endpoint string (e.g. 'tcp://127.0.0.1:9000' or '127.0.0.1:9000')."""
        clean = endpoint
        if clean.startswith("tcp://"):
            clean = clean[6:]
        parts = clean.split(":")
        if len(parts) != 2:
            raise ValueError(f"Invalid TCP endpoint format '{endpoint}'. Expected 'host:port' or 'tcp://host:port'.")
        host = parts[0]
        port = int(parts[1])
        return host, port

    def listen(self, endpoint: Optional[str] = None) -> None:
        """
        Bind server socket to endpoint and start listening.
        """
        if endpoint:
            self.endpoint = endpoint
        host, port = self._parse_endpoint(self.endpoint)

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((host, port))
        sock.listen(5)
        sock.settimeout(5.0)

        self._server_sock = sock
        self._bound_port = sock.getsockname()[1]
        self.endpoint = f"tcp://{host}:{self._bound_port}"
        self._status = TransportStatus.LISTENING

    @property
    def bound_port(self) -> Optional[int]:
        return self._bound_port

    def accept(self, timeout_seconds: float = 5.0) -> Optional[socket.socket]:
        """
        Accept an incoming client connection.
        """
        if not self._server_sock:
            raise TransportError("Cannot accept: TCP transport is not listening.")
        self._server_sock.settimeout(timeout_seconds)
        try:
            client_sock, client_addr = self._server_sock.accept()
            client_sock.settimeout(timeout_seconds)
            self._client_sock = client_sock
            self._status = TransportStatus.CONNECTED
            return client_sock
        except socket.timeout:
            return None
        except Exception as e:
            self._error_count += 1
            raise TransportError(f"Failed to accept TCP connection: {e}")

    def connect(self, endpoint: str, timeout_seconds: float = 5.0) -> bool:
        """
        Connect to a remote TCP endpoint.
        """
        self.endpoint = endpoint
        host, port = self._parse_endpoint(endpoint)
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout_seconds)
        try:
            sock.connect((host, port))
            self._client_sock = sock
            self._status = TransportStatus.CONNECTED
            return True
        except (ConnectionRefusedError, socket.gaierror) as e:
            sock.close()
            self._error_count += 1
            raise TransportUnavailableError(f"Cannot connect to '{endpoint}': {e}")
        except socket.timeout:
            sock.close()
            self._error_count += 1
            raise TransportTimeoutError(f"Connection to '{endpoint}' timed out.")
        except Exception as e:
            sock.close()
            self._error_count += 1
            raise TransportError(f"Connection failed: {e}")

    def send(self, envelope: WireEnvelope, timeout_seconds: float = 5.0) -> bool:
        """
        Serialize envelope, frame with length header, and transmit over active socket.
        """
        if not self._client_sock:
            raise TransportError("Cannot send: TCP socket is not connected.")

        raw_payload = DeterministicWireSerializer.serialize(envelope)
        framed_data = LengthPrefixedFramer.encode_frame(raw_payload)

        self._client_sock.settimeout(timeout_seconds)
        try:
            self._client_sock.sendall(framed_data)
            self._messages_sent += 1
            return True
        except socket.timeout:
            self._error_count += 1
            raise TransportTimeoutError("TCP send timed out.")
        except (BrokenPipeError, ConnectionResetError) as e:
            self._status = TransportStatus.DISCONNECTED
            self._error_count += 1
            raise TransportError(f"Socket disconnected during send: {e}")

    def receive(self, timeout_seconds: float = 5.0) -> Optional[WireEnvelope]:
        """
        Read from socket, extract complete length-prefixed frame, and deserialize.
        """
        if not self._client_sock:
            raise TransportError("Cannot receive: TCP socket is not connected.")

        # Check if we already have a complete frame in the buffer
        cached_frame = LengthPrefixedFramer.decode_frame(self._rx_buffer)
        if cached_frame is not None:
            self._messages_received += 1
            return DeterministicWireSerializer.deserialize(cached_frame)

        self._client_sock.settimeout(timeout_seconds)
        start_time = time.time()

        while True:
            remaining_time = timeout_seconds - (time.time() - start_time)
            if remaining_time <= 0:
                return None
            self._client_sock.settimeout(remaining_time)

            try:
                chunk = self._client_sock.recv(4096)
                if not chunk:
                    # Remote closed connection
                    self._status = TransportStatus.DISCONNECTED
                    return None
                self._rx_buffer.extend(chunk)

                # Attempt decoding frame
                frame = LengthPrefixedFramer.decode_frame(self._rx_buffer)
                if frame is not None:
                    self._messages_received += 1
                    return DeterministicWireSerializer.deserialize(frame)

            except socket.timeout:
                return None
            except Exception as e:
                self._error_count += 1
                raise TransportError(f"TCP socket error during receive: {e}")

    def close(self) -> None:
        """
        Cleanly shut down client and server sockets.
        """
        if self._client_sock:
            try:
                self._client_sock.shutdown(socket.SHUT_RDWR)
            except Exception:
                pass
            self._client_sock.close()
            self._client_sock = None

        if self._server_sock:
            self._server_sock.close()
            self._server_sock = None

        self._status = TransportStatus.DISCONNECTED
        self._rx_buffer.clear()

    def health(self) -> TransportHealth:
        return TransportHealth(
            is_healthy=self._status in (TransportStatus.CONNECTED, TransportStatus.LISTENING),
            endpoint=self.endpoint,
            transport_type="tcp",
            latency_ms=0.5,
            active_connections=1 if self._client_sock else 0,
            messages_sent=self._messages_sent,
            messages_received=self._messages_received,
            error_count=self._error_count,
            details={"bound_port": self._bound_port, "status": self._status.value},
        )

    def capabilities(self) -> Dict[str, Any]:
        return {
            "transport_type": "tcp",
            "is_physical": True,
            "supports_duplex": True,
            "supports_streaming": True,
            "framing": "length_prefixed_4byte",
            "max_frame_bytes": MAX_WIRE_FRAME_BYTES,
        }
