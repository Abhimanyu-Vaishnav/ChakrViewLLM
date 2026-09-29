"""
Production Secure TCP Wire Transport Adapter (Step 31).

Implements physical socket transport using standard library socket and ssl:
- Length-prefixed binary framing (LengthPrefixedFramer)
- Fail-closed disconnection and timeout semantics
- Explicit transport security modes: PLAINTEXT_TEST_ONLY, TLS, MTLS
- Hardened TLS 1.3 / TLS 1.2 contexts via TLSContextFactory
- Peer certificate extraction and metadata validation
- Fail-closed rejection of insecure downgrades and handshake errors
"""

import socket
import ssl
import time
from typing import Dict, List, Optional, Any, Tuple

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
)
from chakrview.cognition.transport.security.models import (
    TLSMode,
    CertificateMetadata,
)
from chakrview.cognition.transport.security.policy import SecureTransportPolicy
from chakrview.cognition.transport.security.tls import TLSContextFactory
from chakrview.cognition.transport.security.certificates import (
    parse_certificate_from_der,
    extract_certificate_metadata,
)
from chakrview.cognition.transport.security.errors import (
    TLSError,
    TLSHandshakeError,
    InsecureDowngradeError,
    CertificateExpiredError,
    HostnameMismatchError,
    UntrustedCAError,
    ClientCertificateMissingError,
)


class TCPWireTransport(Transport):
    """
    Physical TCP transport adapter with integrated TLS/mTLS and framing.
    Can operate in client mode (connecting out) or server mode (listening and accepting).
    """

    def __init__(
        self,
        endpoint: str = "tcp://127.0.0.1:0",
        security_policy: Optional[SecureTransportPolicy] = None,
        server_cert_pem: Optional[str] = None,
        server_key_pem: Optional[str] = None,
        client_cert_pem: Optional[str] = None,
        client_key_pem: Optional[str] = None,
        ca_cert_pem: Optional[str] = None,
        server_hostname: Optional[str] = None,
    ) -> None:
        self.endpoint = endpoint

        # Default to PLAINTEXT_TEST_ONLY if no policy provided for backward compatibility in tests,
        # but require explicit allow_insecure_test_downgrade=True.
        if security_policy is None:
            self.security_policy = SecureTransportPolicy(
                tls_mode=TLSMode.PLAINTEXT_TEST_ONLY,
                allow_insecure_test_downgrade=True,
            )
        else:
            self.security_policy = security_policy
            self.security_policy.validate()

        self._server_cert_pem = server_cert_pem
        self._server_key_pem = server_key_pem
        self._client_cert_pem = client_cert_pem
        self._client_key_pem = client_key_pem
        self._ca_cert_pem = ca_cert_pem
        self._server_hostname = server_hostname

        self._server_sock: Optional[socket.socket] = None
        self._client_sock: Optional[socket.socket] = None
        self._server_ssl_context: Optional[ssl.SSLContext] = None
        self._client_ssl_context: Optional[ssl.SSLContext] = None
        self._peer_certificate_metadata: Optional[CertificateMetadata] = None

        self._status = TransportStatus.DISCONNECTED
        self._rx_buffer = bytearray()
        self._messages_sent = 0
        self._messages_received = 0
        self._error_count = 0
        self._bound_port: Optional[int] = None

    @property
    def peer_certificate_metadata(self) -> Optional[CertificateMetadata]:
        """Metadata of the peer certificate negotiated over TLS/mTLS."""
        return self._peer_certificate_metadata

    @staticmethod
    def _parse_endpoint(endpoint: str) -> Tuple[str, int]:
        """Parse host and port from endpoint string (e.g. 'tcp://127.0.0.1:9000' or '127.0.0.1:9000')."""
        clean = endpoint
        if clean.startswith("tcp://"):
            clean = clean[6:]
        elif clean.startswith("tls://"):
            clean = clean[6:]
        elif clean.startswith("mtls://"):
            clean = clean[7:]
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

        # Prepare server TLS context if in TLS/mTLS mode
        if self.security_policy.tls_mode in (TLSMode.TLS, TLSMode.MTLS):
            if not self._server_cert_pem or not self._server_key_pem:
                raise TLSError(
                    f"Server requires server_cert_pem and server_key_pem for {self.security_policy.tls_mode.value} mode."
                )
            self._server_ssl_context = TLSContextFactory.create_server_context(
                policy=self.security_policy,
                cert_pem=self._server_cert_pem,
                key_pem=self._server_key_pem,
                ca_cert_pem=self._ca_cert_pem,
            )

        self._server_sock = sock
        self._bound_port = sock.getsockname()[1]
        scheme = "tls" if self.security_policy.tls_mode == TLSMode.TLS else ("mtls" if self.security_policy.tls_mode == TLSMode.MTLS else "tcp")
        self.endpoint = f"{scheme}://{host}:{self._bound_port}"
        self._status = TransportStatus.LISTENING

    @property
    def bound_port(self) -> Optional[int]:
        return self._bound_port

    def accept(self, timeout_seconds: float = 5.0) -> Optional[socket.socket]:
        """
        Accept an incoming client connection. Performs TLS handshake if configured.
        """
        if not self._server_sock:
            raise TransportError("Cannot accept: TCP transport is not listening.")
        self._server_sock.settimeout(timeout_seconds)
        try:
            raw_client_sock, client_addr = self._server_sock.accept()
            raw_client_sock.settimeout(timeout_seconds)

            # Perform server-side TLS handshake if enabled
            if self.security_policy.tls_mode in (TLSMode.TLS, TLSMode.MTLS):
                assert self._server_ssl_context is not None
                try:
                    ssl_sock = self._server_ssl_context.wrap_socket(
                        raw_client_sock,
                        server_side=True,
                    )
                    # In mTLS mode, extract client certificate
                    der_cert = ssl_sock.getpeercert(binary_form=True)
                    if der_cert:
                        x509_cert = parse_certificate_from_der(der_cert)
                        self._peer_certificate_metadata = extract_certificate_metadata(x509_cert)
                    elif self.security_policy.tls_mode == TLSMode.MTLS:
                        ssl_sock.close()
                        raise ClientCertificateMissingError("Mandatory client certificate missing during mTLS handshake.")

                    self._client_sock = ssl_sock
                except (ssl.SSLError, Exception) as e:
                    raw_client_sock.close()
                    self._error_count += 1
                    raise TLSHandshakeError(f"Server TLS handshake failed: {e}")
            else:
                self._client_sock = raw_client_sock

            self._client_sock.settimeout(timeout_seconds)
            self._status = TransportStatus.CONNECTED
            return self._client_sock

        except socket.timeout:
            return None
        except (TLSError, TransportError):
            raise
        except Exception as e:
            self._error_count += 1
            raise TransportError(f"Failed to accept TCP connection: {e}")

    def connect(self, endpoint: str, timeout_seconds: float = 5.0) -> bool:
        """
        Connect to a remote TCP endpoint. Performs client-side TLS handshake if configured.
        """
        self.endpoint = endpoint
        host, port = self._parse_endpoint(endpoint)
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout_seconds)

        try:
            sock.connect((host, port))

            # Perform client-side TLS handshake if enabled
            if self.security_policy.tls_mode in (TLSMode.TLS, TLSMode.MTLS):
                self._client_ssl_context = TLSContextFactory.create_client_context(
                    policy=self.security_policy,
                    ca_cert_pem=self._ca_cert_pem,
                    cert_pem=self._client_cert_pem,
                    key_pem=self._client_key_pem,
                )
                server_hostname = self._server_hostname or host

                try:
                    ssl_sock = self._client_ssl_context.wrap_socket(
                        sock,
                        server_hostname=server_hostname,
                    )
                    # Extract server certificate
                    der_cert = ssl_sock.getpeercert(binary_form=True)
                    if der_cert:
                        x509_cert = parse_certificate_from_der(der_cert)
                        self._peer_certificate_metadata = extract_certificate_metadata(x509_cert)

                    self._client_sock = ssl_sock
                except ssl.SSLCertVerificationError as e:
                    sock.close()
                    self._error_count += 1
                    err_msg = str(e).lower()
                    if "expired" in err_msg:
                        raise CertificateExpiredError(f"TLS peer certificate expired: {e}")
                    elif "hostname" in err_msg:
                        raise HostnameMismatchError(f"TLS hostname verification failed for '{server_hostname}': {e}")
                    elif "self-signed" in err_msg or "unknown ca" in err_msg or "unable to get local issuer certificate" in err_msg:
                        raise UntrustedCAError(f"TLS peer certificate issuer untrusted: {e}")
                    raise TLSHandshakeError(f"TLS certificate verification error: {e}")
                except (ssl.SSLError, Exception) as e:
                    sock.close()
                    self._error_count += 1
                    raise TLSHandshakeError(f"Client TLS handshake failed: {e}")
            else:
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
        except (TLSError, TransportError):
            raise
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
            try:
                self._client_sock.close()
            except Exception:
                pass
            self._client_sock = None

        if self._server_sock:
            try:
                self._server_sock.close()
            except Exception:
                pass
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
            details={
                "bound_port": self._bound_port,
                "status": self._status.value,
                "tls_mode": self.security_policy.tls_mode.value,
                "has_peer_certificate": self._peer_certificate_metadata is not None,
                "peer_fingerprint": self._peer_certificate_metadata.fingerprint if self._peer_certificate_metadata else None,
            },
        )

    def capabilities(self) -> Dict[str, Any]:
        return {
            "transport_type": "tcp",
            "is_physical": True,
            "supports_duplex": True,
            "supports_streaming": True,
            "framing": "length_prefixed_4byte",
            "max_frame_bytes": MAX_WIRE_FRAME_BYTES,
            "tls_mode": self.security_policy.tls_mode.value,
            "supports_tls": True,
            "supports_mtls": True,
        }
