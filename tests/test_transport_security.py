"""
Step 31 Test Suite: Production Transport Security, TLS/mTLS & Certificate Lifecycle.

Tests comprehensive transport security invariants:
1. TLS policy configuration, version enforcement, and fail-closed downgrade protection
2. X.509 certificate parsing, fingerprinting, lifecycle inspection, and deterministic validation
3. Peer identity to TLS certificate binding (TLS_IDENTITY != FEDERATION_AUTHORITY)
4. TCP TLS and mutual TLS (mTLS) socket transport round trips
5. Rejection of untrusted CAs, expired certificates, hostname mismatches, and missing client certs
6. HTTP/2 and gRPC adapter capability boundaries
7. CapabilityGate mediation, tenant isolation, and trust independence (TLS != TRUST)
8. Strict zero-leakage protection of private key material in logs, repr, and traces
9. ChakrMicro neural core immutability (ΔW = 0)
"""

import os
import ssl
import time
import threading
from typing import Dict, Any, Optional

import pytest
import torch

from chakrview.cognition.transport.security import (
    TLSMode,
    TLSProtocolVersion,
    CertificateLifecycleState,
    CertificateMetadata,
    PeerCertificateBinding,
    PeerCertificateBinder,
    SecureTransportPolicy,
    HermeticPKIBuilder,
    CertificateRevocationRegistry,
    compute_certificate_fingerprint,
    parse_certificate_from_pem,
    extract_certificate_metadata,
    validate_certificate,
    validate_certificate_or_raise,
    TLSContextFactory,
    TLSError,
    TLSConfigurationError,
    TLSHandshakeError,
    InsecureDowngradeError,
    CertificateValidationError,
    CertificateExpiredError,
    CertificateRevokedError,
    HostnameMismatchError,
    UntrustedCAError,
    ClientCertificateMissingError,
    PeerBindingMismatchError,
)
from chakrview.cognition.transport.tcp import TCPWireTransport
from chakrview.cognition.transport.http2 import HTTP2WireTransport, is_http2_available
from chakrview.cognition.transport.grpc import GRPCWireTransport, is_grpc_available
from chakrview.cognition.transport.models import WireEnvelope, MessageType
from chakrview.cognition.peering.engine import CrossZoneFederationEngine, CrossZoneAuthorizationError
from chakrview.cognition.peering.isolation import IsolationViolationError
from chakrview.cognition.peering.policy import CrossZoneFederationPolicy
from chakrview.cognition.peering.models import (
    PeerIdentity,
    PeerDeclaration,
    TrustLevel,
    TrustGrant,
    FederationScope,
    AuditEventType,
)
from chakrview.cognition.peering.authentication import ChallengeResponseAuthenticator
from chakrview.cognition.peering.crypto import Ed25519PrivateKeyWrapper, CryptographicPeerIdentity
from chakrview.capability.gate import CapabilityGate, CapabilityAuthorizationError
from chakrview.capability.registry import CapabilityRegistry
from chakrview.capability.contract import (
    Capability,
    CapabilityDescriptor,
    CapabilityRequest,
    CapabilityResult,
    CapabilityContext,
    RiskClassification,
)
from chakrview.brain.model import ChakrMicro
from chakrview.brain.config import ModelConfig

from chakrview.cognition.diagnostics.integrity import CoreIntegrityGuard


# ============================================================================
# Test Fixtures & Helpers
# ============================================================================

class MockEchoCapability(Capability):
    """Simple safe capability for test verification."""

    @property
    def descriptor(self) -> CapabilityDescriptor:
        return CapabilityDescriptor(
            capability_id="echo_task",
            name="Mock Echo",
            description="Echoes input",
            risk_level=RiskClassification.READ_ONLY,
            required_permissions=["capability.compute.echo"],
        )

    def execute(self, request: CapabilityRequest, context: Optional[CapabilityContext] = None) -> CapabilityResult:
        return CapabilityResult(
            request_id=request.request_id,
            capability_id="echo_task",
            success=True,
            output={"echo": request.parameters.get("input", "OK")},
        )


def create_test_gate() -> CapabilityGate:
    registry = CapabilityRegistry()
    registry.register(MockEchoCapability())
    return CapabilityGate(registry=registry)



@pytest.fixture(scope="module")
def pki_bundle():
    """Module-scoped hermetic PKI bundle for fast tests."""
    ca_pem, ca_key_pem, ca_cert, ca_key = HermeticPKIBuilder.create_ca()
    srv_pem, srv_key_pem, srv_cert, srv_key = HermeticPKIBuilder.create_server_cert(
        ca_cert, ca_key, common_name="127.0.0.1", san_ips=["127.0.0.1"], san_dns=["localhost"]
    )
    cli_pem, cli_key_pem, cli_cert, cli_key = HermeticPKIBuilder.create_client_cert(
        ca_cert, ca_key, common_name="client-peer"
    )
    return {
        "ca_pem": ca_pem,
        "ca_key_pem": ca_key_pem,
        "ca_cert": ca_cert,
        "ca_key": ca_key,
        "srv_pem": srv_pem,
        "srv_key_pem": srv_key_pem,
        "srv_cert": srv_cert,
        "cli_pem": cli_pem,
        "cli_key_pem": cli_key_pem,
        "cli_cert": cli_cert,
    }


# ============================================================================
# 1. TLS Policy & Context Configuration Tests
# ============================================================================

def test_tls_policy_defaults_and_validation():
    policy = SecureTransportPolicy()
    assert policy.tls_mode == TLSMode.TLS
    assert policy.minimum_tls_version == TLSProtocolVersion.TLS_1_3
    assert policy.verify_peer_certificate is True
    assert policy.verify_hostname is True
    assert policy.allow_insecure_test_downgrade is False
    assert policy.allow_tls_1_2 is False

    # Policy dict has zero secrets
    p_dict = policy.to_dict()
    assert "tls_mode" in p_dict
    assert "private_key" not in str(p_dict).lower()


def test_tls_policy_rejects_insecure_downgrades():
    # Plaintext mode without test downgrade flag fails
    with pytest.raises(InsecureDowngradeError):
        SecureTransportPolicy(tls_mode=TLSMode.PLAINTEXT_TEST_ONLY, allow_insecure_test_downgrade=False)

    # Disabling certificate verification without test flag fails
    with pytest.raises(InsecureDowngradeError):
        SecureTransportPolicy(verify_peer_certificate=False, allow_insecure_test_downgrade=False)

    # TLS 1.2 without allow_tls_1_2 fails
    with pytest.raises(TLSConfigurationError):
        SecureTransportPolicy(minimum_tls_version=TLSProtocolVersion.TLS_1_2, allow_tls_1_2=False)


def test_tls_context_factory_server_and_client_creation(pki_bundle):
    policy = SecureTransportPolicy(tls_mode=TLSMode.TLS, allow_tls_1_2=True)
    server_ctx = TLSContextFactory.create_server_context(
        policy=policy,
        cert_pem=pki_bundle["srv_pem"],
        key_pem=pki_bundle["srv_key_pem"],
        ca_cert_pem=pki_bundle["ca_pem"],
    )
    assert isinstance(server_ctx, ssl.SSLContext)
    assert server_ctx.minimum_version == ssl.TLSVersion.TLSv1_3

    client_ctx = TLSContextFactory.create_client_context(
        policy=policy,
        ca_cert_pem=pki_bundle["ca_pem"],
    )
    assert isinstance(client_ctx, ssl.SSLContext)
    assert client_ctx.verify_mode == ssl.CERT_REQUIRED
    assert client_ctx.check_hostname is True


def test_tls_context_factory_rejects_mtls_server_without_ca(pki_bundle):
    mtls_policy = SecureTransportPolicy(tls_mode=TLSMode.MTLS, allow_tls_1_2=True)
    with pytest.raises(TLSConfigurationError, match="requires at least one trusted CA source"):
        TLSContextFactory.create_server_context(
            policy=mtls_policy,
            cert_pem=pki_bundle["srv_pem"],
            key_pem=pki_bundle["srv_key_pem"],
            ca_cert_pem=None,
        )


def test_tls_context_factory_rejects_mtls_client_without_cert(pki_bundle):
    mtls_policy = SecureTransportPolicy(tls_mode=TLSMode.MTLS, allow_tls_1_2=True)
    with pytest.raises(TLSConfigurationError, match="Client context for mTLS requires cert_pem and key_pem"):
        TLSContextFactory.create_client_context(
            policy=mtls_policy,
            ca_cert_pem=pki_bundle["ca_pem"],
            cert_pem=None,
            key_pem=None,
        )


# ============================================================================
# 2. X.509 Certificate & Lifecycle Tests
# ============================================================================

def test_certificate_metadata_extraction_and_fingerprint(pki_bundle):
    meta = extract_certificate_metadata(pki_bundle["srv_cert"])
    assert meta.fingerprint.startswith("SHA256:")
    assert meta.subject.get("commonName") == "127.0.0.1"
    assert "127.0.0.1" in meta.san_ip_addresses
    assert "localhost" in meta.san_dns_names
    assert "server_auth" in meta.extended_key_usage
    assert meta.is_ca is False
    assert meta.lifecycle_state == CertificateLifecycleState.VALID

    # Check safe sanitized representation
    s_rep = meta.sanitized_repr()
    assert "Cert(CN=" in s_rep
    assert "SHA256:" in s_rep


def test_certificate_validation_helpers_happy_path(pki_bundle):
    meta = extract_certificate_metadata(pki_bundle["srv_cert"])
    valid, reason = validate_certificate(meta, expected_hostname="127.0.0.1")
    assert valid is True
    assert reason is None

    # validate_certificate_or_raise does not raise
    validate_certificate_or_raise(meta, expected_hostname="127.0.0.1")


def test_certificate_validation_fails_on_hostname_mismatch(pki_bundle):
    meta = extract_certificate_metadata(pki_bundle["srv_cert"])
    valid, reason = validate_certificate(meta, expected_hostname="remote.host.invalid")
    assert valid is False
    assert "mismatch" in reason

    with pytest.raises(HostnameMismatchError):
        validate_certificate_or_raise(meta, expected_hostname="remote.host.invalid")


def test_certificate_validation_fails_on_expired(pki_bundle):
    meta = extract_certificate_metadata(pki_bundle["srv_cert"])
    # Pass a simulated epoch time 100 days in the future (cert validity is 60 days)
    future_time = time.time() + (86400.0 * 100.0)
    valid, reason = validate_certificate(meta, current_time=future_time)
    assert valid is False
    assert "expired" in reason

    with pytest.raises(CertificateExpiredError):
        validate_certificate_or_raise(meta, current_time=future_time)


def test_certificate_revocation_registry_and_validation(pki_bundle):
    meta = extract_certificate_metadata(pki_bundle["srv_cert"])
    registry = CertificateRevocationRegistry()
    assert registry.is_revoked(meta.fingerprint) is False

    registry.revoke(meta.fingerprint, reason="Compromised test key", epoch=5)
    assert registry.is_revoked(meta.fingerprint) is True
    rec = registry.get_revocation_record(meta.fingerprint)
    assert rec["reason"] == "Compromised test key"
    assert rec["revoked_at_epoch"] == 5

    valid, reason = validate_certificate(meta, revocation_registry=registry)
    assert valid is False
    assert "revoked" in reason

    with pytest.raises(CertificateRevokedError):
        validate_certificate_or_raise(meta, revocation_registry=registry)


def test_certificate_fingerprint_pinning(pki_bundle):
    meta = extract_certificate_metadata(pki_bundle["srv_cert"])
    allowed = {meta.fingerprint}
    valid, reason = validate_certificate(meta, allowed_fingerprints=allowed)
    assert valid is True

    # Unknown fingerprint in pinning set
    disallowed = {"SHA256:00:11:22:33:44:55:66:77:88:99:AA:BB:CC:DD:EE:FF:00:11:22:33:44:55:66:77:88:99:AA:BB:CC:DD:EE:FF"}
    valid, reason = validate_certificate(meta, allowed_fingerprints=disallowed)
    assert valid is False
    assert "pinning" in reason


def test_certificate_parsing_fails_on_malformed_pem():
    with pytest.raises(CertificateValidationError):
        parse_certificate_from_pem("-----BEGIN CERTIFICATE-----\nNOT_VALID_PEM\n-----END CERTIFICATE-----")


# ============================================================================
# 3. Peer Identity to Certificate Binding Tests
# ============================================================================

def test_peer_certificate_binding_lifecycle(pki_bundle):
    binder = PeerCertificateBinder()
    meta = extract_certificate_metadata(pki_bundle["cli_cert"])

    binding = binder.bind_peer(
        peer_id="peer-123",
        certificate_fingerprint=meta.fingerprint,
        expected_common_name="client-peer",
        epoch=1,
    )
    assert binding.is_active is True
    assert binding.certificate_fingerprint == meta.fingerprint

    # Verification passes
    valid, reason = binder.verify_binding("peer-123", meta)
    assert valid is True
    assert reason is None

    # Mismatched fingerprint fails
    srv_meta = extract_certificate_metadata(pki_bundle["srv_cert"])
    valid, reason = binder.verify_binding("peer-123", srv_meta)
    assert valid is False
    assert "mismatch" in reason

    with pytest.raises(PeerBindingMismatchError):
        binder.verify_binding_or_raise("peer-123", srv_meta)

    # Unknown peer fails
    valid, reason = binder.verify_binding("unknown-peer", meta)
    assert valid is False
    assert "No certificate binding exists" in reason

    # Rotation works
    rotated = binder.rotate_binding("peer-123", new_certificate_fingerprint=srv_meta.fingerprint, epoch=2)
    assert rotated.certificate_fingerprint == srv_meta.fingerprint
    valid, _ = binder.verify_binding("peer-123", srv_meta)
    assert valid is True


# ============================================================================
# 4. TCP Transport TLS & mTLS Integration Tests
# ============================================================================

def test_tcp_wire_transport_standard_tls_round_trip(pki_bundle):
    """Test standard 1-way TLS (server authenticated by client)."""
    server_policy = SecureTransportPolicy(tls_mode=TLSMode.TLS, allow_tls_1_2=True)
    client_policy = SecureTransportPolicy(tls_mode=TLSMode.TLS, allow_tls_1_2=True)

    server = TCPWireTransport(
        endpoint="tcp://127.0.0.1:0",
        security_policy=server_policy,
        server_cert_pem=pki_bundle["srv_pem"],
        server_key_pem=pki_bundle["srv_key_pem"],
        ca_cert_pem=pki_bundle["ca_pem"],
    )
    server.listen()
    port = server.bound_port

    client = TCPWireTransport(
        endpoint=f"tcp://127.0.0.1:{port}",
        security_policy=client_policy,
        ca_cert_pem=pki_bundle["ca_pem"],
        server_hostname="127.0.0.1",
    )

    def srv_handler():
        server.accept(timeout_seconds=5.0)
        env = server.receive(timeout_seconds=5.0)
        if env:
            resp = WireEnvelope(
                protocol_version="31.0",
                message_type=MessageType.HEARTBEAT,
                message_id="resp-tls-1",
                session_id=env.session_id,
                sender_peer_id="srv-node",
                receiver_peer_id="cli-node",
                created_epoch=1,
                expires_epoch=10,
                payload={"pong": True},
            )
            server.send(resp)

    th = threading.Thread(target=srv_handler, daemon=True)
    th.start()

    time.sleep(0.05)
    connected = client.connect(f"tcp://127.0.0.1:{port}")
    assert connected is True

    # Verify server certificate was extracted on client
    assert client.peer_certificate_metadata is not None
    assert client.peer_certificate_metadata.subject.get("commonName") == "127.0.0.1"

    req = WireEnvelope(
        protocol_version="31.0",
        message_type=MessageType.HEARTBEAT,
        message_id="req-tls-1",
        session_id="sess-tls-1",
        sender_peer_id="cli-node",
        receiver_peer_id="srv-node",
        created_epoch=1,
        expires_epoch=10,
        payload={"ping": True},
    )
    client.send(req)
    reply = client.receive(timeout_seconds=5.0)
    assert reply is not None
    assert reply.payload == {"pong": True}

    th.join(timeout=2.0)
    server.close()
    client.close()


def test_tcp_wire_transport_mtls_round_trip(pki_bundle):
    """Test mutual TLS (both client and server present certificates)."""
    server_policy = SecureTransportPolicy(tls_mode=TLSMode.MTLS, allow_tls_1_2=True)
    client_policy = SecureTransportPolicy(tls_mode=TLSMode.MTLS, allow_tls_1_2=True)

    server = TCPWireTransport(
        endpoint="tcp://127.0.0.1:0",
        security_policy=server_policy,
        server_cert_pem=pki_bundle["srv_pem"],
        server_key_pem=pki_bundle["srv_key_pem"],
        ca_cert_pem=pki_bundle["ca_pem"],
    )
    server.listen()
    port = server.bound_port

    client = TCPWireTransport(
        endpoint=f"tcp://127.0.0.1:{port}",
        security_policy=client_policy,
        client_cert_pem=pki_bundle["cli_pem"],
        client_key_pem=pki_bundle["cli_key_pem"],
        ca_cert_pem=pki_bundle["ca_pem"],
        server_hostname="127.0.0.1",
    )

    def srv_handler():
        server.accept(timeout_seconds=5.0)
        env = server.receive(timeout_seconds=5.0)
        if env:
            resp = WireEnvelope(
                protocol_version="31.0",
                message_type=MessageType.HEARTBEAT,
                message_id="resp-mtls-1",
                session_id=env.session_id,
                sender_peer_id="srv-node",
                receiver_peer_id="cli-node",
                created_epoch=1,
                expires_epoch=10,
                payload={"mtls_status": "AUTHENTICATED"},
            )
            server.send(resp)

    th = threading.Thread(target=srv_handler, daemon=True)
    th.start()

    time.sleep(0.05)
    connected = client.connect(f"tcp://127.0.0.1:{port}")
    assert connected is True

    # Client has server certificate metadata
    assert client.peer_certificate_metadata is not None
    assert client.peer_certificate_metadata.subject.get("commonName") == "127.0.0.1"

    req = WireEnvelope(
        protocol_version="31.0",
        message_type=MessageType.HEARTBEAT,
        message_id="req-mtls-1",
        session_id="sess-mtls-1",
        sender_peer_id="cli-node",
        receiver_peer_id="srv-node",
        created_epoch=1,
        expires_epoch=10,
        payload={"auth": "mtls_handshake"},
    )
    client.send(req)
    reply = client.receive(timeout_seconds=5.0)
    assert reply is not None
    assert reply.payload == {"mtls_status": "AUTHENTICATED"}

    th.join(timeout=2.0)

    # Server has client certificate metadata
    assert server.peer_certificate_metadata is not None
    assert server.peer_certificate_metadata.subject.get("commonName") == "client-peer"

    server.close()
    client.close()


def test_tcp_wire_transport_mtls_rejects_missing_client_cert(pki_bundle):
    """Server in mTLS mode rejects connection if client does not present a certificate."""
    server_policy = SecureTransportPolicy(tls_mode=TLSMode.MTLS, allow_tls_1_2=True)
    # Client in standard TLS mode (no client cert)
    client_policy = SecureTransportPolicy(tls_mode=TLSMode.TLS, allow_tls_1_2=True)

    server = TCPWireTransport(
        endpoint="tcp://127.0.0.1:0",
        security_policy=server_policy,
        server_cert_pem=pki_bundle["srv_pem"],
        server_key_pem=pki_bundle["srv_key_pem"],
        ca_cert_pem=pki_bundle["ca_pem"],
    )
    server.listen()
    port = server.bound_port

    client = TCPWireTransport(
        endpoint=f"tcp://127.0.0.1:{port}",
        security_policy=client_policy,
        ca_cert_pem=pki_bundle["ca_pem"],
        server_hostname="127.0.0.1",
    )

    server_errors = []
    def srv_handler():
        try:
            server.accept(timeout_seconds=5.0)
        except Exception as e:
            server_errors.append(e)

    th = threading.Thread(target=srv_handler, daemon=True)
    th.start()

    time.sleep(0.05)
    # Client will either fail connect or server will abort handshake
    try:
        client.connect(f"tcp://127.0.0.1:{port}")
        client.send(WireEnvelope(
            protocol_version="31.0",
            message_type=MessageType.HEARTBEAT,
            message_id="msg-err",
            session_id="sess-err",
            sender_peer_id="cli",
            receiver_peer_id="srv",
            created_epoch=1,
            expires_epoch=10,
            payload={},
        ))
    except Exception:
        pass

    th.join(timeout=2.0)
    assert len(server_errors) > 0 or client._status.value == "DISCONNECTED"

    server.close()
    client.close()


def test_tcp_wire_transport_rejects_untrusted_ca(pki_bundle):
    """Client with trusted CA rejects server signed by an untrusted rogue CA."""
    # Create a rogue untrusted CA
    rogue_ca_pem, rogue_ca_key_pem, rogue_ca_cert, rogue_ca_key = HermeticPKIBuilder.create_ca(
        common_name="Rogue Untrusted CA"
    )
    rogue_srv_pem, rogue_srv_key_pem, _, _ = HermeticPKIBuilder.create_server_cert(
        rogue_ca_cert, rogue_ca_key, common_name="127.0.0.1", san_ips=["127.0.0.1"]
    )

    server_policy = SecureTransportPolicy(tls_mode=TLSMode.TLS, allow_tls_1_2=True)
    server = TCPWireTransport(
        endpoint="tcp://127.0.0.1:0",
        security_policy=server_policy,
        server_cert_pem=rogue_srv_pem,
        server_key_pem=rogue_srv_key_pem,
    )
    server.listen()
    port = server.bound_port

    # Client only trusts legitimate CA from fixture
    client_policy = SecureTransportPolicy(tls_mode=TLSMode.TLS, allow_tls_1_2=True)
    client = TCPWireTransport(
        endpoint=f"tcp://127.0.0.1:{port}",
        security_policy=client_policy,
        ca_cert_pem=pki_bundle["ca_pem"],
        server_hostname="127.0.0.1",
    )

    def srv_handler():
        try:
            server.accept(timeout_seconds=5.0)
        except Exception:
            pass

    th = threading.Thread(target=srv_handler, daemon=True)
    th.start()

    time.sleep(0.05)
    with pytest.raises((UntrustedCAError, TLSHandshakeError)):
        client.connect(f"tcp://127.0.0.1:{port}")

    th.join(timeout=2.0)
    server.close()
    client.close()


def test_tcp_wire_transport_rejects_hostname_mismatch(pki_bundle):
    """Client rejects connection when server certificate SAN does not match expected hostname."""
    server_policy = SecureTransportPolicy(tls_mode=TLSMode.TLS, allow_tls_1_2=True)
    server = TCPWireTransport(
        endpoint="tcp://127.0.0.1:0",
        security_policy=server_policy,
        server_cert_pem=pki_bundle["srv_pem"],
        server_key_pem=pki_bundle["srv_key_pem"],
    )
    server.listen()
    port = server.bound_port

    client_policy = SecureTransportPolicy(tls_mode=TLSMode.TLS, allow_tls_1_2=True)
    client = TCPWireTransport(
        endpoint=f"tcp://127.0.0.1:{port}",
        security_policy=client_policy,
        ca_cert_pem=pki_bundle["ca_pem"],
        server_hostname="mismatched.node.domain",  # Server cert is for 127.0.0.1 / localhost
    )

    def srv_handler():
        try:
            server.accept(timeout_seconds=5.0)
        except Exception:
            pass

    th = threading.Thread(target=srv_handler, daemon=True)
    th.start()

    time.sleep(0.05)
    with pytest.raises((HostnameMismatchError, TLSHandshakeError)):
        client.connect(f"tcp://127.0.0.1:{port}")

    th.join(timeout=2.0)
    server.close()
    client.close()


# ============================================================================
# 5. HTTP/2 and gRPC Adapter Security Boundary Tests
# ============================================================================

def test_http2_adapter_security_policy():
    policy = SecureTransportPolicy(tls_mode=TLSMode.TLS, allow_tls_1_2=True)
    transport = HTTP2WireTransport(security_policy=policy)
    caps = transport.capabilities()
    assert caps["supports_tls"] is True
    assert caps["supports_mtls"] is True
    assert caps["tls_mode"] == "TLS"

    if not is_http2_available():
        with pytest.raises(Exception):
            transport.connect("http2://127.0.0.1:8080")


def test_grpc_adapter_security_policy():
    policy = SecureTransportPolicy(tls_mode=TLSMode.MTLS, allow_tls_1_2=True)
    transport = GRPCWireTransport(security_policy=policy)
    caps = transport.capabilities()
    assert caps["supports_tls"] is True
    assert caps["supports_mtls"] is True
    assert caps["tls_mode"] == "MTLS"

    if not is_grpc_available():
        with pytest.raises(Exception):
            transport.connect("grpc://127.0.0.1:50051")


# ============================================================================
# 6. Security, CapabilityGate & Invariant Integration Tests
# ============================================================================

def test_cross_zone_engine_full_mtls_wire_capability_execution(pki_bundle):
    """
    Verify complete secure federation pipeline with mTLS certificate binding,
    Ed25519 signature, trust negotiation, and CapabilityGate mediation.
    """
    policy = CrossZoneFederationPolicy(
        allowed_zones={"zone-remote"},
        allowed_scopes={FederationScope.ALLOW_COGNITIVE_TASK_DELEGATION},
    )
    gate = create_test_gate()
    engine = CrossZoneFederationEngine(local_zone_id="zone-local", capability_gate=gate, policy=policy)

    remote_key = Ed25519PrivateKeyWrapper.generate()
    remote_crypto_id = CryptographicPeerIdentity.create(
        zone_id="zone-remote",
        organization_id="org-remote",
        public_key=remote_key.public_key(),
        peer_id="peer-remote",
    )
    engine.register_cryptographic_peer(remote_crypto_id)

    # Challenge-Response Authentication
    challenge = engine.issue_authentication_challenge(target_peer_id=remote_crypto_id.peer_id)
    auth_resp = ChallengeResponseAuthenticator.sign_challenge(
        challenge, remote_key, remote_crypto_id.peer_id, current_epoch=1
    )
    auth_ok, _, session = engine.verify_authentication_response(auth_resp)
    assert auth_ok is True
    assert session is not None

    # Trust Grant (Authentication != Trust)
    grant = TrustGrant(
        grant_id="grant_mtls_exec",
        issuer_zone_id="zone-local",
        subject_peer_id=remote_crypto_id.peer_id,
        subject_zone_id="zone-remote",
        trust_level=TrustLevel.FEDERATED,
        permitted_scopes=[FederationScope.ALLOW_COGNITIVE_TASK_DELEGATION],
        issued_epoch=1,
        expires_epoch=50,
    )
    engine.registry.update_trust_grant(remote_crypto_id.peer_id, grant)
    session.trust_grant_id = grant.grant_id

    # Bind peer to mTLS certificate fingerprint
    cli_cert_meta = extract_certificate_metadata(pki_bundle["cli_cert"])
    engine.bind_peer_certificate(
        peer_id="peer-remote",
        certificate_fingerprint=cli_cert_meta.fingerprint,
        expected_common_name="client-peer",
    )

    # Build signed capability request envelope
    envelope = WireEnvelope(
        protocol_version="31.0",
        message_type=MessageType.CAPABILITY_REQUEST,
        message_id="msg-cap-100",
        session_id=session.session_id,
        sender_peer_id="peer-remote",
        receiver_peer_id=engine.local_peer_id,
        created_epoch=1,
        expires_epoch=10,
        payload={
            "requested_scope": FederationScope.ALLOW_COGNITIVE_TASK_DELEGATION.value,
            "peer_zone_id": "zone-remote",
            "peer_tenant_id": "default_tenant",
            "target_tenant_id": "default_tenant",
            "capability_request": {
                "capability_id": "echo_task",
                "parameters": {"input": "Hello Secure Federation"},
            },
        },
    )
    envelope.sign(remote_key)

    cap_ctx = CapabilityContext(
        user_id="peer-remote",
        granted_permissions={"capability.compute.echo"},
    )

    # Execute with certificate metadata passed from mTLS transport
    resp_envelope = engine.authorize_and_execute_wire_envelope(
        envelope=envelope,
        capability_gate=gate,
        capability_context=cap_ctx,
        peer_certificate_metadata=cli_cert_meta,
    )

    assert resp_envelope.message_type == MessageType.CAPABILITY_RESPONSE
    assert resp_envelope.payload["authorized"] is True
    assert resp_envelope.verify_signature(engine.local_public_key) is True



def test_tls_peer_binding_mismatch_rejected_by_engine(pki_bundle):
    """
    Even with valid TLS certificate and valid Ed25519 signature, if the certificate
    fingerprint does not match the bound peer identity, execution must fail closed.
    """
    gate = create_test_gate()
    engine = CrossZoneFederationEngine(local_zone_id="zone-local", capability_gate=gate)

    remote_key = Ed25519PrivateKeyWrapper.generate()
    remote_crypto_id = CryptographicPeerIdentity.create(
        zone_id="zone-remote",
        organization_id="org-remote",
        public_key=remote_key.public_key(),
        peer_id="peer-remote-2",
    )
    engine.register_cryptographic_peer(remote_crypto_id)
    session = engine.create_secure_session("peer-remote-2")

    # Bind peer to a different fingerprint
    cli_cert_meta = extract_certificate_metadata(pki_bundle["cli_cert"])
    engine.bind_peer_certificate(
        peer_id="peer-remote-2",
        certificate_fingerprint="SHA256:00:11:22:33:44:55:66:77:88:99:AA:BB:CC:DD:EE:FF:00:11:22:33:44:55:66:77:88:99:AA:BB:CC:DD:EE:FF",
    )

    envelope = WireEnvelope(
        protocol_version="31.0",
        message_type=MessageType.HEARTBEAT,
        message_id="msg-hb-1",
        session_id=session.session_id,
        sender_peer_id="peer-remote-2",
        receiver_peer_id=engine.local_peer_id,
        created_epoch=1,
        expires_epoch=10,
        payload={},
    )
    envelope.sign(remote_key)

    with pytest.raises(PeerBindingMismatchError):
        engine.authorize_and_execute_wire_envelope(
            envelope=envelope,
            peer_certificate_metadata=cli_cert_meta,
        )


def test_tls_authenticated_peer_still_requires_trust_grant(pki_bundle):
    """
    INVARIANT: TLS != TRUST
    A valid TLS connection and certificate does NOT confer trust.
    A peer without an active trust grant must be rejected.
    """
    policy = CrossZoneFederationPolicy(allowed_zones={"zone-remote"})
    gate = create_test_gate()
    engine = CrossZoneFederationEngine(local_zone_id="zone-local", capability_gate=gate, policy=policy)

    remote_key = Ed25519PrivateKeyWrapper.generate()
    remote_crypto_id = CryptographicPeerIdentity.create(
        zone_id="zone-remote",
        organization_id="org-remote",
        public_key=remote_key.public_key(),
        peer_id="peer-untrusted",
    )
    engine.register_cryptographic_peer(remote_crypto_id)

    cli_cert_meta = extract_certificate_metadata(pki_bundle["cli_cert"])
    engine.bind_peer_certificate("peer-untrusted", cli_cert_meta.fingerprint)

    # Create session but DO NOT negotiate trust grant
    session = engine.create_secure_session("peer-untrusted")

    envelope = WireEnvelope(
        protocol_version="31.0",
        message_type=MessageType.CAPABILITY_REQUEST,
        message_id="msg-notrust-1",
        session_id=session.session_id,
        sender_peer_id="peer-untrusted",
        receiver_peer_id=engine.local_peer_id,
        created_epoch=1,
        expires_epoch=10,
        payload={
            "requested_scope": FederationScope.ALLOW_COGNITIVE_TASK_DELEGATION.value,
            "peer_zone_id": "zone-remote",
            "peer_tenant_id": "default_tenant",
            "target_tenant_id": "default_tenant",
            "capability_request": {
                "capability_id": "mock.echo",
                "parameters": {"msg": "No trust test"},
            },
        },
    )
    envelope.sign(remote_key)

    with pytest.raises(CrossZoneAuthorizationError, match="denied by trust grant"):
        engine.authorize_and_execute_wire_envelope(
            envelope=envelope,
            capability_gate=gate,
            peer_certificate_metadata=cli_cert_meta,
        )


def test_tls_authenticated_peer_tenant_isolation(pki_bundle):
    """
    INVARIANT: Identity != Tenant Authorization
    Even with valid mTLS, valid Ed25519 signature, and active trust,
    cross-tenant requests must fail closed.
    """
    policy = CrossZoneFederationPolicy(allowed_zones={"zone-remote"})
    gate = create_test_gate()
    engine = CrossZoneFederationEngine(local_zone_id="zone-local", capability_gate=gate, policy=policy)

    remote_key = Ed25519PrivateKeyWrapper.generate()
    remote_crypto_id = CryptographicPeerIdentity.create(
        zone_id="zone-remote",
        organization_id="org-remote",
        public_key=remote_key.public_key(),
        peer_id="peer-tenant-a",
    )
    engine.register_cryptographic_peer(remote_crypto_id)

    session = engine.create_secure_session("peer-tenant-a")

    grant = TrustGrant(
        grant_id="grant_tenant_iso",
        issuer_zone_id="zone-local",
        subject_peer_id=remote_crypto_id.peer_id,
        subject_zone_id="zone-remote",
        trust_level=TrustLevel.FEDERATED,
        permitted_scopes=[FederationScope.ALLOW_COGNITIVE_TASK_DELEGATION],
        issued_epoch=1,
        expires_epoch=50,
    )
    engine.registry.update_trust_grant(remote_crypto_id.peer_id, grant)
    session.trust_grant_id = grant.grant_id

    cli_cert_meta = extract_certificate_metadata(pki_bundle["cli_cert"])
    engine.bind_peer_certificate("peer-tenant-a", cli_cert_meta.fingerprint)

    # Sender is Tenant A, but attempts to target Tenant B
    envelope = WireEnvelope(
        protocol_version="31.0",
        message_type=MessageType.CAPABILITY_REQUEST,
        message_id="msg-tenant-cross",
        session_id=session.session_id,
        sender_peer_id="peer-tenant-a",
        receiver_peer_id=engine.local_peer_id,
        created_epoch=1,
        expires_epoch=10,
        payload={
            "requested_scope": FederationScope.ALLOW_COGNITIVE_TASK_DELEGATION.value,
            "peer_zone_id": "zone-remote",
            "peer_tenant_id": "tenant-a",
            "target_tenant_id": "tenant-b",
            "capability_request": {"capability_id": "mock.echo", "parameters": {}},
        },
    )
    envelope.sign(remote_key)

    cap_ctx = CapabilityContext(
        user_id="peer-tenant-a",
        granted_permissions={"capability.compute.echo"},
    )

    with pytest.raises((CrossZoneAuthorizationError, IsolationViolationError), match="Cross-tenant crossover"):
        engine.authorize_and_execute_wire_envelope(
            envelope=envelope,
            capability_gate=gate,
            capability_context=cap_ctx,
            peer_certificate_metadata=cli_cert_meta,
        )


# ============================================================================
# 7. Secret Safety & Leakage Review Tests
# ============================================================================

def test_private_keys_never_appear_in_repr_or_str():
    # Ed25519 private key wrapper
    ed_key = Ed25519PrivateKeyWrapper.generate()
    ed_repr = repr(ed_key)
    ed_str = str(ed_key)
    assert "REDACTED" in ed_repr
    assert "REDACTED" in ed_str
    with pytest.raises(PermissionError):
        ed_key.to_dict()

    # TLS private key PEM strings must never be included in CertificateMetadata
    ca_pem, ca_key_pem, ca_cert, _ = HermeticPKIBuilder.create_ca()
    meta = extract_certificate_metadata(ca_cert)
    meta_dict = meta.to_dict()
    meta_str = str(meta_dict)
    assert "private" not in meta_str.lower()
    assert "key_pem" not in meta_str.lower()
    assert "-----begin" not in meta_str.lower()


def test_private_keys_never_appear_in_audit_records():
    logger = CrossZoneFederationEngine(local_zone_id="test-zone").audit_logger
    logger.log(
        event_type=AuditEventType.PEER_BINDING_VERIFIED,
        epoch=1,
        peer_id="peer-test",
        details={"fingerprint": "SHA256:11:22:33"},
    )
    history = logger.get_recent(limit=10)
    for rec in history:
        rec_str = str(rec.to_dict()).lower()
        assert "private_key" not in rec_str
        assert "secret" not in rec_str


# ============================================================================
# 8. Neural Weight Invariance Test
# ============================================================================

def test_chakrmicro_neural_weight_invariance_step31():
    config = ModelConfig()
    model = ChakrMicro(config)
    model.eval()

    integrity = CoreIntegrityGuard.verify_model(model)
    assert integrity.passed is True
    assert integrity.parameter_count == 3_443_136
    assert integrity.vocab_size == 4096
    assert integrity.max_seq_len == 512

    pre_hash = CoreIntegrityGuard.compute_weight_fingerprint(model)

    # Instantiate engine with neural core
    gate = create_test_gate()
    engine = CrossZoneFederationEngine(
        local_zone_id="zone-local",
        capability_gate=gate,
        model=model,
    )

    # Perform wire operations
    remote_key = Ed25519PrivateKeyWrapper.generate()
    remote_crypto_id = CryptographicPeerIdentity.create(
        zone_id="zone-remote",
        organization_id="org-remote",
        public_key=remote_key.public_key(),
        peer_id="peer-remote-inv",
    )
    engine.register_cryptographic_peer(remote_crypto_id)

    session = engine.create_secure_session("peer-remote-inv")

    envelope = WireEnvelope(
        protocol_version="31.0",
        message_type=MessageType.HEARTBEAT,
        message_id="msg-inv-1",
        session_id=session.session_id,
        sender_peer_id="peer-remote-inv",
        receiver_peer_id=engine.local_peer_id,
        created_epoch=1,
        expires_epoch=10,
        payload={},
    )
    envelope.sign(remote_key)

    resp = engine.authorize_and_execute_wire_envelope(envelope=envelope)
    assert resp.message_type == MessageType.HEARTBEAT
    assert resp.payload["status"] == "PONG"

    post_hash = CoreIntegrityGuard.compute_weight_fingerprint(model)
    assert pre_hash == post_hash, "Neural weights mutated during Step 31 transport security operations! ΔW != 0."
