"""
Unit, Integration, and Invariant Tests for Secure Physical Transport &
Cryptographic Peer Identity (Step 30).

Validates:
1. Ed25519 Cryptographic Keypair Generation, Signing, and Verification
2. Zero Private Key Leakage (No serialization, no dict export, redacted repr)
3. Cryptographic Peer Identity Model, Lifecycle, and Key Rotation
4. Bounded Challenge-Response Peer Authentication with Nonces and Replay Protection
5. Secure Peer Session Model, Epoch Expiration, and Bounded Replay Cache
6. WireEnvelope Schema, Bounded Payloads, SHA-256 Digest & Ed25519 Signatures
7. Deterministic Canonical Wire Serialization and Deserialization (No Pickle)
8. Length-Prefixed Binary Framing (Header validation, truncation, oversized rejection)
9. Loopback and Physical TCP Transport Lifecycle (Connect, send, receive, close)
10. HTTP/2 and gRPC Dependency Adapter Boundaries
11. End-to-End Engine Wire Envelope Execution with CapabilityGate Mediation
12. Security Invariants:
    - AUTHENTICATION != AUTHORIZATION
    - AUTHENTICATION != TRUST
    - TRANSPORT != AUTHORITY
    - SIGNATURE_VALIDITY != CAPABILITY_PERMISSION
    - Cross-tenant isolation enforcement
    - Replay attack rejection
    - Revoked peer and key rejection
    - Zero neural weight mutation (ΔW = 0)
    - Frozen ChakrMicro invariants (3,443,136 params, 4096 vocab, 512 context)
"""

import hashlib
import json
import time
import pytest
import torch

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.cognition.diagnostics.integrity import CoreIntegrityGuard

from chakrview.cognition.peering.crypto import (
    Ed25519PrivateKeyWrapper,
    Ed25519PublicKeyWrapper,
    CryptographicPeerIdentity,
    KeyLifecycleState,
    SignatureVerificationError,
    KeyStateError,
)
from chakrview.cognition.peering.authentication import (
    AuthChallenge,
    AuthChallengeResponse,
    ChallengeResponseAuthenticator,
    PeerAuthenticationState,
    AuthenticationError,
    ReplayedChallengeError,
)
from chakrview.cognition.peering.session import (
    SecurePeerSession,
    SessionStatus,
)
from chakrview.cognition.peering.models import (
    FederationScope,
    ProhibitedScope,
    TrustLevel,
    TrustGrant,
    TrustStatus,
    DiscoveryStatus,
    AuditEventType,
)
from chakrview.cognition.peering.policy import CrossZoneFederationPolicy
from chakrview.cognition.peering.engine import (
    CrossZoneFederationEngine,
    CrossZoneAuthorizationError,
    WeightMutationDetectedError,
)

from chakrview.cognition.transport.models import (
    WireEnvelope,
    MessageType,
    TransportStatus,
    DEFAULT_MAX_PAYLOAD_BYTES,
    MAX_WIRE_FRAME_BYTES,
)
from chakrview.cognition.transport.framing import LengthPrefixedFramer
from chakrview.cognition.transport.serialization import DeterministicWireSerializer
from chakrview.cognition.transport.loopback import LoopbackWireTransport
from chakrview.cognition.transport.tcp import TCPWireTransport
from chakrview.cognition.transport.http2 import HTTP2WireTransport, is_http2_available
from chakrview.cognition.transport.grpc import GRPCWireTransport, is_grpc_available
from chakrview.cognition.transport.registry import TransportRegistry
from chakrview.cognition.transport.errors import (
    TransportProtocolError,
    OversizedPayloadError,
    FrameError,
    ReplayAttackError,
    TransportUnavailableError,
    TransportTimeoutError,
)

from chakrview.capability.registry import CapabilityRegistry
from chakrview.capability.gate import CapabilityGate, CapabilityAuthorizationError
from chakrview.capability.contract import (
    Capability,
    CapabilityDescriptor,
    CapabilityRequest,
    CapabilityResult,
    CapabilityContext,
    RiskClassification,
)
from chakrview.cognition.peering.isolation import IsolationViolationError


class MockEchoCapability(Capability):
    """Simple safe capability for test verification."""

    @property
    def descriptor(self) -> CapabilityDescriptor:
        return CapabilityDescriptor(
            capability_id="mock.echo",
            name="Mock Echo",
            description="Echoes input back to caller",
            risk_level=RiskClassification.READ_ONLY,
            required_permissions=["capability.compute.echo"],
        )

    def execute(self, request: CapabilityRequest, context: Optional[CapabilityContext] = None) -> CapabilityResult:
        return CapabilityResult(
            request_id=request.request_id,
            capability_id="mock.echo",
            success=True,
            output={"echo": request.parameters.get("msg", "none")},
        )


# ============================================================================
# 1. Cryptographic Primitive Tests (Ed25519)
# ============================================================================

def test_ed25519_keypair_generation_and_properties():
    priv = Ed25519PrivateKeyWrapper.generate()
    pub = priv.public_key()

    assert len(pub.public_bytes) == 32
    assert len(pub.public_hex) == 64
    assert len(pub.fingerprint) == 64
    assert pub.fingerprint == hashlib.sha256(pub.public_bytes).hexdigest()

    # Prohibit private key leakage
    assert "REDACTED" in repr(priv)
    assert "REDACTED" in str(priv)
    with pytest.raises(PermissionError, match="Private keys must never be serialized"):
        priv.to_dict()


def test_ed25519_signing_and_verification():
    priv = Ed25519PrivateKeyWrapper.generate()
    pub = priv.public_key()

    msg = b"Cross-Zone Federation Message Payload"
    sig = priv.sign(msg)
    assert len(sig) == 64

    # Valid verification
    assert pub.verify(sig, msg) is True

    # Tampered message
    assert pub.verify(sig, b"Tampered Message Payload") is False

    # Tampered signature
    tampered_sig = bytearray(sig)
    tampered_sig[0] ^= 0xFF
    assert pub.verify(bytes(tampered_sig), msg) is False

    # Wrong public key
    other_priv = Ed25519PrivateKeyWrapper.generate()
    assert other_priv.public_key().verify(sig, msg) is False


def test_ed25519_seed_determinism():
    seed = b"\x01" * 32
    priv1 = Ed25519PrivateKeyWrapper.from_seed(seed)
    priv2 = Ed25519PrivateKeyWrapper.from_seed(seed)

    assert priv1.public_key().public_hex == priv2.public_key().public_hex
    assert priv1.public_key().fingerprint == priv2.public_key().fingerprint

    msg = b"Deterministic signature test"
    assert priv1.sign(msg) == priv2.sign(msg)


# ============================================================================
# 2. Cryptographic Peer Identity Model Tests
# ============================================================================

def test_cryptographic_peer_identity_lifecycle():
    priv = Ed25519PrivateKeyWrapper.generate()
    pub = priv.public_key()

    identity = CryptographicPeerIdentity.create(
        zone_id="zone-eu-1",
        organization_id="org-acme",
        public_key=pub,
        created_epoch=1,
        ttl_epochs=10,
    )

    assert identity.peer_id == f"peer_{pub.fingerprint[:16]}"
    assert identity.key_state == KeyLifecycleState.ACTIVE

    # Active at epoch 5
    is_valid, msg = identity.is_valid(5)
    assert is_valid is True

    # Expired at epoch 15
    is_valid, msg = identity.is_valid(15)
    assert is_valid is False
    assert "EXPIRED" in msg

    # Revocation
    rec = identity.revoke(reason="Key compromised", revoked_epoch=8)
    assert identity.key_state == KeyLifecycleState.REVOKED
    assert rec.key_fingerprint == pub.fingerprint
    is_valid, msg = identity.is_valid(5)
    assert is_valid is False
    assert "REVOKED" in msg


def test_cryptographic_peer_identity_key_rotation():
    priv_old = Ed25519PrivateKeyWrapper.generate()
    priv_new = Ed25519PrivateKeyWrapper.generate()

    identity = CryptographicPeerIdentity.create(
        zone_id="zone-ap-1",
        organization_id="org-acme",
        public_key=priv_old.public_key(),
        created_epoch=1,
    )

    # Valid rotation proof
    proof_payload = f"ROTATE_KEY:{priv_old.public_key().fingerprint}:{priv_new.public_key().fingerprint}:5".encode("utf-8")
    sig = priv_old.sign_hex(proof_payload)

    identity.rotate_key(priv_new.public_key(), rotation_epoch=5, signature_from_old_key=sig)
    assert identity.public_key == priv_new.public_key()
    assert len(identity.rotation_history) == 1

    # Invalid rotation signature
    invalid_sig = priv_new.sign_hex(b"wrong payload")
    priv_third = Ed25519PrivateKeyWrapper.generate()
    with pytest.raises(SignatureVerificationError):
        identity.rotate_key(priv_third.public_key(), rotation_epoch=6, signature_from_old_key=invalid_sig)


# ============================================================================
# 3. Challenge-Response Authentication Tests
# ============================================================================

def test_challenge_response_authentication_flow():
    authenticator = ChallengeResponseAuthenticator()
    priv = Ed25519PrivateKeyWrapper.generate()
    identity = CryptographicPeerIdentity.create(
        zone_id="zone-b",
        organization_id="org-b",
        public_key=priv.public_key(),
    )

    # Issue challenge
    challenge = authenticator.issue_challenge(
        session_id="sess_101",
        challenger_peer_id="peer_local",
        target_peer_id=identity.peer_id,
        current_epoch=1,
        ttl_epochs=3,
    )

    assert challenge.session_id == "sess_101"
    assert challenge.target_peer_id == identity.peer_id
    assert len(challenge.nonce) == 64

    # Target peer signs challenge
    response = ChallengeResponseAuthenticator.sign_challenge(
        challenge=challenge,
        private_key=priv,
        signer_peer_id=identity.peer_id,
        current_epoch=2,
    )

    # Verify response
    is_valid, msg = authenticator.verify_response(response, identity, current_epoch=2)
    assert is_valid is True
    assert "successfully authenticated" in msg


def test_challenge_response_one_time_consumption_and_replay_rejection():
    authenticator = ChallengeResponseAuthenticator()
    priv = Ed25519PrivateKeyWrapper.generate()
    identity = CryptographicPeerIdentity.create(
        zone_id="zone-b",
        organization_id="org-b",
        public_key=priv.public_key(),
    )

    challenge = authenticator.issue_challenge("sess_101", "peer_local", identity.peer_id, current_epoch=1)
    response = ChallengeResponseAuthenticator.sign_challenge(challenge, priv, identity.peer_id, current_epoch=1)

    # First verification succeeds
    ok1, _ = authenticator.verify_response(response, identity, current_epoch=1)
    assert ok1 is True

    # Replay of the same challenge response MUST fail (one-time consumption)
    ok2, err2 = authenticator.verify_response(response, identity, current_epoch=1)
    assert ok2 is False
    assert "not found or already consumed" in err2


def test_challenge_response_expired_challenge():
    authenticator = ChallengeResponseAuthenticator()
    priv = Ed25519PrivateKeyWrapper.generate()
    identity = CryptographicPeerIdentity.create(
        zone_id="zone-b",
        organization_id="org-b",
        public_key=priv.public_key(),
    )

    challenge = authenticator.issue_challenge("sess_101", "peer_local", identity.peer_id, current_epoch=1, ttl_epochs=2)
    response = ChallengeResponseAuthenticator.sign_challenge(challenge, priv, identity.peer_id, current_epoch=5)

    # At epoch 5, challenge (expires at 3) is expired
    ok, err = authenticator.verify_response(response, identity, current_epoch=5)
    assert ok is False
    assert "expired" in err.lower()


def test_challenge_response_wrong_session_or_wrong_peer():
    authenticator = ChallengeResponseAuthenticator()
    priv = Ed25519PrivateKeyWrapper.generate()
    identity = CryptographicPeerIdentity.create(
        zone_id="zone-b",
        organization_id="org-b",
        public_key=priv.public_key(),
    )

    challenge = authenticator.issue_challenge("sess_101", "peer_local", identity.peer_id, current_epoch=1)

    # Response with wrong session_id
    wrong_sess_resp = AuthChallengeResponse(
        challenge_id=challenge.challenge_id,
        session_id="wrong_session",
        signer_peer_id=identity.peer_id,
        signature_hex=priv.sign_hex(challenge.canonical_payload()),
        response_epoch=1,
    )
    ok, err = authenticator.verify_response(wrong_sess_resp, identity, current_epoch=1)
    assert ok is False
    assert "Session mismatch" in err


# ============================================================================
# 4. Secure Peer Session Model Tests
# ============================================================================

def test_secure_peer_session_replay_tracking():
    session = SecurePeerSession(
        session_id="sess_abc",
        local_peer_id="peer_local",
        remote_peer_id="peer_remote",
        local_zone_id="zone-a",
        remote_zone_id="zone-b",
        created_epoch=1,
        expires_at_epoch=10,
        max_message_history=5,
    )
    session.status = SessionStatus.ACTIVE

    # Fresh messages
    assert session.record_and_check_message_id("msg_1") is True
    assert session.record_and_check_message_id("msg_2") is True
    assert session.record_and_check_message_id("msg_3") is True

    # Replayed message
    assert session.record_and_check_message_id("msg_1") is False

    # Capacity pruning
    assert session.record_and_check_message_id("msg_4") is True
    assert session.record_and_check_message_id("msg_5") is True
    assert session.record_and_check_message_id("msg_6") is True  # msg_1 pruned from FIFO

    assert session.to_dict()["replay_cache_size"] <= 5


# ============================================================================
# 5. WireEnvelope & Framing Tests
# ============================================================================

def test_wire_envelope_signing_and_digest_validation():
    priv = Ed25519PrivateKeyWrapper.generate()
    envelope = WireEnvelope(
        protocol_version="30.0",
        message_type=MessageType.CAPABILITY_REQUEST,
        message_id="msg_101",
        session_id="sess_101",
        sender_peer_id="peer_alice",
        receiver_peer_id="peer_bob",
        created_epoch=1,
        expires_epoch=10,
        payload={"task": "echo", "args": {"value": 42}},
    )
    envelope.sign(priv)

    assert envelope.verify_digest() is True
    assert envelope.verify_signature(priv.public_key()) is True

    # Tampered payload fails digest
    envelope.payload["args"]["value"] = 999
    assert envelope.verify_digest() is False


def test_deterministic_wire_serializer_canonical_json():
    priv = Ed25519PrivateKeyWrapper.generate()
    envelope = WireEnvelope(
        protocol_version="30.0",
        message_type=MessageType.HEARTBEAT,
        message_id="msg_hb",
        session_id="sess_1",
        sender_peer_id="peer_a",
        receiver_peer_id="peer_b",
        created_epoch=1,
        expires_epoch=5,
        payload={"status": "PING"},
    )
    envelope.sign(priv)

    serialized = DeterministicWireSerializer.serialize(envelope)
    assert isinstance(serialized, bytes)

    deserialized = DeterministicWireSerializer.deserialize(serialized)
    assert deserialized.message_id == envelope.message_id
    assert deserialized.session_id == envelope.session_id
    assert deserialized.payload_digest == envelope.payload_digest
    assert deserialized.signature == envelope.signature


def test_wire_envelope_oversized_payload_rejection():
    priv = Ed25519PrivateKeyWrapper.generate()
    large_payload = {"data": "A" * (DEFAULT_MAX_PAYLOAD_BYTES + 100)}
    envelope = WireEnvelope(
        protocol_version="30.0",
        message_type=MessageType.EVIDENCE_EXCHANGE,
        message_id="msg_large",
        session_id="sess_1",
        sender_peer_id="peer_a",
        receiver_peer_id="peer_b",
        created_epoch=1,
        expires_epoch=5,
        payload=large_payload,
    )
    with pytest.raises(OversizedPayloadError):
        envelope.validate()


def test_length_prefixed_framing_encode_decode():
    data = b'{"msg": "test_wire_frame"}'
    framed = LengthPrefixedFramer.encode_frame(data)

    assert len(framed) == 4 + len(data)

    buf = bytearray(framed)
    extracted = LengthPrefixedFramer.decode_frame(buf)
    assert extracted == data
    assert len(buf) == 0  # Buffer fully consumed

    # Partial frame returns None
    partial = framed[:6]
    buf_partial = bytearray(partial)
    assert LengthPrefixedFramer.decode_frame(buf_partial) is None
    assert len(buf_partial) == 6  # Preserved in buffer


# ============================================================================
# 6. Transport Implementations (Loopback & TCP)
# ============================================================================

def test_loopback_transport_lifecycle():
    LoopbackWireTransport.reset_all()

    server = LoopbackWireTransport(endpoint="loopback://zone-a")
    server.listen("loopback://zone-a")

    client = LoopbackWireTransport(endpoint="loopback://zone-b")
    client.connect("loopback://zone-a")

    envelope = WireEnvelope(
        protocol_version="30.0",
        message_type=MessageType.HEARTBEAT,
        message_id="msg_lb_1",
        session_id="sess_lb",
        sender_peer_id="peer_b",
        receiver_peer_id="peer_a",
        created_epoch=1,
        expires_epoch=5,
        payload={"ping": 1},
    )

    # Client sends, server receives
    assert client.send(envelope) is True
    recv = server.receive(timeout_seconds=1.0)
    assert recv is not None
    assert recv.message_id == "msg_lb_1"

    health = server.health()
    assert health.is_healthy is True
    assert health.messages_received == 1

    server.close()
    client.close()


def test_tcp_transport_socket_lifecycle():
    # Server listens on dynamic port on 127.0.0.1
    server = TCPWireTransport(endpoint="tcp://127.0.0.1:0")
    server.listen()
    server_port = server.bound_port
    assert server_port is not None and server_port > 0

    client = TCPWireTransport()
    connected = client.connect(f"tcp://127.0.0.1:{server_port}", timeout_seconds=2.0)
    assert connected is True

    accepted = server.accept(timeout_seconds=2.0)
    assert accepted is not None

    priv = Ed25519PrivateKeyWrapper.generate()
    envelope = WireEnvelope(
        protocol_version="30.0",
        message_type=MessageType.HEARTBEAT,
        message_id="msg_tcp_1",
        session_id="sess_tcp",
        sender_peer_id="peer_client",
        receiver_peer_id="peer_server",
        created_epoch=1,
        expires_epoch=5,
        payload={"ping": "tcp"},
    )
    envelope.sign(priv)

    # Client sends to server
    assert client.send(envelope, timeout_seconds=2.0) is True

    received = server.receive(timeout_seconds=2.0)
    assert received is not None
    assert received.message_id == "msg_tcp_1"
    assert received.payload["ping"] == "tcp"

    # Server sends response
    resp = WireEnvelope(
        protocol_version="30.0",
        message_type=MessageType.HEARTBEAT,
        message_id="msg_tcp_resp",
        session_id="sess_tcp",
        sender_peer_id="peer_server",
        receiver_peer_id="peer_client",
        created_epoch=1,
        expires_epoch=5,
        payload={"pong": "tcp"},
    )
    resp.sign(priv)
    assert server.send(resp, timeout_seconds=2.0) is True

    client_recv = client.receive(timeout_seconds=2.0)
    assert client_recv is not None
    assert client_recv.message_id == "msg_tcp_resp"

    client.close()
    server.close()


def test_transport_registry_and_dependency_checks():
    schemes = TransportRegistry.list_supported_schemes()
    assert "loopback" in schemes
    assert "tcp" in schemes
    assert "http2" in schemes
    assert "grpc" in schemes

    assert schemes["loopback"] is True
    assert schemes["tcp"] is True

    # HTTP/2 and gRPC fail closed if libraries are absent
    http2_transport = HTTP2WireTransport()
    if not is_http2_available():
        assert http2_transport.is_available is False
        with pytest.raises(TransportUnavailableError):
            http2_transport.connect("http2://127.0.0.1:8080")

    grpc_transport = GRPCWireTransport()
    if not is_grpc_available():
        assert grpc_transport.is_available is False
        with pytest.raises(TransportUnavailableError):
            grpc_transport.connect("grpc://127.0.0.1:50051")


# ============================================================================
# 7. End-to-End Engine & Security Invariant Tests
# ============================================================================

def test_engine_end_to_end_wire_envelope_execution():
    policy = CrossZoneFederationPolicy(
        allowed_zones={"zone-peer"},
        allowed_scopes={FederationScope.ALLOW_COGNITIVE_TASK_DELEGATION},
    )
    registry = CapabilityRegistry()
    registry.register(MockEchoCapability())
    gate = CapabilityGate(registry=registry)

    engine = CrossZoneFederationEngine(
        local_zone_id="zone-local",
        policy=policy,
        capability_gate=gate,
    )

    # 1. Register remote peer with cryptographic identity
    remote_priv = Ed25519PrivateKeyWrapper.generate()
    remote_identity = CryptographicPeerIdentity.create(
        zone_id="zone-peer",
        organization_id="org-remote",
        public_key=remote_priv.public_key(),
    )
    engine.register_cryptographic_peer(remote_identity)

    # 2. Challenge-Response Authentication
    challenge = engine.issue_authentication_challenge(target_peer_id=remote_identity.peer_id)
    auth_resp = ChallengeResponseAuthenticator.sign_challenge(challenge, remote_priv, remote_identity.peer_id, current_epoch=1)
    auth_ok, _, session = engine.verify_authentication_response(auth_resp)
    assert auth_ok is True
    assert session is not None

    # 3. Negotiate Trust Grant (Authentication != Trust)
    grant = TrustGrant(
        grant_id="grant_delegation",
        issuer_zone_id="zone-local",
        subject_peer_id=remote_identity.peer_id,
        subject_zone_id="zone-peer",
        trust_level=TrustLevel.FEDERATED,
        permitted_scopes=[FederationScope.ALLOW_COGNITIVE_TASK_DELEGATION],
        issued_epoch=1,
        expires_epoch=50,
    )
    engine.registry.update_trust_grant(remote_identity.peer_id, grant)
    session.trust_grant_id = grant.grant_id

    # 4. Construct signed CAPABILITY_REQUEST WireEnvelope
    envelope = WireEnvelope(
        protocol_version="30.0",
        message_type=MessageType.CAPABILITY_REQUEST,
        message_id="msg_req_1",
        session_id=session.session_id,
        sender_peer_id=remote_identity.peer_id,
        receiver_peer_id=engine.local_peer_id,
        created_epoch=1,
        expires_epoch=10,
        payload={
            "requested_scope": FederationScope.ALLOW_COGNITIVE_TASK_DELEGATION.value,
            "peer_zone_id": "zone-peer",
            "peer_tenant_id": "tenant-corp",
            "target_tenant_id": "tenant-corp",
            "capability_request": {
                "capability_id": "mock.echo",
                "parameters": {"msg": "Hello Federated Wire"},
            },
            "data": {"context_info": "safe metadata"},
        },
    )
    envelope.sign(remote_priv)

    # CapabilityContext with granted permissions
    cap_ctx = CapabilityContext(
        user_id=remote_identity.peer_id,
        granted_permissions={"capability.compute.echo"},
    )

    # 5. Engine executes wire envelope
    resp_envelope = engine.authorize_and_execute_wire_envelope(
        envelope=envelope,
        capability_context=cap_ctx,
    )

    assert resp_envelope.message_type == MessageType.CAPABILITY_RESPONSE
    assert resp_envelope.verify_signature(engine.local_public_key) is True
    assert resp_envelope.payload["authorized"] is True


def test_engine_wire_envelope_replayed_message_rejected():
    policy = CrossZoneFederationPolicy(allowed_zones={"zone-peer"})
    engine = CrossZoneFederationEngine(local_zone_id="zone-local", policy=policy)

    remote_priv = Ed25519PrivateKeyWrapper.generate()
    remote_identity = CryptographicPeerIdentity.create(
        zone_id="zone-peer",
        organization_id="org-remote",
        public_key=remote_priv.public_key(),
    )
    engine.register_cryptographic_peer(remote_identity)
    session = engine.create_secure_session(remote_identity.peer_id)

    envelope = WireEnvelope(
        protocol_version="30.0",
        message_type=MessageType.HEARTBEAT,
        message_id="msg_replay_test",
        session_id=session.session_id,
        sender_peer_id=remote_identity.peer_id,
        receiver_peer_id=engine.local_peer_id,
        created_epoch=1,
        expires_epoch=10,
        payload={},
    )
    envelope.sign(remote_priv)

    # First presentation succeeds
    resp1 = engine.authorize_and_execute_wire_envelope(envelope)
    assert resp1.payload["status"] == "PONG"

    # Replay of the exact same envelope MUST raise ReplayAttackError
    with pytest.raises(ReplayAttackError, match="Replay attack detected"):
        engine.authorize_and_execute_wire_envelope(envelope)


def test_engine_wire_envelope_cross_tenant_isolation_violation():
    policy = CrossZoneFederationPolicy(
        allowed_zones={"zone-peer"},
        allowed_scopes={FederationScope.ALLOW_COGNITIVE_TASK_DELEGATION},
    )
    engine = CrossZoneFederationEngine(local_zone_id="zone-local", policy=policy)

    remote_priv = Ed25519PrivateKeyWrapper.generate()
    remote_identity = CryptographicPeerIdentity.create(
        zone_id="zone-peer",
        organization_id="org-remote",
        public_key=remote_priv.public_key(),
    )
    engine.register_cryptographic_peer(remote_identity)
    session = engine.create_secure_session(remote_identity.peer_id)

    # Tenant mismatch: peer from tenant-A attempts access to tenant-B in local zone
    envelope = WireEnvelope(
        protocol_version="30.0",
        message_type=MessageType.CAPABILITY_REQUEST,
        message_id="msg_cross_tenant",
        session_id=session.session_id,
        sender_peer_id=remote_identity.peer_id,
        receiver_peer_id=engine.local_peer_id,
        created_epoch=1,
        expires_epoch=10,
        payload={
            "requested_scope": FederationScope.ALLOW_COGNITIVE_TASK_DELEGATION.value,
            "peer_zone_id": "zone-peer",
            "peer_tenant_id": "tenant-A",
            "target_tenant_id": "tenant-B",  # Mismatch!
        },
    )
    envelope.sign(remote_priv)

    with pytest.raises(IsolationViolationError, match="Cross-tenant crossover denied"):
        engine.authorize_and_execute_wire_envelope(envelope)


def test_engine_wire_envelope_untrusted_peer_cannot_execute():
    # Peer is authenticated, but NO trust grant exists (AUTHENTICATION != TRUST)
    policy = CrossZoneFederationPolicy(allowed_zones={"zone-peer"})
    engine = CrossZoneFederationEngine(local_zone_id="zone-local", policy=policy)

    remote_priv = Ed25519PrivateKeyWrapper.generate()
    remote_identity = CryptographicPeerIdentity.create(
        zone_id="zone-peer",
        organization_id="org-remote",
        public_key=remote_priv.public_key(),
    )
    engine.register_cryptographic_peer(remote_identity)
    session = engine.create_secure_session(remote_identity.peer_id)

    envelope = WireEnvelope(
        protocol_version="30.0",
        message_type=MessageType.CAPABILITY_REQUEST,
        message_id="msg_no_trust",
        session_id=session.session_id,
        sender_peer_id=remote_identity.peer_id,
        receiver_peer_id=engine.local_peer_id,
        created_epoch=1,
        expires_epoch=10,
        payload={
            "requested_scope": FederationScope.ALLOW_COGNITIVE_TASK_DELEGATION.value,
            "peer_zone_id": "zone-peer",
            "peer_tenant_id": "tenant-1",
            "target_tenant_id": "tenant-1",
        },
    )
    envelope.sign(remote_priv)

    with pytest.raises(CrossZoneAuthorizationError, match="No trust grant present"):
        engine.authorize_and_execute_wire_envelope(envelope)


def test_engine_wire_envelope_revoked_key_rejection():
    policy = CrossZoneFederationPolicy(allowed_zones={"zone-peer"})
    engine = CrossZoneFederationEngine(local_zone_id="zone-local", policy=policy)

    remote_priv = Ed25519PrivateKeyWrapper.generate()
    remote_identity = CryptographicPeerIdentity.create(
        zone_id="zone-peer",
        organization_id="org-remote",
        public_key=remote_priv.public_key(),
    )
    engine.register_cryptographic_peer(remote_identity)
    session = engine.create_secure_session(remote_identity.peer_id)

    # Revoke key
    remote_identity.revoke(reason="Administrative key revocation", revoked_epoch=1)

    envelope = WireEnvelope(
        protocol_version="30.0",
        message_type=MessageType.HEARTBEAT,
        message_id="msg_revoked_key",
        session_id=session.session_id,
        sender_peer_id=remote_identity.peer_id,
        receiver_peer_id=engine.local_peer_id,
        created_epoch=1,
        expires_epoch=10,
        payload={},
    )
    envelope.sign(remote_priv)

    with pytest.raises(CrossZoneAuthorizationError, match="REVOKED"):
        engine.authorize_and_execute_wire_envelope(envelope)


# ============================================================================
# 8. Neural Core Invariance Tests
# ============================================================================

def test_frozen_neural_core_invariants_during_wire_operations():
    config = ModelConfig(
        vocab_size=4096,
        max_seq_len=512,
        n_layers=6,
        n_heads=6,
        d_model=192,
        hidden_dim=512,
    )
    model = ChakrMicro(config)
    model.eval()

    integrity = CoreIntegrityGuard.verify_model(model)
    assert integrity.passed is True
    assert integrity.parameter_count == 3_443_136
    assert integrity.vocab_size == 4096
    assert integrity.max_seq_len == 512

    pre_hash = CoreIntegrityGuard.compute_weight_fingerprint(model)

    engine = CrossZoneFederationEngine(
        local_zone_id="zone-neural-test",
        model=model,
    )

    remote_priv = Ed25519PrivateKeyWrapper.generate()
    remote_identity = CryptographicPeerIdentity.create(
        zone_id="zone-peer",
        organization_id="org-peer",
        public_key=remote_priv.public_key(),
    )
    engine.register_cryptographic_peer(remote_identity)
    session = engine.create_secure_session(remote_identity.peer_id)

    envelope = WireEnvelope(
        protocol_version="30.0",
        message_type=MessageType.HEARTBEAT,
        message_id="msg_neural_1",
        session_id=session.session_id,
        sender_peer_id=remote_identity.peer_id,
        receiver_peer_id=engine.local_peer_id,
        created_epoch=1,
        expires_epoch=10,
        payload={},
    )
    envelope.sign(remote_priv)

    resp = engine.authorize_and_execute_wire_envelope(envelope)
    assert resp.payload["status"] == "PONG"

    post_hash = CoreIntegrityGuard.compute_weight_fingerprint(model)
    assert pre_hash == post_hash, "FATAL: Neural core mutated during wire operations! ΔW != 0."


# ============================================================================
# 9. Additional Dedicated Wire & Security Invariant Tests
# ============================================================================

def test_challenge_response_invalid_signature_rejection():
    authenticator = ChallengeResponseAuthenticator()
    priv = Ed25519PrivateKeyWrapper.generate()
    wrong_priv = Ed25519PrivateKeyWrapper.generate()
    identity = CryptographicPeerIdentity.create(
        zone_id="zone-b",
        organization_id="org-b",
        public_key=priv.public_key(),
    )

    challenge = authenticator.issue_challenge("sess_wrong_sig", "peer_local", identity.peer_id, current_epoch=1)
    # Sign with wrong private key!
    bad_resp = ChallengeResponseAuthenticator.sign_challenge(challenge, wrong_priv, identity.peer_id, current_epoch=1)

    ok, err = authenticator.verify_response(bad_resp, identity, current_epoch=1)
    assert ok is False
    assert "signature verification failed" in err.lower()


def test_wire_envelope_expired_message_rejection():
    policy = CrossZoneFederationPolicy(allowed_zones={"zone-peer"})
    engine = CrossZoneFederationEngine(local_zone_id="zone-local", policy=policy, initial_epoch=15)

    remote_priv = Ed25519PrivateKeyWrapper.generate()
    remote_identity = CryptographicPeerIdentity.create(
        zone_id="zone-peer",
        organization_id="org-remote",
        public_key=remote_priv.public_key(),
    )
    engine.register_cryptographic_peer(remote_identity)
    session = engine.create_secure_session(remote_identity.peer_id, ttl_epochs=50)

    # Message created with expires_epoch=10, but engine is at epoch 15
    envelope = WireEnvelope(
        protocol_version="30.0",
        message_type=MessageType.HEARTBEAT,
        message_id="msg_expired_1",
        session_id=session.session_id,
        sender_peer_id=remote_identity.peer_id,
        receiver_peer_id=engine.local_peer_id,
        created_epoch=5,
        expires_epoch=10,  # Expired!
        payload={},
    )
    envelope.sign(remote_priv)

    with pytest.raises(CrossZoneAuthorizationError, match="Envelope expired"):
        engine.authorize_and_execute_wire_envelope(envelope)


def test_wire_envelope_wrong_sender_binding_rejection():
    policy = CrossZoneFederationPolicy(allowed_zones={"zone-peer"})
    engine = CrossZoneFederationEngine(local_zone_id="zone-local", policy=policy)

    remote_priv = Ed25519PrivateKeyWrapper.generate()
    remote_identity = CryptographicPeerIdentity.create(
        zone_id="zone-peer",
        organization_id="org-remote",
        public_key=remote_priv.public_key(),
    )
    engine.register_cryptographic_peer(remote_identity)
    session = engine.create_secure_session(remote_identity.peer_id)

    # Envelope claims sender is 'imposter_peer' instead of registered remote_peer_id
    envelope = WireEnvelope(
        protocol_version="30.0",
        message_type=MessageType.HEARTBEAT,
        message_id="msg_wrong_sender",
        session_id=session.session_id,
        sender_peer_id="imposter_peer",
        receiver_peer_id=engine.local_peer_id,
        created_epoch=1,
        expires_epoch=10,
        payload={},
    )
    envelope.sign(remote_priv)

    with pytest.raises(CrossZoneAuthorizationError, match="not registered"):
        engine.authorize_and_execute_wire_envelope(envelope)


def test_wire_envelope_deserialization_malformed_errors():
    with pytest.raises(TransportProtocolError, match="Cannot deserialize empty"):
        DeterministicWireSerializer.deserialize(b"")

    with pytest.raises(TransportProtocolError, match="Malformed JSON"):
        DeterministicWireSerializer.deserialize(b"{not-valid-json}")

    with pytest.raises(TransportProtocolError, match="missing required keys"):
        DeterministicWireSerializer.deserialize(b'{"protocol_version": "30.0"}')


def test_tcp_transport_connection_refused_and_timeout():
    client = TCPWireTransport()
    # Connect to invalid port on localhost
    with pytest.raises((TransportUnavailableError, TransportTimeoutError)):
        client.connect("tcp://127.0.0.1:59999", timeout_seconds=0.2)


def test_revoked_peer_in_registry_cannot_execute_over_wire():
    policy = CrossZoneFederationPolicy(allowed_zones={"zone-peer"})
    engine = CrossZoneFederationEngine(local_zone_id="zone-local", policy=policy)

    remote_priv = Ed25519PrivateKeyWrapper.generate()
    remote_identity = CryptographicPeerIdentity.create(
        zone_id="zone-peer",
        organization_id="org-remote",
        public_key=remote_priv.public_key(),
    )
    engine.register_cryptographic_peer(remote_identity)
    session = engine.create_secure_session(remote_identity.peer_id)

    # Administrate revocation of peer
    engine.revoke_peer(remote_identity.peer_id, reason="Policy violation audit")

    envelope = WireEnvelope(
        protocol_version="30.0",
        message_type=MessageType.HEARTBEAT,
        message_id="msg_revoked_peer",
        session_id=session.session_id,
        sender_peer_id=remote_identity.peer_id,
        receiver_peer_id=engine.local_peer_id,
        created_epoch=1,
        expires_epoch=10,
        payload={},
    )
    envelope.sign(remote_priv)

    with pytest.raises(CrossZoneAuthorizationError, match="revoked"):
        engine.authorize_and_execute_wire_envelope(envelope)
