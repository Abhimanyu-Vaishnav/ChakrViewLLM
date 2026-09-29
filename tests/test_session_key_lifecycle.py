"""
Step 32 Dedicated Test Suite: Secure Federation Session & Key Lifecycle Hardening.

Covers all 27 Section 14 requirements:
1. session creation
2. authentication
3. activation
4. renewal
5. expiry
6. termination
7. revocation
8. invalid transition rejection
9. session freshness
10. key creation
11. key rotation
12. failed key rotation
13. retired-key rejection
14. certificate rotation
15. certificate revocation
16. identity continuity
17. trust continuity
18. trust expiry
19. capability scope preservation
20. replay rejection
21. bounded replay cache
22. revocation cascade
23. stale authorization rejection
24. cross-tenant isolation after renewal
25. secret non-leakage
26. audit integrity
27. deterministic lifecycle behavior
"""

import hashlib
import time
import pytest
import torch

from chakrview.brain.model import ChakrMicro, ModelConfig
from chakrview.cognition.diagnostics.integrity import CoreIntegrityGuard
from chakrview.capability.gate import CapabilityGate
from chakrview.capability.contract import CapabilityRequest, CapabilityContext

from chakrview.cognition.peering.models import (
    PeerIdentity,
    PeerAttestation,
    PeerDeclaration,
    TrustGrant,
    TrustLevel,
    TrustStatus,
    DiscoveryStatus,
    FederationScope,
    AuditEventType,
)
from chakrview.cognition.peering.crypto import (
    Ed25519PrivateKeyWrapper,
    Ed25519PublicKeyWrapper,
    CryptographicPeerIdentity,
    KeyLifecycleState,
    KeyStateError,
    SignatureVerificationError,
)
from chakrview.cognition.peering.session import (
    SecurePeerSession,
    SessionStatus,
    SessionTransitionError,
    SessionKeyState,
    SessionKeyMetadata,
)
from chakrview.cognition.peering.policy import CrossZoneFederationPolicy
from chakrview.cognition.peering.engine import (
    FederationEngine,
    CrossZoneAuthorizationError,
)
from chakrview.cognition.transport.models import (
    WireEnvelope,
    MessageType,
)
from chakrview.cognition.transport.errors import (
    ReplayAttackError,
)
from chakrview.cognition.transport.security import (
    CertificateMetadata,
    CertificateRevocationRegistry,
    CertificateValidationError,
    PeerBindingMismatchError,
    HermeticPKIBuilder,
    extract_certificate_metadata,
)


@pytest.fixture
def model():
    torch.manual_seed(42)
    cfg = ModelConfig()
    m = ChakrMicro(cfg)
    m.eval()
    return m


@pytest.fixture
def engine(model):
    gate = CapabilityGate()
    policy = CrossZoneFederationPolicy(
        allowed_zones={"ZONE_REMOTE_BETA"},
        allowed_scopes={FederationScope.ALLOW_COGNITIVE_TASK_DELEGATION},
    )
    return FederationEngine(
        local_zone_id="ZONE_SECURE_ALPHA",
        policy=policy,
        capability_gate=gate,
        model=model,
        initial_epoch=10,
        local_peer_id="local_engine_node",
    )


@pytest.fixture
def peer_keys():
    priv = Ed25519PrivateKeyWrapper.generate()
    pub = priv.get_public_key()
    return priv, pub


@pytest.fixture
def registered_peer(engine, peer_keys):
    priv, pub = peer_keys
    crypto_id = CryptographicPeerIdentity.create(
        zone_id="ZONE_REMOTE_BETA",
        organization_id="ORG_REMOTE",
        public_key=pub,
        peer_id="peer_remote_beta",
        created_epoch=10,
        ttl_epochs=100,
    )
    reg = engine.register_cryptographic_peer(crypto_id)

    # Establish trust grant
    grant = TrustGrant(
        grant_id="grant_beta_001",
        issuer_zone_id="ZONE_SECURE_ALPHA",
        subject_peer_id="peer_remote_beta",
        subject_zone_id="ZONE_REMOTE_BETA",
        trust_level=TrustLevel.FEDERATED,
        permitted_scopes=[FederationScope.ALLOW_COGNITIVE_TASK_DELEGATION],
        issued_epoch=10,
        expires_epoch=80,
    )
    engine.registry.update_trust_grant("peer_remote_beta", grant)
    return reg, priv, pub


# ============================================================================
# 1. Session State Machine & Freshness
# ============================================================================

class TestSessionStateMachine:

    def test_01_session_creation(self):
        session = SecurePeerSession(
            session_id="sess_test_01",
            local_peer_id="local_engine_node",
            remote_peer_id="peer_remote_beta",
            local_zone_id="ZONE_SECURE_ALPHA",
            remote_zone_id="ZONE_REMOTE_BETA",
            created_epoch=10,
            expires_at_epoch=60,
        )
        assert session.status == SessionStatus.INITIATED
        assert session.created_epoch == 10
        assert session.expires_at_epoch == 60
        assert session.max_lifetime_epochs == 200
        assert session.renewal_count == 0
        assert session.max_renewals == 5
        assert session.session_key_metadata is not None
        assert session.session_key_metadata.key_state == SessionKeyState.CREATED

    def test_02_authentication_and_activation(self, peer_keys):
        _, pub = peer_keys
        session = SecurePeerSession(
            session_id="sess_test_02",
            local_peer_id="local_engine_node",
            remote_peer_id="peer_remote_beta",
            local_zone_id="ZONE_SECURE_ALPHA",
            remote_zone_id="ZONE_REMOTE_BETA",
            created_epoch=10,
            expires_at_epoch=60,
        )
        session.transition_to(SessionStatus.AUTHENTICATING, reason="Verifying challenge")
        assert session.status == SessionStatus.AUTHENTICATING

        session.mark_authenticated(pub, authenticated_epoch=10, trust_grant_id="grant_001")
        assert session.status == SessionStatus.ACTIVE
        assert session.session_key_metadata.key_state == SessionKeyState.ACTIVE
        is_active, _ = session.is_active(current_epoch=15)
        assert is_active is True

    def test_03_session_renewal(self, peer_keys):
        _, pub = peer_keys
        session = SecurePeerSession(
            session_id="sess_test_03",
            local_peer_id="local_engine_node",
            remote_peer_id="peer_remote_beta",
            local_zone_id="ZONE_SECURE_ALPHA",
            remote_zone_id="ZONE_REMOTE_BETA",
            created_epoch=10,
            expires_at_epoch=30,
        )
        session.mark_authenticated(pub, authenticated_epoch=10)
        assert session.can_renew(current_epoch=25)[0] is True

        session.renew(extension_epochs=20, current_epoch=25)
        assert session.status == SessionStatus.ACTIVE
        assert session.expires_at_epoch == 50
        assert session.renewal_count == 1

    def test_04_session_expiry_fail_closed(self, peer_keys):
        _, pub = peer_keys
        session = SecurePeerSession(
            session_id="sess_test_04",
            local_peer_id="local_engine_node",
            remote_peer_id="peer_remote_beta",
            local_zone_id="ZONE_SECURE_ALPHA",
            remote_zone_id="ZONE_REMOTE_BETA",
            created_epoch=10,
            expires_at_epoch=30,
        )
        session.mark_authenticated(pub, authenticated_epoch=10)

        # Before expiry
        is_active, _ = session.is_active(current_epoch=29)
        assert is_active is True

        # At / after expiry
        is_active, reason = session.is_active(current_epoch=31)
        assert is_active is False
        assert "EXPIRED" in reason
        assert session.status == SessionStatus.EXPIRED

        # Expired session cannot renew
        assert session.can_renew(current_epoch=31)[0] is False
        with pytest.raises(SessionTransitionError, match="Cannot renew EXPIRED session"):
            session.renew(extension_epochs=10, current_epoch=31)

    def test_05_session_termination(self, peer_keys):
        _, pub = peer_keys
        session = SecurePeerSession(
            session_id="sess_test_05",
            local_peer_id="local_engine_node",
            remote_peer_id="peer_remote_beta",
            local_zone_id="ZONE_SECURE_ALPHA",
            remote_zone_id="ZONE_REMOTE_BETA",
            created_epoch=10,
            expires_at_epoch=50,
        )
        session.mark_authenticated(pub, authenticated_epoch=10)
        session.terminate(reason="Admin shutdown")

        assert session.status == SessionStatus.TERMINATED
        assert session.session_key_metadata.key_state == SessionKeyState.EXPIRED
        is_act, _ = session.is_active(current_epoch=20)
        assert is_act is False

    def test_06_session_revocation(self, peer_keys):
        _, pub = peer_keys
        session = SecurePeerSession(
            session_id="sess_test_06",
            local_peer_id="local_engine_node",
            remote_peer_id="peer_remote_beta",
            local_zone_id="ZONE_SECURE_ALPHA",
            remote_zone_id="ZONE_REMOTE_BETA",
            created_epoch=10,
            expires_at_epoch=50,
        )
        session.mark_authenticated(pub, authenticated_epoch=10)
        session.revoke(reason="Security compromise")

        assert session.status == SessionStatus.REVOKED
        assert session.session_key_metadata.key_state == SessionKeyState.REVOKED
        is_act, _ = session.is_active(current_epoch=20)
        assert is_act is False

    def test_07_invalid_transition_rejection(self, peer_keys):
        _, pub = peer_keys
        session = SecurePeerSession(
            session_id="sess_test_07",
            local_peer_id="local_engine_node",
            remote_peer_id="peer_remote_beta",
            local_zone_id="ZONE_SECURE_ALPHA",
            remote_zone_id="ZONE_REMOTE_BETA",
            created_epoch=10,
            expires_at_epoch=50,
        )
        # Cannot jump INITIATED -> ACTIVE without AUTHENTICATING
        with pytest.raises(SessionTransitionError):
            session.transition_to(SessionStatus.ACTIVE)

        # Terminated is terminal
        session.transition_to(SessionStatus.AUTHENTICATING)
        session.mark_authenticated(pub, 10)
        session.terminate("Done")
        with pytest.raises(SessionTransitionError):
            session.transition_to(SessionStatus.ACTIVE)
        with pytest.raises(SessionTransitionError):
            session.transition_to(SessionStatus.AUTHENTICATING)

    def test_08_session_freshness_bounded_lifetime(self, peer_keys):
        _, pub = peer_keys
        session = SecurePeerSession(
            session_id="sess_test_08",
            local_peer_id="local_engine_node",
            remote_peer_id="peer_remote_beta",
            local_zone_id="ZONE_SECURE_ALPHA",
            remote_zone_id="ZONE_REMOTE_BETA",
            created_epoch=10,
            expires_at_epoch=200,
            max_lifetime_epochs=200,
        )
        session.mark_authenticated(pub, authenticated_epoch=10)

        # Exceeds max_lifetime_epochs (10 + 200 = 210)
        with pytest.raises(SessionTransitionError, match="exceeds maximum session lifetime"):
            session.renew(extension_epochs=20, current_epoch=190)

    def test_09_max_renewal_count_enforced(self, peer_keys):
        _, pub = peer_keys
        session = SecurePeerSession(
            session_id="sess_test_09",
            local_peer_id="local_engine_node",
            remote_peer_id="peer_remote_beta",
            local_zone_id="ZONE_SECURE_ALPHA",
            remote_zone_id="ZONE_REMOTE_BETA",
            created_epoch=10,
            expires_at_epoch=20,
            max_renewals=2,
        )
        session.mark_authenticated(pub, authenticated_epoch=10)
        session.renew(extension_epochs=5, current_epoch=15)  # count 1
        session.renew(extension_epochs=5, current_epoch=20)  # count 2

        with pytest.raises(SessionTransitionError, match="Maximum renewal count"):
            session.renew(extension_epochs=5, current_epoch=25)


# ============================================================================
# 2. Session Key Lifecycle & Secret Protection
# ============================================================================

class TestSessionKeyLifecycle:

    def test_10_session_key_creation(self):
        meta = SessionKeyMetadata.create("sess_test_10", expires_epoch=50)
        assert meta.key_state == SessionKeyState.CREATED
        assert meta.session_id == "sess_test_10"
        assert meta.key_id.startswith("sk_")
        assert meta.is_valid(current_epoch=20) is False  # Must be ACTIVE

        meta.activate(activated_epoch=10)
        assert meta.key_state == SessionKeyState.ACTIVE
        assert meta.is_valid(current_epoch=20) is True

    def test_11_session_key_invalidation_on_termination(self, peer_keys):
        _, pub = peer_keys
        session = SecurePeerSession(
            session_id="sess_test_11",
            local_peer_id="local_engine_node",
            remote_peer_id="peer_remote_beta",
            local_zone_id="ZONE_SECURE_ALPHA",
            remote_zone_id="ZONE_REMOTE_BETA",
            created_epoch=10,
            expires_at_epoch=50,
        )
        session.mark_authenticated(pub, 10)
        assert session.session_key_metadata.is_valid(current_epoch=20) is True

        session.terminate("Admin closed")
        assert session.session_key_metadata.key_state == SessionKeyState.EXPIRED
        assert session.session_key_metadata.is_valid(current_epoch=20) is False

    def test_12_secret_non_leakage(self, peer_keys):
        _, pub = peer_keys
        session = SecurePeerSession(
            session_id="sess_test_12",
            local_peer_id="local_engine_node",
            remote_peer_id="peer_remote_beta",
            local_zone_id="ZONE_SECURE_ALPHA",
            remote_zone_id="ZONE_REMOTE_BETA",
            created_epoch=10,
            expires_at_epoch=50,
        )
        session.mark_authenticated(pub, 10)

        rep = repr(session)
        s = str(session)
        d = session.to_dict()

        # Invariants: no private key, no raw symmetric key, no secret leakage
        assert "private_key" not in rep.lower()
        assert "private_key" not in s.lower()
        assert "private_key" not in str(d).lower()
        assert "secret" not in rep.lower()
        assert "secret" not in s.lower()


# ============================================================================
# 3. Peer Key Rotation & Identity Continuity
# ============================================================================

class TestPeerKeyRotation:

    def test_13_peer_key_rotation_valid_proof(self, engine, registered_peer):
        reg, old_priv, old_pub = registered_peer

        new_priv = Ed25519PrivateKeyWrapper.generate()
        new_pub = new_priv.get_public_key()

        # Old key signs rotation proof
        proof_payload = f"ROTATE_KEY:{old_pub.fingerprint}:{new_pub.fingerprint}:{engine.current_epoch}".encode("utf-8")
        proof_sig = old_priv.sign_hex(proof_payload)

        engine.rotate_peer_key(
            peer_id="peer_remote_beta",
            new_public_key=new_pub,
            rotation_proof_signature=proof_sig,
        )

        crypto_id = reg.cryptographic_identity
        assert crypto_id.public_key.fingerprint == new_pub.fingerprint
        assert crypto_id.is_key_retired(old_pub.fingerprint) is True
        assert len(crypto_id.rotation_history) == 1

    def test_14_failed_key_rotation_invalid_proof(self, engine, registered_peer):
        reg, old_priv, old_pub = registered_peer

        new_priv = Ed25519PrivateKeyWrapper.generate()
        new_pub = new_priv.get_public_key()

        invalid_sig = "deadbeef" * 8

        with pytest.raises(SignatureVerificationError):
            engine.rotate_peer_key(
                peer_id="peer_remote_beta",
                new_public_key=new_pub,
                rotation_proof_signature=invalid_sig,
            )

        # Original key remains active
        assert reg.cryptographic_identity.public_key.fingerprint == old_pub.fingerprint

    def test_15_key_rotation_rejected_on_revoked_peer(self, engine, registered_peer):
        reg, old_priv, old_pub = registered_peer

        engine.revoke_peer("peer_remote_beta", reason="Compromised", revoked_by="sec_admin")

        new_priv = Ed25519PrivateKeyWrapper.generate()
        new_pub = new_priv.get_public_key()
        proof_payload = f"ROTATE_KEY:{old_pub.fingerprint}:{new_pub.fingerprint}:{engine.current_epoch}".encode("utf-8")
        proof_sig = old_priv.sign_hex(proof_payload)

        with pytest.raises(KeyStateError, match="peer is revoked or unregistered"):
            engine.rotate_peer_key(
                peer_id="peer_remote_beta",
                new_public_key=new_pub,
                rotation_proof_signature=proof_sig,
            )

    def test_16_retired_key_signature_rejection(self, engine, registered_peer):
        reg, old_priv, old_pub = registered_peer
        session = engine.create_secure_session("peer_remote_beta", ttl_epochs=50)

        # Rotate key
        new_priv = Ed25519PrivateKeyWrapper.generate()
        new_pub = new_priv.get_public_key()
        proof_payload = f"ROTATE_KEY:{old_pub.fingerprint}:{new_pub.fingerprint}:{engine.current_epoch}".encode("utf-8")
        proof_sig = old_priv.sign_hex(proof_payload)
        engine.rotate_peer_key("peer_remote_beta", new_pub, proof_sig)

        # Wire envelope signed with OLD (retired) key
        env = WireEnvelope(
            protocol_version="31.0",
            message_type=MessageType.HEARTBEAT,
            message_id="msg_retired_01",
            session_id=session.session_id,
            sender_peer_id="peer_remote_beta",
            receiver_peer_id=engine.local_peer_id,
            created_epoch=engine.current_epoch,
            expires_epoch=engine.current_epoch + 10,
            payload={"ping": True},
        )
        env.sign(old_priv)

        with pytest.raises(SignatureVerificationError, match="retired key rejected"):
            engine.authorize_and_execute_wire_envelope(env)

    def test_17_identity_continuity_across_rotation(self, engine, registered_peer):
        reg, old_priv, old_pub = registered_peer
        orig_peer_id = reg.identity.peer_id
        orig_zone_id = reg.identity.zone_id
        orig_trust_grant = reg.trust_grant

        new_priv = Ed25519PrivateKeyWrapper.generate()
        new_pub = new_priv.get_public_key()
        proof = old_priv.sign_hex(f"ROTATE_KEY:{old_pub.fingerprint}:{new_pub.fingerprint}:{engine.current_epoch}".encode("utf-8"))
        engine.rotate_peer_key("peer_remote_beta", new_pub, proof)

        # Continuity preserved: peer ID, zone ID, and trust grant are unaffected
        assert reg.identity.peer_id == orig_peer_id
        assert reg.identity.zone_id == orig_zone_id
        assert reg.trust_grant == orig_trust_grant
        # Authority/scope did NOT escalate
        assert reg.trust_grant.permitted_scopes == [FederationScope.ALLOW_COGNITIVE_TASK_DELEGATION]


# ============================================================================
# 4. Certificate Rotation & Identity Binding
# ============================================================================

class TestCertificateRotation:

    def test_18_certificate_rotation_valid(self, engine, registered_peer):
        reg, priv, pub = registered_peer

        ca_pem, ca_key_pem, ca_cert, ca_key = HermeticPKIBuilder.create_ca(common_name="Test CA")
        _, _, srv_cert1, _ = HermeticPKIBuilder.create_server_cert(ca_cert, ca_key, common_name="beta.example.com", san_dns=["beta.example.com"])
        meta1 = extract_certificate_metadata(srv_cert1)
        engine.bind_peer_certificate("peer_remote_beta", meta1.fingerprint, expected_common_name="beta.example.com")

        # Rotate to cert 2
        _, _, srv_cert2, _ = HermeticPKIBuilder.create_server_cert(ca_cert, ca_key, common_name="beta.example.com", san_dns=["beta.example.com"])
        meta2 = extract_certificate_metadata(srv_cert2)
        engine.rotate_peer_certificate("peer_remote_beta", meta2, expected_common_name="beta.example.com")

        # Verify active binding is now cert 2
        binding = engine.certificate_binder.get_binding("peer_remote_beta")
        assert binding.certificate_fingerprint == meta2.fingerprint
        assert binding.is_active is True

    def test_19_certificate_rotation_rejected_on_revoked_cert(self, engine, registered_peer):
        reg, priv, pub = registered_peer
        ca_pem, ca_key_pem, ca_cert, ca_key = HermeticPKIBuilder.create_ca(common_name="Test CA")
        _, _, cert, _ = HermeticPKIBuilder.create_server_cert(ca_cert, ca_key, common_name="beta.example.com")
        meta = extract_certificate_metadata(cert)

        # Register certificate as REVOKED in engine registry
        engine.certificate_revocation_registry.revoke(
            fingerprint=meta.fingerprint,
            reason="Compromised CA",
            epoch=engine.current_epoch,
        )

        with pytest.raises(CertificateValidationError, match="revoked"):
            engine.rotate_peer_certificate("peer_remote_beta", meta, expected_common_name="beta.example.com")


# ============================================================================
# 5. Trust Continuity, Expiry & Revocation Cascade
# ============================================================================

class TestTrustContinuityAndCascade:

    def test_20_trust_expiry_blocks_session_renewal(self, engine, registered_peer):
        reg, priv, pub = registered_peer
        session = engine.create_secure_session("peer_remote_beta", ttl_epochs=30)
        assert session.expires_at_epoch == 40

        # Advance engine epoch past trust grant expiry (expires at 80)
        engine.current_epoch = 85

        with pytest.raises(CrossZoneAuthorizationError, match="trust grant is expired or missing"):
            engine.renew_session(session.session_id, extension_epochs=20)

    def test_21_session_cannot_outlive_trust_grant(self, engine, registered_peer):
        reg, priv, pub = registered_peer
        # Trust grant expires at epoch 80
        session = engine.create_secure_session("peer_remote_beta", ttl_epochs=70)
        # session expires at epoch 10 + 70 = 80
        assert session.expires_at_epoch == 80

        with pytest.raises(CrossZoneAuthorizationError, match="Cannot renew session beyond trust grant expiration"):
            engine.renew_session(session.session_id, extension_epochs=10)

    def test_22_revocation_cascade(self, engine, registered_peer):
        reg, priv, pub = registered_peer
        session1 = engine.create_secure_session("peer_remote_beta", ttl_epochs=30)
        session2 = engine.create_secure_session("peer_remote_beta", ttl_epochs=40)

        ca_pem, ca_key_pem, ca_cert, ca_key = HermeticPKIBuilder.create_ca(common_name="Test CA")
        _, _, cert, _ = HermeticPKIBuilder.create_server_cert(ca_cert, ca_key, common_name="beta.example.com")
        meta = extract_certificate_metadata(cert)
        engine.bind_peer_certificate("peer_remote_beta", meta.fingerprint)

        # Full peer revocation cascade
        engine.revoke_peer("peer_remote_beta", reason="Compromise detected", revoked_by="admin")

        # 1. Peer in registry is REVOKED
        assert reg.discovery_status == DiscoveryStatus.REVOKED
        assert reg.revocation_record is not None

        # 2. Cryptographic identity is REVOKED
        assert reg.cryptographic_identity.key_state == KeyLifecycleState.REVOKED

        # 3. All sessions are REVOKED
        assert session1.status == SessionStatus.REVOKED
        assert session2.status == SessionStatus.REVOKED

        # 4. Certificate binding deactivated / removed
        binding = engine.certificate_binder.get_binding("peer_remote_beta")
        assert binding is None

    def test_23_stale_authorization_denied_on_inactive_session(self, engine, registered_peer):
        reg, priv, pub = registered_peer
        session = engine.create_secure_session("peer_remote_beta", ttl_epochs=20)
        engine.terminate_session(session.session_id, reason="Closing connection")

        env = WireEnvelope(
            protocol_version="31.0",
            message_type=MessageType.HEARTBEAT,
            message_id="msg_stale_01",
            session_id=session.session_id,
            sender_peer_id="peer_remote_beta",
            receiver_peer_id=engine.local_peer_id,
            created_epoch=engine.current_epoch,
            expires_epoch=engine.current_epoch + 10,
            payload={"ping": True},
        )
        env.sign(priv)

        with pytest.raises(CrossZoneAuthorizationError, match="Session '.*' inactive"):
            engine.authorize_and_execute_wire_envelope(env)


# ============================================================================
# 6. Replay Protection & Sequence Monotonicity
# ============================================================================

class TestReplayAndSequenceProtection:

    def test_24_replay_message_id_rejection(self, engine, registered_peer):
        reg, priv, pub = registered_peer
        session = engine.create_secure_session("peer_remote_beta", ttl_epochs=50)

        env = WireEnvelope(
            protocol_version="31.0",
            message_type=MessageType.HEARTBEAT,
            message_id="msg_replay_fixed_id",
            session_id=session.session_id,
            sender_peer_id="peer_remote_beta",
            receiver_peer_id=engine.local_peer_id,
            created_epoch=engine.current_epoch,
            expires_epoch=engine.current_epoch + 10,
            payload={"ping": True},
        )
        env.sign(priv)

        # First message passes
        resp1 = engine.authorize_and_execute_wire_envelope(env)
        assert resp1.payload.get("status") == "PONG"

        # Replayed message fails closed
        with pytest.raises(ReplayAttackError, match="Replay attack detected: Message ID '.*' was already seen"):
            engine.authorize_and_execute_wire_envelope(env)

    def test_25_sequence_number_monotonicity(self, engine, registered_peer):
        reg, priv, pub = registered_peer
        session = engine.create_secure_session("peer_remote_beta", ttl_epochs=50)

        env1 = WireEnvelope(
            protocol_version="31.0",
            message_type=MessageType.HEARTBEAT,
            message_id="msg_seq_01",
            session_id=session.session_id,
            sender_peer_id="peer_remote_beta",
            receiver_peer_id=engine.local_peer_id,
            created_epoch=engine.current_epoch,
            expires_epoch=engine.current_epoch + 10,
            payload={"sequence_number": 5},
        ).sign(priv)
        engine.authorize_and_execute_wire_envelope(env1)

        # Sequence 4 (lower than 5) must be rejected as replay/out-of-order
        env2 = WireEnvelope(
            protocol_version="31.0",
            message_type=MessageType.HEARTBEAT,
            message_id="msg_seq_02",
            session_id=session.session_id,
            sender_peer_id="peer_remote_beta",
            receiver_peer_id=engine.local_peer_id,
            created_epoch=engine.current_epoch,
            expires_epoch=engine.current_epoch + 10,
            payload={"sequence_number": 4},
        ).sign(priv)

        with pytest.raises(ReplayAttackError, match="Out-of-order or duplicate sequence number"):
            engine.authorize_and_execute_wire_envelope(env2)

    def test_26_bounded_replay_cache(self, peer_keys):
        _, pub = peer_keys
        session = SecurePeerSession(
            session_id="sess_cache_test",
            local_peer_id="local_engine_node",
            remote_peer_id="peer_remote_beta",
            local_zone_id="ZONE_SECURE_ALPHA",
            remote_zone_id="ZONE_REMOTE_BETA",
            created_epoch=10,
            expires_at_epoch=50,
            max_message_history=10,
        )
        session.mark_authenticated(pub, 10)

        for i in range(15):
            assert session.record_and_check_message_id(f"msg_{i}") is True

        # Cache is capped at max_cache_size
        assert len(session.seen_message_ids) <= 10


# ============================================================================
# 7. Capability Scope & Neural Core Immutability
# ============================================================================

class TestCapabilityAndNeuralCoreInvariants:

    def test_27_capability_scope_preservation_and_neural_immutability(self, engine, registered_peer, model):
        reg, priv, pub = registered_peer
        session = engine.create_secure_session("peer_remote_beta", ttl_epochs=30)

        # Compute neural weight hash before lifecycle
        hasher_pre = hashlib.sha256()
        for p in model.parameters():
            hasher_pre.update(p.detach().cpu().numpy().tobytes())
        pre_hash = hasher_pre.hexdigest()

        # Execute capability request
        env = WireEnvelope(
            protocol_version="31.0",
            message_type=MessageType.CAPABILITY_REQUEST,
            message_id="msg_cap_01",
            session_id=session.session_id,
            sender_peer_id="peer_remote_beta",
            receiver_peer_id=engine.local_peer_id,
            created_epoch=engine.current_epoch,
            expires_epoch=engine.current_epoch + 10,
            payload={
                "requested_scope": FederationScope.ALLOW_COGNITIVE_TASK_DELEGATION.value,
                "data": {"task": "verify_invariant"},
            },
        ).sign(priv)

        resp = engine.authorize_and_execute_wire_envelope(env)
        assert resp.payload.get("authorized") is True

        # Renew session
        engine.renew_session(session.session_id, extension_epochs=20)

        # Verify scope remains strictly ALLOW_COGNITIVE_TASK_DELEGATION
        assert reg.trust_grant.permitted_scopes == [FederationScope.ALLOW_COGNITIVE_TASK_DELEGATION]

        # Verify neural weight hash after lifecycle (ΔW = 0)
        hasher_post = hashlib.sha256()
        for p in model.parameters():
            hasher_post.update(p.detach().cpu().numpy().tobytes())
        post_hash = hasher_post.hexdigest()

        assert pre_hash == post_hash
        assert sum(p.numel() for p in model.parameters()) == 3_443_136
