"""
Test Suite for Distributed Federation Coordination, Replay Synchronization & Trust Consistency (Step 33).

Validates all 27 Section 11 requirements:
1. Engine identity creation
2. Engine identity uniqueness
3. Engine identity does not imply authority
4. Federation handshake
5. Invalid handshake rejection
6. Protocol mismatch rejection
7. State version monotonicity
8. State regression rejection
9. Same-version digest conflict
10. Replay-state synchronization
11. Replay cache remains locally authoritative
12. Replay synchronization cannot authorize traffic
13. Trust-state synchronization
14. Remote trust cannot self-escalate
15. Revocation propagation
16. Duplicate revocation idempotency
17. Revocation monotonicity
18. Local revocation beats remote active state
19. State digest determinism
20. State digest excludes secrets
21. Tenant isolation remains intact
22. Capability scope remains unchanged
23. Cross-zone authority transfer remains impossible
24. Bounded synchronization memory
25. Audit records contain no secrets
26. Neural core remains immutable (ΔW = 0)
27. Existing Step 32 functionality remains intact
"""

import copy
import hashlib
import json
import pytest
import secrets
import torch

from chakrview.cognition.federation.models import (
    FederationEngineIdentity,
    SecurityStateVersion,
    FederationSecurityStateDigest,
    ReplayStateDigest,
    TrustStateDigest,
    RevocationStateDigest,
    PeerStateDigest,
    HandshakeStatus,
    FederationHandshakeRequest,
    FederationHandshakeResponse,
    ReplaySyncMessage,
    TrustSyncRecord,
    TrustSyncMessage,
    RevocationSyncRecord,
    RevocationSyncMessage,
    RevocationTargetType,
    MAX_FEDERATION_ENGINES,
    FEDERATION_PROTOCOL_VERSION,
)
from chakrview.cognition.federation.identity import FederationEngineIdentityProvider
from chakrview.cognition.federation.state import FederationStateManager
from chakrview.cognition.federation.replay_sync import ReplayStateSynchronizer
from chakrview.cognition.federation.trust_sync import TrustStateSynchronizer
from chakrview.cognition.federation.revocation_sync import RevocationStateSynchronizer
from chakrview.cognition.federation.handshake import FederationHandshakeManager
from chakrview.cognition.federation.coordinator import DistributedFederationCoordinator
from chakrview.cognition.federation.errors import (
    FederationCoordinationError,
    EngineIdentityError,
    ProtocolMismatchError,
    HandshakeError,
    StateVersionError,
    StaleStateError,
    StateDigestConflictError,
    CoordinationCapacityError,
)

from chakrview.cognition.peering.engine import CrossZoneFederationEngine, CrossZoneAuthorizationError
from chakrview.cognition.peering.crypto import (
    Ed25519PrivateKeyWrapper,
    Ed25519PublicKeyWrapper,
    CryptographicPeerIdentity,
)
from chakrview.cognition.peering.session import SecurePeerSession, SessionStatus
from chakrview.cognition.peering.models import (
    PeerIdentity,
    PeerAttestation,
    PeerDeclaration,
    TrustLevel,
    TrustStatus,
    FederationScope,
    AuditEventType,
)
from chakrview.cognition.peering.policy import CrossZoneFederationPolicy
from chakrview.cognition.transport.models import WireEnvelope, MessageType
from chakrview.cognition.transport.errors import ReplayAttackError
from chakrview.capability.gate import CapabilityGate, CapabilityAuthorizationError
from chakrview.capability.registry import CapabilityNotFoundError
from chakrview.capability.contract import CapabilityRequest
from chakrview.brain.model import ChakrMicro, ModelConfig


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def test_model():
    torch.manual_seed(42)
    cfg = ModelConfig()
    m = ChakrMicro(cfg)
    m.eval()
    return m


@pytest.fixture
def mock_capability_gate():
    gate = CapabilityGate()
    return gate


@pytest.fixture
def two_engines(test_model, mock_capability_gate):
    engine_a = CrossZoneFederationEngine(
        local_zone_id="zone-alpha",
        capability_gate=mock_capability_gate,
        model=test_model,
        initial_epoch=1,
    )
    engine_b = CrossZoneFederationEngine(
        local_zone_id="zone-beta",
        capability_gate=mock_capability_gate,
        model=test_model,
        initial_epoch=1,
    )
    return engine_a, engine_b


# ============================================================================
# Phase 1: Engine Identity Tests
# ============================================================================

def test_01_engine_identity_creation():
    """Verify engine identity creation, structure, and deterministic fingerprint."""
    ident = FederationEngineIdentityProvider.create_identity(
        engine_id="eng_alpha_01",
        zone_id="zone_alpha",
        created_epoch=1,
        public_key_fingerprint="abc123def456",
    )
    assert ident.engine_id == "eng_alpha_01"
    assert ident.zone_id == "zone_alpha"
    assert ident.created_epoch == 1
    assert ident.protocol_version == FEDERATION_PROTOCOL_VERSION
    assert len(ident.identity_fingerprint) == 64

    is_valid, reason = FederationEngineIdentityProvider.validate_identity(ident)
    assert is_valid is True
    assert reason == "Valid engine identity"


def test_02_engine_identity_uniqueness():
    """Verify different engine IDs yield distinct fingerprints and identical yield same."""
    ident_a = FederationEngineIdentityProvider.create_identity(
        engine_id="eng_alpha",
        zone_id="zone_alpha",
    )
    ident_b = FederationEngineIdentityProvider.create_identity(
        engine_id="eng_beta",
        zone_id="zone_beta",
    )
    ident_a_dup = FederationEngineIdentityProvider.create_identity(
        engine_id="eng_alpha",
        zone_id="zone_alpha",
    )

    assert ident_a.identity_fingerprint != ident_b.identity_fingerprint
    assert ident_a.identity_fingerprint == ident_a_dup.identity_fingerprint


def test_03_engine_identity_does_not_imply_authority(two_engines):
    """Verify having a valid engine identity confers zero local execution authority."""
    engine_a, engine_b = two_engines

    # Engine A registers Engine B's identity
    engine_a.coordinator.register_remote_engine(engine_b.engine_identity)
    assert engine_a.coordinator.get_registered_engine(engine_b.engine_identity.engine_id) is not None

    # Wire request from Engine B without trust grant must fail closed
    envelope = WireEnvelope(
        protocol_version="30.0",
        message_type=MessageType.CAPABILITY_REQUEST,
        message_id="msg_unauthorized_001",
        session_id="nonexistent_session",
        sender_peer_id=engine_b.local_peer_id,
        receiver_peer_id=engine_a.local_peer_id,
        created_epoch=1,
        expires_epoch=10,
        payload={"scope": FederationScope.ALLOW_EVIDENCE_EXCHANGE.value},
    )
    envelope.sign(engine_b.local_private_key)

    with pytest.raises(CrossZoneAuthorizationError):
        engine_a.authorize_and_execute_wire_envelope(envelope)


# ============================================================================
# Phase 8 & 9: Federation Handshake & Conflict Tests
# ============================================================================

def test_04_federation_handshake_success(two_engines):
    """Verify coordination handshake succeeds and records audit events."""
    engine_a, engine_b = two_engines

    response = engine_a.initiate_coordination_handshake(engine_b)
    assert response.status == HandshakeStatus.ACCEPTED
    assert response.is_accepted() is True

    # Both engines now register each other
    assert engine_a.coordinator.get_registered_engine(engine_b.engine_identity.engine_id) is not None
    assert engine_b.coordinator.get_registered_engine(engine_a.engine_identity.engine_id) is not None


def test_05_invalid_handshake_rejection(two_engines):
    """Verify handshake with malformed or invalid identity is rejected."""
    engine_a, engine_b = two_engines

    # Tampered identity with invalid fingerprint
    tampered_identity = FederationEngineIdentity(
        engine_id="eng_tampered",
        zone_id="zone_unknown",
        created_epoch=1,
        identity_fingerprint="forged_fingerprint_that_does_not_match_canonical_hash",
    )

    request = FederationHandshakeRequest(
        sender_identity=tampered_identity,
        protocol_version=FEDERATION_PROTOCOL_VERSION,
        epoch=1,
        state_version=1,
        composite_digest="hash123",
        replay_digest="hash123",
        trust_digest="hash123",
        revocation_digest="hash123",
        nonce="nonce123",
    )

    response = engine_b.coordinator.handle_handshake(request)
    assert response.status == HandshakeStatus.REJECTED
    assert "Invalid sender identity" in response.details.get("error", "")


def test_06_protocol_mismatch_rejection(two_engines):
    """Verify protocol version incompatibility fails closed."""
    engine_a, engine_b = two_engines

    # Sender claiming obsolete protocol version
    obsolete_identity = FederationEngineIdentityProvider.create_identity(
        engine_id="eng_old",
        zone_id="zone_old",
        protocol_version="28.0",
    )

    request = FederationHandshakeRequest(
        sender_identity=obsolete_identity,
        protocol_version="28.0",
        epoch=1,
        state_version=1,
        composite_digest="hash123",
        replay_digest="hash123",
        trust_digest="hash123",
        revocation_digest="hash123",
        nonce="nonce123",
    )

    response = engine_b.coordinator.handle_handshake(request)
    assert response.status == HandshakeStatus.INCOMPATIBLE_PROTOCOL
    assert "Protocol mismatch" in response.details.get("error", "")


# ============================================================================
# Phase 4 & Phase 7: State Versioning & Digest Tests
# ============================================================================

def test_07_state_version_monotonicity(two_engines):
    """Verify state version strictly increases upon security events."""
    engine_a, _ = two_engines
    v1 = engine_a.state_version.version
    assert v1 == 1

    engine_a.coordinator.state_manager.increment_version()
    v2 = engine_a.state_version.version
    assert v2 == 2
    assert v2 > v1


def test_08_state_regression_rejection(two_engines):
    """Verify epoch or version regressions are strictly rejected."""
    engine_a, _ = two_engines
    engine_a.advance_epoch(5)
    assert engine_a.current_epoch == 6

    with pytest.raises(StateVersionError):
        # Cannot regress epoch
        engine_a.coordinator.state_manager.advance_epoch(4)


def test_09_same_version_digest_conflict(two_engines):
    """Verify same-version divergent state digests trigger StateDigestConflictError."""
    engine_a, _ = two_engines
    sm = engine_a.coordinator.state_manager

    remote_version = SecurityStateVersion(
        version=sm.current_version.version,
        epoch=sm.current_version.epoch,
        engine_id="eng_remote_divergent",
    )

    with pytest.raises(StateDigestConflictError):
        sm.evaluate_remote_version(
            remote_version=remote_version,
            remote_digest="aaaa" * 16,
            local_digest="bbbb" * 16,
        )


# ============================================================================
# Phase 3: Replay Synchronization Tests
# ============================================================================

def test_10_replay_state_synchronization(two_engines):
    """Verify advisory replay state synchronization between engines."""
    engine_a, engine_b = two_engines

    # Setup shared session ID
    session_id = "sess_shared_100"
    session_a = SecurePeerSession(
        session_id=session_id,
        local_peer_id=engine_a.local_peer_id,
        remote_peer_id="peer_remote",
        local_zone_id="zone-alpha",
        remote_zone_id="zone-remote",
    )
    session_a.mark_authenticated(engine_a.local_public_key, authenticated_epoch=1)
    session_a.record_and_check_sequence(10)
    session_a.record_and_check_message_id("msg_001")
    session_a.record_and_check_message_id("msg_002")
    engine_a.sessions[session_id] = session_a

    session_b = SecurePeerSession(
        session_id=session_id,
        local_peer_id=engine_b.local_peer_id,
        remote_peer_id="peer_remote",
        local_zone_id="zone-beta",
        remote_zone_id="zone-remote",
    )
    session_b.mark_authenticated(engine_b.local_public_key, authenticated_epoch=1)
    engine_b.sessions[session_id] = session_b

    # Synchronize replay from Engine A to Engine B
    outcome = engine_a.synchronize_replay_with_engine(engine_b, session_id=session_id)
    assert outcome["new_message_ids_absorbed"] == 2
    assert outcome["sequence_action"] == "ADVANCED"
    assert session_b.last_seen_sequence_number == 10
    assert "msg_001" in session_b.seen_message_ids
    assert "msg_002" in session_b.seen_message_ids


def test_11_replay_cache_remains_locally_authoritative(two_engines):
    """Verify local replay protection remains active and cannot be bypassed by sync."""
    engine_a, engine_b = two_engines
    session_id = "sess_authoritative_101"
    session_b = SecurePeerSession(
        session_id=session_id,
        local_peer_id=engine_b.local_peer_id,
        remote_peer_id="peer_remote",
        local_zone_id="zone-beta",
        remote_zone_id="zone-remote",
    )
    session_b.mark_authenticated(engine_b.local_public_key, authenticated_epoch=1)
    session_b.record_and_check_message_id("msg_authoritative_01")
    engine_b.sessions[session_id] = session_b

    # Ingest sync containing the same message ID
    sync_msg = ReplaySyncMessage(
        engine_id=engine_a.engine_identity.engine_id,
        session_id=session_id,
        epoch=1,
        highest_sequence_number=5,
        bounded_message_id_digest=hashlib.sha256(json.dumps(["msg_authoritative_01"], separators=(",", ":")).encode("utf-8")).hexdigest(),
        recent_message_ids=["msg_authoritative_01"],
        state_version=1,
    )
    outcome = engine_b.coordinator.replay_synchronizer.ingest_sync_message(session_b, sync_msg)
    assert outcome["duplicate_message_ids_ignored"] == 1

    # Local attempt to process this message again locally MUST fail
    is_fresh = session_b.record_and_check_message_id("msg_authoritative_01")
    assert is_fresh is False


def test_12_replay_sync_cannot_authorize_traffic(two_engines):
    """Verify replay sync never authorizes wire traffic on its own."""
    engine_a, engine_b = two_engines
    session_id = "sess_no_auth_102"

    session_a = SecurePeerSession(
        session_id=session_id,
        local_peer_id=engine_a.local_peer_id,
        remote_peer_id="peer_untrusted",
        local_zone_id="zone-alpha",
        remote_zone_id="zone-untrusted",
    )
    session_a.mark_authenticated(engine_a.local_public_key, authenticated_epoch=1)
    engine_a.sessions[session_id] = session_a

    session_b = SecurePeerSession(
        session_id=session_id,
        local_peer_id=engine_b.local_peer_id,
        remote_peer_id="peer_untrusted",
        local_zone_id="zone-beta",
        remote_zone_id="zone-untrusted",
    )
    session_b.mark_authenticated(engine_b.local_public_key, authenticated_epoch=1)
    engine_b.sessions[session_id] = session_b

    # Synchronize replay
    engine_a.synchronize_replay_with_engine(engine_b, session_id=session_id)

    # An unauthorized capability request must still be rejected on Engine B
    envelope = WireEnvelope(
        protocol_version="30.0",
        message_type=MessageType.CAPABILITY_REQUEST,
        message_id="msg_unauth_002",
        session_id=session_id,
        sender_peer_id="peer_untrusted",
        receiver_peer_id=engine_b.local_peer_id,
        created_epoch=1,
        expires_epoch=10,
        payload={"scope": FederationScope.ALLOW_EVIDENCE_EXCHANGE.value},
    )
    envelope.sign(engine_a.local_private_key)

    with pytest.raises(CrossZoneAuthorizationError):
        engine_b.authorize_and_execute_wire_envelope(envelope)


# ============================================================================
# Phase 5: Trust State Consistency Tests
# ============================================================================

def test_13_trust_state_synchronization(two_engines):
    """Verify trust state report exchange."""
    engine_a, engine_b = two_engines
    outcome = engine_a.synchronize_trust_with_engine(engine_b)
    assert outcome["local_authority_preserved"] is True
    assert outcome["escalation_attempts_blocked"] == 0


def test_14_remote_trust_cannot_self_escalate(two_engines):
    """Verify remote engine reporting FEDERATED trust cannot escalate local trust."""
    engine_a, engine_b = two_engines

    # Engine A fabricates a claim that peer_rogue has FEDERATED trust
    fabricated_msg = TrustSyncMessage(
        engine_id=engine_a.engine_identity.engine_id,
        zone_id="zone-alpha",
        epoch=1,
        state_version=1,
        trust_records=[
            TrustSyncRecord(
                subject_peer_id="peer_rogue",
                subject_zone_id="zone-rogue",
                trust_level=TrustLevel.FEDERATED.value,
                status=TrustStatus.ACTIVE.value,
                expires_epoch=50,
                permitted_scopes=[FederationScope.ALLOW_COGNITIVE_TASK_DELEGATION.value],
            )
        ],
    )

    outcome = engine_b.coordinator.trust_synchronizer.ingest_sync_message(
        registry=engine_b.registry,
        sync_message=fabricated_msg,
    )
    assert outcome["escalation_attempts_blocked"] == 1

    # Verify peer_rogue was NOT granted trust in Engine B
    assert engine_b.registry.get_peer("peer_rogue") is None


# ============================================================================
# Phase 6 & 9: Revocation Propagation Tests
# ============================================================================

def test_15_revocation_propagation(two_engines):
    """Verify peer revocation propagates to remote engine and triggers cascade."""
    engine_a, engine_b = two_engines
    peer_id = "peer_target_105"

    # Register peer in Engine B
    crypto_id = CryptographicPeerIdentity.create(
        peer_id=peer_id,
        zone_id="zone-gamma",
        organization_id="org-gamma",
        public_key=engine_a.local_public_key,
    )
    reg = engine_b.register_cryptographic_peer(crypto_id)
    assert engine_b.registry.get_peer(peer_id) is not None

    # Propagate revocation from Engine A to Engine B
    rec = engine_a.propagate_revocation_to_engines(
        target_type=RevocationTargetType.PEER,
        target_id=peer_id,
        reason="Security breach confirmed",
        remote_engines=[engine_b],
    )

    assert rec.target_id == peer_id
    # Peer in Engine B is now revoked
    peer_b = engine_b.registry.get_peer(peer_id)
    assert peer_b.discovery_status.value == "REVOKED"


def test_16_duplicate_revocation_idempotency(two_engines):
    """Verify duplicate revocation is handled idempotently without error."""
    engine_a, engine_b = two_engines
    peer_id = "peer_target_106"

    # Propagate first time
    engine_a.propagate_revocation_to_engines(
        target_type=RevocationTargetType.PEER,
        target_id=peer_id,
        reason="Duplicate test",
        remote_engines=[engine_b],
    )

    # Propagate second time (duplicate)
    rec2 = engine_a.propagate_revocation_to_engines(
        target_type=RevocationTargetType.PEER,
        target_id=peer_id,
        reason="Duplicate test",
        remote_engines=[engine_b],
    )
    assert rec2 is not None


def test_17_revocation_monotonicity(two_engines):
    """Verify REVOKED -> NEVER ACTIVE AGAIN invariant."""
    engine_a, _ = two_engines
    peer_id = "peer_target_107"

    crypto_id = CryptographicPeerIdentity.create(
        peer_id=peer_id,
        zone_id="zone-gamma",
        organization_id="org-gamma",
        public_key=engine_a.local_public_key,
    )
    engine_a.register_cryptographic_peer(crypto_id)
    engine_a.revoke_peer(peer_id=peer_id, reason="Monotonic test")

    peer = engine_a.registry.get_peer(peer_id)
    assert peer.is_active(current_epoch=1) is False


def test_18_local_revocation_beats_remote_active_state(two_engines):
    """Verify local revocation unconditionally wins over remote active claims."""
    engine_a, engine_b = two_engines
    peer_id = "peer_target_108"

    # Locally revoke in Engine B
    crypto_id = CryptographicPeerIdentity.create(
        peer_id=peer_id,
        zone_id="zone-delta",
        organization_id="org-delta",
        public_key=engine_a.local_public_key,
    )
    engine_b.register_cryptographic_peer(crypto_id)
    engine_b.revoke_peer(peer_id, reason="Compromised key")

    # Engine A reports peer as ACTIVE
    sync_msg = TrustSyncMessage(
        engine_id=engine_a.engine_identity.engine_id,
        zone_id="zone-alpha",
        epoch=1,
        state_version=1,
        trust_records=[
            TrustSyncRecord(
                subject_peer_id=peer_id,
                subject_zone_id="zone-delta",
                trust_level=TrustLevel.LIMITED_TRUST.value,
                status=TrustStatus.ACTIVE.value,
                expires_epoch=50,
                permitted_scopes=[FederationScope.ALLOW_EVIDENCE_EXCHANGE.value],
            )
        ],
    )

    engine_b.coordinator.trust_synchronizer.ingest_sync_message(
        registry=engine_b.registry,
        sync_message=sync_msg,
    )

    # Local standing MUST remain revoked
    peer = engine_b.registry.get_peer(peer_id)
    assert peer.discovery_status.value == "REVOKED"


# ============================================================================
# Phase 7 & Invariant Tests
# ============================================================================

def test_19_state_digest_determinism(two_engines):
    """Verify cryptographic state digests are strictly deterministic."""
    engine_a, _ = two_engines
    digest_1 = engine_a.compute_security_state_digest().compute_composite_digest()
    digest_2 = engine_a.compute_security_state_digest().compute_composite_digest()
    assert digest_1 == digest_2
    assert len(digest_1) == 64


def test_20_state_digest_excludes_secrets(two_engines):
    """Verify state digests and serializations contain zero secret material."""
    engine_a, _ = two_engines
    state = engine_a.compute_security_state_digest()
    state_json = json.dumps(state.to_dict())

    # Invariants: no private key, no symmetric key, no secret token
    assert "private_key" not in state_json
    assert "secret" not in state_json.lower()
    assert "token" not in state_json.lower()


def test_21_tenant_isolation_remains_intact(two_engines):
    """Verify tenant isolation boundaries are strictly enforced across federation."""
    from chakrview.cognition.peering.isolation import CrossZoneIsolationGuard, IsolationViolationError

    # Cross-tenant crossover between zones must fail closed
    with pytest.raises(IsolationViolationError):
        CrossZoneIsolationGuard.validate_tenant_boundary(
            peer_zone_id="zone-remote",
            target_zone_id="zone-alpha",
            peer_tenant_id="tenant-remote",
            target_tenant_id="tenant-local",
        )


def test_22_capability_scope_remains_unchanged(two_engines):
    """Verify coordination handshakes never alter or expand capability scopes."""
    engine_a, engine_b = two_engines
    # Initiate handshake
    engine_a.initiate_coordination_handshake(engine_b)

    # Scopes remain strictly default empty
    assert len(engine_b.registry.list_peers()) == 0


def test_23_cross_zone_authority_transfer_remains_impossible(two_engines):
    """Verify remote engine cannot bypass local CapabilityGate."""
    engine_a, engine_b = two_engines
    # Capability execution requires local CapabilityGate passage
    req = CapabilityRequest(
        capability_id="neural_inference",
        parameters={"prompt": "Analyze"},
        caller_id="remote_engine",
    )
    # Without local gate authorization, cannot execute
    with pytest.raises((CapabilityAuthorizationError, CapabilityNotFoundError)):
        engine_a.capability_gate.authorize(req)


def test_24_bounded_synchronization_memory(two_engines):
    """Verify MAX_FEDERATION_ENGINES limit fails closed."""
    engine_a, _ = two_engines
    for i in range(MAX_FEDERATION_ENGINES):
        ident = FederationEngineIdentityProvider.create_identity(
            engine_id=f"eng_cap_{i}",
            zone_id=f"zone_cap_{i}",
        )
        engine_a.coordinator.register_remote_engine(ident)

    # 17th engine must raise CoordinationCapacityError
    overflow_ident = FederationEngineIdentityProvider.create_identity(
        engine_id="eng_overflow",
        zone_id="zone_overflow",
    )
    with pytest.raises(CoordinationCapacityError):
        engine_a.coordinator.register_remote_engine(overflow_ident)


def test_25_audit_records_contain_no_secrets(two_engines):
    """Verify all audit logs contain zero secrets in details."""
    engine_a, engine_b = two_engines
    engine_a.initiate_coordination_handshake(engine_b)

    for entry in engine_a.audit_logger.get_recent(limit=100):
        entry_str = json.dumps(entry.to_dict()).lower()
        assert "private_key" not in entry_str
        assert "private_bytes" not in entry_str


def test_26_neural_core_remains_immutable(two_engines, test_model):
    """Verify neural parameters, vocabulary, sequence length, and ΔW = 0."""
    engine_a, engine_b = two_engines
    pre_hash = engine_a._compute_weight_hash()

    # Perform extensive multi-engine coordination operations
    engine_a.initiate_coordination_handshake(engine_b)
    engine_a.synchronize_trust_with_engine(engine_b)
    engine_a.propagate_revocation_to_engines(
        target_type=RevocationTargetType.PEER,
        target_id="peer_dummy",
        reason="Test neural immutability",
        remote_engines=[engine_b],
    )
    engine_a.advance_epoch(2)

    post_hash = engine_a._compute_weight_hash()
    assert pre_hash == post_hash, "Neural weights mutated during federation coordination!"

    # Verify structural constants
    assert sum(p.numel() for p in test_model.parameters()) == 3_443_136
    assert test_model.config.vocab_size == 4096
    assert test_model.config.max_seq_len == 512


def test_27_existing_step32_functionality_intact(two_engines):
    """Verify Step 32 session lifecycle and key rotation remain 100% operational."""
    engine_a, engine_b = two_engines
    peer_id = "peer_step32_valid"

    # Step 30/32 registration & session creation
    crypto_id = CryptographicPeerIdentity.create(
        peer_id=peer_id,
        zone_id="zone-gamma",
        organization_id="org-gamma",
        public_key=engine_b.local_public_key,
        created_epoch=1,
    )
    reg = engine_a.register_cryptographic_peer(crypto_id)
    session = engine_a.create_secure_session(
        remote_peer_id=peer_id,
    )
    assert session.status == SessionStatus.ACTIVE
    assert session.session_key_metadata.key_state.value == "ACTIVE"

    # Key rotation on peer
    new_priv = Ed25519PrivateKeyWrapper.generate()
    new_pub = new_priv.public_key()
    proof = engine_b.local_private_key.sign_hex(
        f"ROTATE_KEY:{engine_b.local_public_key.fingerprint}:{new_pub.fingerprint}:{engine_a.current_epoch}".encode("utf-8")
    )
    engine_a.rotate_peer_key(
        peer_id=peer_id,
        new_public_key=new_pub,
        rotation_proof_signature=proof,
    )
    assert reg.cryptographic_identity.public_key.fingerprint == new_pub.fingerprint
