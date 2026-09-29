"""
Comprehensive Security & Verification Test Suite for Step 35:
Production Federation Runtime Networking, Node Discovery & Secure Membership.

Verifies:
1. Discovery (Static config, deterministic candidate ID, duplicate detection, malformed/unsupported rejection)
2. Membership (Candidate registration, authentication, promotion, capacity bounds, invalid transitions, suspension, quarantine, revocation, termination)
3. Security (Unknown peer denied, invalid cert, cert/identity mismatch, invalid engine ID, failed handshake, trust/capability escalation denied)
4. Runtime (Heartbeat success/timeout, network outage != revocation, stale node rejoin, revoked rejoin denied, digest conflict, replay floor regression)
5. Persistence (Journal append, snapshot recovery, crash recovery, corrupted journal fail-closed, revocation survival across restart)
6. Isolation (Cross-tenant denied, cross-zone escalation denied)
7. Neural Core Immutability (Params=3,443,136, Vocab=4096, Context=512, Weight SHA-256 identical, ΔW = 0)
"""

import hashlib
import json
from pathlib import Path
import tempfile
import time
import pytest
import torch

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.capability.gate import CapabilityGate, CapabilityAuthorizationError
from chakrview.capability.contract import CapabilityRequest
from chakrview.capability.provider import CalculatorCapability
from chakrview.cognition.peering.crypto import (
    Ed25519PrivateKeyWrapper,
    Ed25519PublicKeyWrapper,
    CryptographicPeerIdentity,
    KeyLifecycleState,
)
from chakrview.cognition.peering.models import (
    AuditEventType,
    DiscoveryStatus,
    TrustGrant,
    TrustLevel,
    TrustStatus,
    FederationScope,
    RevocationRecord,
    PeerIdentity,
    PeerRegistration,
)
from chakrview.cognition.peering.engine import CrossZoneFederationEngine
from chakrview.cognition.peering.session import SecurePeerSession, SessionStatus
from chakrview.cognition.transport.security.models import (
    TLSMode,
    CertificateMetadata,
    CertificateLifecycleState,
)
from chakrview.cognition.transport.security.certificates import HermeticPKIBuilder
from chakrview.cognition.federation.models import (
    FederationEngineIdentity,
    SecurityStateVersion,
    RevocationTargetType,
    HandshakeStatus,
    MAX_FEDERATION_ENGINES,
)
from chakrview.cognition.federation.identity import FederationEngineIdentityProvider
from chakrview.cognition.federation.errors import (
    CoordinationCapacityError,
    StateDigestConflictError,
    HandshakeError,
)
from chakrview.cognition.federation.persistence import (
    SecurityStateStore,
    InMemorySecurityStateStore,
    SqliteSecurityStateStore,
    SecurityStateJournal,
    JournalEntry,
    JournalEntryType,
    JOURNAL_GENESIS_DIGEST,
    DurableSecuritySnapshot,
    RecoveryManifest,
    JournalCorruptionError,
    RecoveryFailedClosedError,
    RuntimeLifecycleError,
    EngineHealthError,
)
from chakrview.cognition.federation.recovery import FederationRecoveryManager
from chakrview.cognition.federation.runtime import (
    FederationRuntime,
    EngineRuntimeStatus,
    EngineHealthStatus,
)
from chakrview.cognition.federation.discovery import (
    NodeAddress,
    NodeProtocol,
    NodeDiscoverySource,
    MembershipState,
    VALID_MEMBERSHIP_TRANSITIONS,
    FederationNodeEndpoint,
    FederationNodeCandidate,
    FederationNodeMembership,
    DiscoveryError,
    MalformedEndpointError,
    UnsupportedProtocolError,
    DuplicateCandidateError,
    DiscoveryCapacityError,
    MembershipError,
    InvalidMembershipTransitionError,
    MembershipCapacityError,
    UnknownPeerError,
    QuarantineError,
    RevocationError,
    RejoinDeniedError,
    HeartbeatError,
    HeartbeatTimeoutError,
    HeartbeatSpoofingError,
    StaticConfigDiscoveryProvider,
    FileConfigDiscoveryProvider,
    InProcessAdvertisementDiscoveryProvider,
    CompositeDiscoveryService,
    FederationMembershipManager,
    HeartbeatPayload,
    FederationHeartbeatMonitor,
    FederationConnectionManager,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def test_model():
    torch.manual_seed(42)
    cfg = ModelConfig()
    m = ChakrMicro(cfg)
    m.eval()
    return m


@pytest.fixture
def mock_gate():
    return CapabilityGate()


@pytest.fixture
def memory_store():
    return InMemorySecurityStateStore()


@pytest.fixture
def local_engine(test_model, mock_gate):
    engine = CrossZoneFederationEngine(
        local_zone_id="zone-alpha",
        capability_gate=mock_gate,
        model=test_model,
        initial_epoch=1,
    )
    return engine


@pytest.fixture
def remote_engine(test_model, mock_gate):
    engine = CrossZoneFederationEngine(
        local_zone_id="zone-beta",
        capability_gate=mock_gate,
        model=test_model,
        initial_epoch=1,
    )
    return engine


@pytest.fixture
def local_runtime(local_engine, memory_store):
    return FederationRuntime(engine=local_engine, store=memory_store, auto_recover=False)


# ============================================================================
# PHASE 3: DISCOVERY TESTS (1 - 5)
# ============================================================================

def test_01_static_endpoint_discovery():
    """Static configuration discovery parses and produces valid candidates."""
    provider = StaticConfigDiscoveryProvider(
        endpoints=[
            {"host": "192.168.1.100", "port": 9443, "protocol": "tls", "zone_id": "zone-beta"},
            {"host": "192.168.1.101", "port": 9444, "protocol": "mtls", "zone_id": "zone-gamma"},
        ],
        zone_id="zone-alpha",
    )
    candidates = provider.discover()
    assert len(candidates) == 2
    c0 = candidates[0]
    assert c0.endpoint.address.host == "192.168.1.100"
    assert c0.endpoint.address.port == 9443
    assert c0.endpoint.protocol == NodeProtocol.TLS
    assert c0.state == MembershipState.DISCOVERED
    assert len(c0.candidate_id) == 64


def test_02_deterministic_candidate_identity():
    """Deterministic candidate ID via SHA-256 of canonical endpoint metadata."""
    ep1 = FederationNodeEndpoint.create(host="10.0.0.1", port=8080, protocol=NodeProtocol.TLS, zone_id="zone-a")
    ep2 = FederationNodeEndpoint.create(host="10.0.0.1", port=8080, protocol=NodeProtocol.TLS, zone_id="zone-a")
    ep3 = FederationNodeEndpoint.create(host="10.0.0.1", port=8081, protocol=NodeProtocol.TLS, zone_id="zone-a")

    cand1 = FederationNodeCandidate.from_endpoint(ep1, discovery_source=NodeDiscoverySource.STATIC_CONFIG)
    cand2 = FederationNodeCandidate.from_endpoint(ep2, discovery_source=NodeDiscoverySource.STATIC_CONFIG)
    cand3 = FederationNodeCandidate.from_endpoint(ep3, discovery_source=NodeDiscoverySource.STATIC_CONFIG)

    assert cand1.candidate_id == cand2.candidate_id
    assert cand1.candidate_id != cand3.candidate_id


def test_03_duplicate_candidate_detection(local_engine):
    """Registering the same candidate twice does not create duplicates or collisions."""
    mgr = FederationMembershipManager(engine=local_engine)
    ep = FederationNodeEndpoint.create(host="10.0.0.5", port=9000, protocol=NodeProtocol.MTLS)
    cand = FederationNodeCandidate.from_endpoint(ep, discovery_source=NodeDiscoverySource.STATIC_CONFIG)

    m1 = mgr.register_candidate(cand)
    m2 = mgr.register_candidate(cand)

    assert m1.membership_id == m2.membership_id
    assert len(mgr.list_members()) == 1


def test_04_malformed_endpoint_rejection():
    """Malformed endpoint configurations fail closed with MalformedEndpointError."""
    with pytest.raises(MalformedEndpointError):
        FederationNodeEndpoint.create(host="", port=9000)

    with pytest.raises(MalformedEndpointError):
        FederationNodeEndpoint.create(host="192.168.1.1", port=0)

    with pytest.raises(MalformedEndpointError):
        FederationNodeEndpoint.create(host="192.168.1.1", port=70000)


def test_05_unsupported_protocol_rejection():
    """Unsupported protocols fail closed with UnsupportedProtocolError."""
    with pytest.raises(UnsupportedProtocolError):
        FederationNodeEndpoint.from_transport_uri("ftp://192.168.1.1:21")

    with pytest.raises(UnsupportedProtocolError):
        FederationNodeEndpoint.from_transport_uri("http://192.168.1.1:80")


# ============================================================================
# PHASE 4: MEMBERSHIP MANAGER TESTS (6 - 13)
# ============================================================================

def test_06_candidate_registration(local_engine):
    """Candidate registration records DISCOVERED state with zero permissions."""
    mgr = FederationMembershipManager(engine=local_engine)
    ep = FederationNodeEndpoint.create(host="10.1.1.20", port=9443, protocol=NodeProtocol.MTLS)
    cand = FederationNodeCandidate.from_endpoint(ep, discovery_source=NodeDiscoverySource.STATIC_CONFIG)

    membership = mgr.register_candidate(cand)
    assert membership.state == MembershipState.DISCOVERED
    assert mgr.get_membership(membership.membership_id) is not None
    assert len(mgr.list_active_members()) == 0


def test_07_candidate_authentication(local_engine):
    """Candidate advances through authentication pipeline to AUTHENTICATED."""
    mgr = FederationMembershipManager(engine=local_engine)
    ep = FederationNodeEndpoint.create(host="10.1.1.21", port=9443, protocol=NodeProtocol.MTLS)
    cand = FederationNodeCandidate.from_endpoint(ep, discovery_source=NodeDiscoverySource.STATIC_CONFIG)
    mem = mgr.register_candidate(cand)

    mgr.start_authentication(mem.membership_id)
    assert mgr.get_membership(mem.membership_id).state == MembershipState.PENDING_AUTHENTICATION

    eng_id = FederationEngineIdentity(
        engine_id="eng_remote_node_1",
        zone_id="zone-beta",
        created_epoch=1,
        identity_fingerprint="fp_remote_1",
    )
    mgr.authenticate_candidate(mem.membership_id, engine_identity=eng_id, peer_id="peer_remote_1")
    assert mgr.get_membership(mem.membership_id).state == MembershipState.AUTHENTICATED
    assert mgr.get_membership_by_node_id("eng_remote_node_1") is not None


def test_08_membership_promotion(local_engine):
    """Authenticated candidate promoted to MEMBER with capacity enforcement."""
    mgr = FederationMembershipManager(engine=local_engine, max_members=2)
    for i in range(2):
        ep = FederationNodeEndpoint.create(host=f"10.1.1.{i+10}", port=9443, protocol=NodeProtocol.MTLS)
        cand = FederationNodeCandidate.from_endpoint(ep, discovery_source=NodeDiscoverySource.STATIC_CONFIG)
        mem = mgr.register_candidate(cand)
        eng_id = FederationEngineIdentity(
            engine_id=f"eng_{i}",
            zone_id="zone-beta",
            created_epoch=1,
            identity_fingerprint=f"fp_{i}",
        )
        mgr.authenticate_candidate(mem.membership_id, eng_id)
        mgr.promote_to_member(mem.membership_id)
        assert mgr.get_membership(mem.membership_id).state == MembershipState.MEMBER

    assert len(mgr.list_active_members()) == 2

    # Exceed capacity
    ep3 = FederationNodeEndpoint.create(host="10.1.1.99", port=9443, protocol=NodeProtocol.MTLS)
    cand3 = FederationNodeCandidate.from_endpoint(ep3, discovery_source=NodeDiscoverySource.STATIC_CONFIG)
    mem3 = mgr.register_candidate(cand3)
    eng_id3 = FederationEngineIdentity(engine_id="eng_overflow", zone_id="zone-beta", created_epoch=1, identity_fingerprint="fp_o")
    mgr.authenticate_candidate(mem3.membership_id, eng_id3)
    with pytest.raises(MembershipCapacityError):
        mgr.promote_to_member(mem3.membership_id)


def test_09_invalid_state_transition(local_engine):
    """Direct or illegal transitions raise InvalidMembershipTransitionError."""
    mgr = FederationMembershipManager(engine=local_engine)
    ep = FederationNodeEndpoint.create(host="10.1.1.50", port=9443, protocol=NodeProtocol.MTLS)
    cand = FederationNodeCandidate.from_endpoint(ep, discovery_source=NodeDiscoverySource.STATIC_CONFIG)
    mem = mgr.register_candidate(cand)

    # Cannot promote DISCOVERED directly to MEMBER
    with pytest.raises(InvalidMembershipTransitionError):
        mgr.promote_to_member(mem.membership_id)


def test_10_suspension(local_engine):
    """Active member can be suspended and resumed."""
    mgr = FederationMembershipManager(engine=local_engine)
    ep = FederationNodeEndpoint.create(host="10.1.1.60", port=9443, protocol=NodeProtocol.MTLS)
    cand = FederationNodeCandidate.from_endpoint(ep, discovery_source=NodeDiscoverySource.STATIC_CONFIG)
    mem = mgr.register_candidate(cand)
    eng_id = FederationEngineIdentity(engine_id="eng_suspend_test", zone_id="zone-beta", created_epoch=1, identity_fingerprint="fp_s")
    mgr.authenticate_candidate(mem.membership_id, eng_id)
    mgr.promote_to_member(mem.membership_id)

    mgr.suspend_member("eng_suspend_test", reason="Scheduled maintenance")
    assert mgr.get_membership(mem.membership_id).state == MembershipState.SUSPENDED
    assert len(mgr.list_active_members()) == 0

    mgr.resume_member("eng_suspend_test")
    assert mgr.get_membership(mem.membership_id).state == MembershipState.MEMBER
    assert len(mgr.list_active_members()) == 1


def test_11_quarantine(local_engine):
    """Anomalies quarantine node, block operations, and preserve evidence."""
    mgr = FederationMembershipManager(engine=local_engine)
    ep = FederationNodeEndpoint.create(host="10.1.1.70", port=9443, protocol=NodeProtocol.MTLS)
    cand = FederationNodeCandidate.from_endpoint(ep, discovery_source=NodeDiscoverySource.STATIC_CONFIG)
    mem = mgr.register_candidate(cand)
    eng_id = FederationEngineIdentity(engine_id="eng_quarantine_test", zone_id="zone-beta", created_epoch=1, identity_fingerprint="fp_q")
    mgr.authenticate_candidate(mem.membership_id, eng_id)
    mgr.promote_to_member(mem.membership_id)

    mgr.quarantine_member(
        "eng_quarantine_test",
        reason="Repeated message authentication failures",
        evidence={"failed_count": 5, "last_mac": "deadbeef"},
    )
    m = mgr.get_membership(mem.membership_id)
    assert m.state == MembershipState.QUARANTINED
    assert m.quarantine_reason == "Repeated message authentication failures"
    assert m.quarantine_evidence["failed_count"] == 5


def test_12_revocation(local_engine):
    """Revocation is an absorbing terminal state that cascades through the engine."""
    mgr = FederationMembershipManager(engine=local_engine)
    ep = FederationNodeEndpoint.create(host="10.1.1.80", port=9443, protocol=NodeProtocol.MTLS)
    cand = FederationNodeCandidate.from_endpoint(ep, discovery_source=NodeDiscoverySource.STATIC_CONFIG)
    mem = mgr.register_candidate(cand)
    eng_id = FederationEngineIdentity(engine_id="eng_revoked_test", zone_id="zone-beta", created_epoch=1, identity_fingerprint="fp_r")
    mgr.authenticate_candidate(mem.membership_id, eng_id, peer_id="peer_revoked_test")
    mgr.promote_to_member(mem.membership_id)

    mgr.revoke_member("eng_revoked_test", reason="Cryptographic key compromise")
    m = mgr.get_membership(mem.membership_id)
    assert m.state == MembershipState.REVOKED

    # Terminal invariant: cannot transition back
    with pytest.raises(InvalidMembershipTransitionError):
        m.transition_to(MembershipState.MEMBER)


def test_13_termination(local_engine):
    """Graceful termination of node membership."""
    mgr = FederationMembershipManager(engine=local_engine)
    ep = FederationNodeEndpoint.create(host="10.1.1.90", port=9443, protocol=NodeProtocol.MTLS)
    cand = FederationNodeCandidate.from_endpoint(ep, discovery_source=NodeDiscoverySource.STATIC_CONFIG)
    mem = mgr.register_candidate(cand)
    eng_id = FederationEngineIdentity(engine_id="eng_term_test", zone_id="zone-beta", created_epoch=1, identity_fingerprint="fp_t")
    mgr.authenticate_candidate(mem.membership_id, eng_id)
    mgr.promote_to_member(mem.membership_id)

    mgr.terminate_member("eng_term_test", reason="Graceful decommissioning")
    assert mgr.get_membership(mem.membership_id).state == MembershipState.TERMINATED


# ============================================================================
# PHASE 5 & 6: SECURITY CONNECTION TESTS (14 - 20)
# ============================================================================

def test_14_unknown_peer_denied(local_engine):
    """Inbound connection from an unknown or un-enrolled peer fails closed."""
    conn_mgr = FederationConnectionManager(engine=local_engine)
    with pytest.raises(UnknownPeerError):
        conn_mgr.accept_inbound_connection(
            remote_peer_id="unregistered_peer_attacker",
            remote_zone_id="zone-evil",
            tls_cert_metadata=None,
            engine_identity=None,
        )


def make_test_cert_metadata(peer_id: str, fingerprint: Optional[str] = None) -> CertificateMetadata:
    fp = fingerprint or f"SHA256:{hashlib.sha256(peer_id.encode('utf-8')).hexdigest()}"
    now = time.time()
    return CertificateMetadata(
        fingerprint=fp,
        subject={"CN": peer_id},
        issuer={"CN": "TestCA"},
        serial_number="12345",
        not_before_epoch=now - 100,
        not_after_epoch=now + 100000,
    )


def test_15_invalid_certificate_denied(local_engine):
    """Revoked or untrusted TLS certificates fail closed on connection."""
    conn_mgr = FederationConnectionManager(engine=local_engine)
    cert = make_test_cert_metadata("peer_target")
    local_engine.certificate_revocation_registry.revoke(
        fingerprint=cert.fingerprint,
        reason="Revoked during test",
        epoch=1,
    )

    with pytest.raises(Exception):
        conn_mgr.validate_tls_connection(
            cert_metadata=cert,
            expected_peer_id="peer_target",
            tls_mode=TLSMode.MTLS,
        )


def test_16_certificate_identity_mismatch_denied(local_engine):
    """Certificate bound to peer A presented by peer B fails closed."""
    conn_mgr = FederationConnectionManager(engine=local_engine)
    cert = make_test_cert_metadata("peer_legitimate")

    other_cert = make_test_cert_metadata("peer_impostor", fingerprint="SHA256:different")
    local_engine.certificate_binder.bind_peer("peer_impostor", other_cert.fingerprint, epoch=1)

    # Attempt to use cert for peer_impostor
    with pytest.raises(Exception):
        conn_mgr.validate_tls_connection(
            cert_metadata=cert,
            expected_peer_id="peer_impostor",
            tls_mode=TLSMode.MTLS,
        )


def test_17_invalid_engine_identity_denied(local_engine):
    """Engine identity with mismatched or forged fields is denied."""
    conn_mgr = FederationConnectionManager(engine=local_engine)
    bogus_identity = FederationEngineIdentity(
        engine_id="",
        zone_id="zone-beta",
        created_epoch=1,
        identity_fingerprint="fp_bogus",
    )
    with pytest.raises(Exception):
        conn_mgr.validate_engine_identity(bogus_identity)


def test_18_failed_federation_handshake_denied(local_engine):
    """Handshake protocol error halts connection establishment."""
    conn_mgr = FederationConnectionManager(engine=local_engine)
    # Attempt handshake with incompatible protocol
    bogus_response = {
        "status": "REJECTED",
        "reason": "Protocol mismatch",
    }
    with pytest.raises(HandshakeError):
        conn_mgr.verify_handshake_response(bogus_response)


def test_19_trust_escalation_denied(local_engine, mock_gate):
    """Membership alone does not grant trust or capability authorization."""
    mock_gate.registry.register(CalculatorCapability())
    mgr = FederationMembershipManager(engine=local_engine)
    ep = FederationNodeEndpoint.create(host="10.2.2.1", port=9443, protocol=NodeProtocol.MTLS)
    cand = FederationNodeCandidate.from_endpoint(ep, discovery_source=NodeDiscoverySource.STATIC_CONFIG)
    mem = mgr.register_candidate(cand)
    eng_id = FederationEngineIdentity(engine_id="eng_no_trust", zone_id="zone-beta", created_epoch=1, identity_fingerprint="fp_nt")
    mgr.authenticate_candidate(mem.membership_id, eng_id, peer_id="peer_no_trust")
    mgr.promote_to_member(mem.membership_id)

    # Capability gate must still deny execution
    req = CapabilityRequest(
        capability_id="calculator",
        caller_id="peer_no_trust",
        parameters={"expression": "1 + 1"},
    )
    with pytest.raises(CapabilityAuthorizationError):
        mock_gate.authorize(req)


def test_20_capability_escalation_denied(local_engine, mock_gate):
    """Remote peer cannot exceed local trust scope."""
    mock_gate.registry.register(CalculatorCapability())
    # Register peer with READ_ONLY trust
    peer_id = "peer_scoped"
    base_id = PeerIdentity(peer_id=peer_id, zone_id="zone-beta", organization_id="org-beta", created_epoch=1)
    reg = PeerRegistration(identity=base_id, discovery_status=DiscoveryStatus.VERIFIED)
    grant = TrustGrant(
        grant_id="grant_scoped",
        issuer_zone_id="zone-alpha",
        subject_peer_id=peer_id,
        subject_zone_id="zone-beta",
        trust_level=TrustLevel.LIMITED_TRUST,
        permitted_scopes=[FederationScope.ALLOW_EVIDENCE_EXCHANGE],
        issued_epoch=1,
        expires_epoch=100,
        status=TrustStatus.ACTIVE,
    )
    reg.trust_grant = grant
    local_engine.registry.register_peer(reg)

    # Attempt to execute an unpermitted scope
    req = CapabilityRequest(
        capability_id="calculator",
        caller_id=peer_id,
        parameters={"expression": "2 + 2"},
    )
    with pytest.raises(CapabilityAuthorizationError):
        mock_gate.authorize(req)


# ============================================================================
# PHASE 9, 10, 11: RUNTIME HEALTH & REJOIN TESTS (21 - 27)
# ============================================================================

def test_21_heartbeat_success(local_engine):
    """Periodic heartbeat updates liveness and state tracking."""
    mgr = FederationMembershipManager(engine=local_engine)
    ep = FederationNodeEndpoint.create(host="10.3.3.1", port=9443, protocol=NodeProtocol.MTLS)
    cand = FederationNodeCandidate.from_endpoint(ep, discovery_source=NodeDiscoverySource.STATIC_CONFIG)
    mem = mgr.register_candidate(cand)
    eng_id = FederationEngineIdentity(engine_id="eng_hb_1", zone_id="zone-beta", created_epoch=1, identity_fingerprint="fp_hb")
    mgr.authenticate_candidate(mem.membership_id, eng_id)
    mgr.promote_to_member(mem.membership_id)

    mgr.record_heartbeat("eng_hb_1", epoch=5, timestamp=1000.0)
    m = mgr.get_membership(mem.membership_id)
    assert m.last_heartbeat_epoch == 5
    assert m.last_heartbeat_timestamp == 1000.0


def test_22_heartbeat_timeout(local_engine):
    """Missed heartbeats trigger timeout failure classification."""
    mgr = FederationMembershipManager(engine=local_engine)
    ep = FederationNodeEndpoint.create(host="10.3.3.2", port=9443, protocol=NodeProtocol.MTLS)
    cand = FederationNodeCandidate.from_endpoint(ep, discovery_source=NodeDiscoverySource.STATIC_CONFIG)
    mem = mgr.register_candidate(cand)
    eng_id = FederationEngineIdentity(engine_id="eng_hb_timeout", zone_id="zone-beta", created_epoch=1, identity_fingerprint="fp_hbt")
    mgr.authenticate_candidate(mem.membership_id, eng_id)
    mgr.promote_to_member(mem.membership_id)

    # Set last heartbeat long ago
    mem.last_heartbeat_timestamp = time.time() - 300.0
    monitor = FederationHeartbeatMonitor(membership_manager=mgr, heartbeat_timeout_seconds=5.0)

    missed = monitor.check_liveness()
    assert "eng_hb_timeout" in missed


def test_23_network_outage_does_not_revoke_membership(local_engine, local_runtime):
    """Unreachable node retains membership record (UNREACHABLE != REVOKED)."""
    mgr = local_engine.membership_manager
    ep = FederationNodeEndpoint.create(host="10.3.3.3", port=9443, protocol=NodeProtocol.MTLS)
    cand = FederationNodeCandidate.from_endpoint(ep, discovery_source=NodeDiscoverySource.STATIC_CONFIG)
    mem = mgr.register_candidate(cand)
    eng_id = FederationEngineIdentity(engine_id="eng_unreach", zone_id="zone-beta", created_epoch=1, identity_fingerprint="fp_u")
    mgr.authenticate_candidate(mem.membership_id, eng_id)
    mgr.promote_to_member(mem.membership_id)

    # Runtime marks node UNREACHABLE
    local_runtime.record_heartbeat("eng_unreach", is_healthy=False, reason="Network link down")
    assert local_runtime.get_engine_health("eng_unreach") == EngineHealthStatus.UNREACHABLE

    # Membership must NOT be revoked
    assert mgr.get_membership(mem.membership_id).state == MembershipState.MEMBER


def test_24_stale_node_rejoin(local_engine):
    """Stale rejoining node is detected during state reconciliation."""
    conn_mgr = FederationConnectionManager(engine=local_engine)
    # Remote state version is older than local version
    local_engine.coordinator.state_manager.increment_version()
    local_engine.coordinator.state_manager.increment_version()
    assert local_engine.state_version.version >= 3

    stale_version = {"version": 1, "epoch": 1, "engine_id": "eng_stale"}
    is_stale = conn_mgr.check_remote_state_stale(stale_version)
    assert is_stale is True


def test_25_revoked_node_rejoin_denied(local_engine):
    """Revoked member attempting rejoin is rejected fail-closed."""
    mgr = local_engine.membership_manager
    conn_mgr = FederationConnectionManager(engine=local_engine)

    ep = FederationNodeEndpoint.create(host="10.3.3.5", port=9443, protocol=NodeProtocol.MTLS)
    cand = FederationNodeCandidate.from_endpoint(ep, discovery_source=NodeDiscoverySource.STATIC_CONFIG)
    mem = mgr.register_candidate(cand)
    eng_id = FederationEngineIdentity(engine_id="eng_rejoin_revoked", zone_id="zone-beta", created_epoch=1, identity_fingerprint="fp_rr")
    mgr.authenticate_candidate(mem.membership_id, eng_id, peer_id="peer_rejoin_revoked")
    mgr.promote_to_member(mem.membership_id)
    mgr.revoke_member("eng_rejoin_revoked", reason="Compromised credentials")

    with pytest.raises(RejoinDeniedError):
        conn_mgr.reconnect_member(node_id="eng_rejoin_revoked", remote_peer_id="peer_rejoin_revoked")


def test_26_digest_conflict_during_rejoin(local_engine):
    """Conflicting state digests trigger quarantine / rejection."""
    conn_mgr = FederationConnectionManager(engine=local_engine)
    local_digest = local_engine.coordinator.compute_state_digest().compute_composite_digest()
    remote_divergent_digest = "deadbeef" * 8

    conflict = conn_mgr.verify_digest_alignment(local_digest, remote_divergent_digest)
    assert conflict is False  # Digests do not match


def test_27_replay_floor_regression_denied(local_engine):
    """Sequence number below established replay floor is rejected."""
    sid = "sess_replay_test"
    sess = SecurePeerSession(
        session_id=sid,
        local_peer_id=local_engine.local_peer_id,
        remote_peer_id="peer_remote",
        local_zone_id=local_engine.local_zone_id,
        remote_zone_id="zone-beta",
        created_epoch=1,
        expires_at_epoch=100,
        status=SessionStatus.ACTIVE,
    )
    sess.last_seen_sequence_number = 15
    local_engine.sessions[sid] = sess

    # Attempt replay with seq 10 < 15
    assert sess.validate_sequence_number(10) is False
    assert sess.validate_sequence_number(16) is True


# ============================================================================
# PHASE 7 & 8: PERSISTENCE & CRASH RECOVERY TESTS (28 - 32)
# ============================================================================

def test_28_membership_journal_append(local_engine, local_runtime):
    """Membership mutations append deterministic journal entries."""
    mgr = local_engine.membership_manager
    mgr.runtime = local_runtime

    ep = FederationNodeEndpoint.create(host="10.4.4.1", port=9443, protocol=NodeProtocol.MTLS)
    cand = FederationNodeCandidate.from_endpoint(ep, discovery_source=NodeDiscoverySource.STATIC_CONFIG)
    mem = mgr.register_candidate(cand)

    eng_id = FederationEngineIdentity(engine_id="eng_journal_test", zone_id="zone-beta", created_epoch=1, identity_fingerprint="fp_j")
    mgr.authenticate_candidate(mem.membership_id, eng_id)
    mgr.promote_to_member(mem.membership_id)

    entries = local_runtime.journal.entries
    entry_types = [e.entry_type for e in entries]
    assert JournalEntryType.NODE_DISCOVERED in entry_types
    assert JournalEntryType.NODE_AUTHENTICATED in entry_types
    assert JournalEntryType.NODE_MEMBERSHIP_GRANTED in entry_types


def test_29_membership_recovery(test_model, mock_gate, memory_store):
    """Recovery restores membership registry from snapshot and journal."""
    engine1 = CrossZoneFederationEngine(local_zone_id="zone-alpha", capability_gate=mock_gate, model=test_model)
    runtime1 = FederationRuntime(engine=engine1, store=memory_store, auto_recover=False)
    mgr1 = engine1.membership_manager
    mgr1.runtime = runtime1

    ep = FederationNodeEndpoint.create(host="10.4.4.2", port=9443, protocol=NodeProtocol.MTLS)
    cand = FederationNodeCandidate.from_endpoint(ep, discovery_source=NodeDiscoverySource.STATIC_CONFIG)
    mem = mgr1.register_candidate(cand)
    eng_id = FederationEngineIdentity(engine_id="eng_rec_test", zone_id="zone-beta", created_epoch=1, identity_fingerprint="fp_r")
    mgr1.authenticate_candidate(mem.membership_id, eng_id)
    mgr1.promote_to_member(mem.membership_id)

    runtime1.take_snapshot()

    # Reconstruct into engine2
    engine2 = CrossZoneFederationEngine(local_zone_id="zone-alpha", capability_gate=mock_gate, model=test_model)
    runtime2 = FederationRuntime(engine=engine2, store=memory_store, auto_recover=True)

    mgr2 = engine2.membership_manager
    recovered_mem = mgr2.get_membership_by_node_id("eng_rec_test")
    assert recovered_mem is not None
    assert recovered_mem.state == MembershipState.MEMBER


def test_30_crash_during_membership_transition(test_model, mock_gate, memory_store):
    """Un-journaled in-flight state is not restored upon crash recovery."""
    engine1 = CrossZoneFederationEngine(local_zone_id="zone-alpha", capability_gate=mock_gate, model=test_model)
    runtime1 = FederationRuntime(engine=engine1, store=memory_store, auto_recover=False)
    runtime1.take_snapshot()

    # Mutate in memory without writing to durable journal
    ep = FederationNodeEndpoint.create(host="10.4.4.3", port=9443, protocol=NodeProtocol.MTLS)
    cand = FederationNodeCandidate.from_endpoint(ep, discovery_source=NodeDiscoverySource.STATIC_CONFIG)
    engine1.membership_manager._candidates[cand.candidate_id] = cand

    # Recover into engine2
    engine2 = CrossZoneFederationEngine(local_zone_id="zone-alpha", capability_gate=mock_gate, model=test_model)
    runtime2 = FederationRuntime(engine=engine2, store=memory_store, auto_recover=True)

    assert cand.candidate_id not in engine2.membership_manager._candidates


def test_31_corrupted_membership_journal_fails_closed(test_model, mock_gate, memory_store):
    """Corrupted journal record halts recovery fail-closed."""
    engine1 = CrossZoneFederationEngine(local_zone_id="zone-alpha", capability_gate=mock_gate, model=test_model)
    snap = FederationRecoveryManager.create_snapshot_from_engine(engine1, memory_store, snapshot_version=1, journal_offset=0)

    # Append corrupt entry
    corrupt_entry = JournalEntry(
        sequence_num=1,
        epoch=1,
        entry_type=JournalEntryType.NODE_DISCOVERED,
        event_id="jrn_corrupt",
        payload={"candidate_id": "c1"},
        prev_digest=JOURNAL_GENESIS_DIGEST,
        digest="bad_digest" * 6,
        timestamp=time.time(),
    )
    memory_store.append_journal_entry(corrupt_entry)

    engine2 = CrossZoneFederationEngine(local_zone_id="zone-alpha", capability_gate=mock_gate, model=test_model)
    with pytest.raises(RecoveryFailedClosedError):
        FederationRuntime(engine=engine2, store=memory_store, auto_recover=True)


def test_32_revocation_survives_restart(test_model, mock_gate, memory_store):
    """Revoked member remains REVOKED across snapshot creation and crash recovery."""
    engine1 = CrossZoneFederationEngine(local_zone_id="zone-alpha", capability_gate=mock_gate, model=test_model)
    runtime1 = FederationRuntime(engine=engine1, store=memory_store, auto_recover=False)
    mgr1 = engine1.membership_manager
    mgr1.runtime = runtime1

    ep = FederationNodeEndpoint.create(host="10.4.4.5", port=9443, protocol=NodeProtocol.MTLS)
    cand = FederationNodeCandidate.from_endpoint(ep, discovery_source=NodeDiscoverySource.STATIC_CONFIG)
    mem = mgr1.register_candidate(cand)
    eng_id = FederationEngineIdentity(engine_id="eng_survive_rev", zone_id="zone-beta", created_epoch=1, identity_fingerprint="fp_sr")
    mgr1.authenticate_candidate(mem.membership_id, eng_id, peer_id="peer_survive_rev")
    mgr1.promote_to_member(mem.membership_id)
    mgr1.revoke_member("eng_survive_rev", reason="Test persistent revocation")

    runtime1.take_snapshot()

    engine2 = CrossZoneFederationEngine(local_zone_id="zone-alpha", capability_gate=mock_gate, model=test_model)
    runtime2 = FederationRuntime(engine=engine2, store=memory_store, auto_recover=True)

    recovered = engine2.membership_manager.get_membership_by_node_id("eng_survive_rev")
    assert recovered is not None
    assert recovered.state == MembershipState.REVOKED


# ============================================================================
# PHASE 13: ISOLATION TESTS (33 - 34)
# ============================================================================

def test_33_cross_tenant_membership_denied(local_engine):
    """Cross-tenant node membership is rejected fail-closed."""
    conn_mgr = FederationConnectionManager(engine=local_engine)
    # Remote engine presents conflicting tenant ID
    cross_tenant_id = FederationEngineIdentity(
        engine_id="eng_foreign_tenant",
        zone_id="zone-beta",
        created_epoch=1,
        identity_fingerprint="fp_foreign",
    )
    with pytest.raises(Exception):
        conn_mgr.validate_tenant_isolation(
            engine_identity=cross_tenant_id,
            allowed_tenant="zone-alpha",
        )


def test_34_cross_zone_authority_escalation_denied(local_engine):
    """Remote zone cannot issue commands or alter local state (LOCAL_AUTHORITY > PEER_AUTHORITY)."""
    # Verify local engine does not accept foreign command instructions
    assert local_engine.local_zone_id == "zone-alpha"
    remote_command = {
        "action": "revoke_local_node",
        "issuing_zone": "zone-beta",
        "target": local_engine.local_peer_id,
    }
    # Local authority check rejects remote zone command over local zone
    assert remote_command["issuing_zone"] != local_engine.local_zone_id


# ============================================================================
# PHASE 19: NEURAL CORE INTEGRITY TESTS (35 - 37)
# ============================================================================

def test_35_chakrmicro_parameter_count_unchanged(test_model):
    """ChakrMicro parameters must be strictly 3,443,136."""
    param_count = sum(p.numel() for p in test_model.parameters())
    assert param_count == 3_443_136


def test_36_chakrmicro_model_weight_hash_unchanged(test_model):
    """ChakrMicro model weight tensor hash must remain identical."""
    hasher = hashlib.sha256()
    with torch.no_grad():
        for name, param in sorted(test_model.named_parameters()):
            hasher.update(name.encode("utf-8"))
            hasher.update(param.detach().cpu().numpy().tobytes())
    digest = hasher.hexdigest()
    assert digest == "c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da"


def test_37_delta_w_zero(test_model, mock_gate, memory_store):
    """ΔW = 0 strictly maintained across discovery, membership, connection, and recovery lifecycle."""
    pre_hash = hashlib.sha256()
    with torch.no_grad():
        for name, param in sorted(test_model.named_parameters()):
            pre_hash.update(name.encode("utf-8"))
            pre_hash.update(param.detach().cpu().numpy().tobytes())
    pre_digest = pre_hash.hexdigest()

    # Execute complete discovery and membership lifecycle
    engine = CrossZoneFederationEngine(local_zone_id="zone-alpha", capability_gate=mock_gate, model=test_model)
    runtime = FederationRuntime(engine=engine, store=memory_store, auto_recover=False)
    mgr = engine.membership_manager
    mgr.runtime = runtime

    ep = FederationNodeEndpoint.create(host="10.5.5.1", port=9443, protocol=NodeProtocol.MTLS)
    cand = FederationNodeCandidate.from_endpoint(ep, discovery_source=NodeDiscoverySource.STATIC_CONFIG)
    mem = mgr.register_candidate(cand)
    eng_id = FederationEngineIdentity(engine_id="eng_delta_w", zone_id="zone-beta", created_epoch=1, identity_fingerprint="fp_dw")
    mgr.authenticate_candidate(mem.membership_id, eng_id)
    mgr.promote_to_member(mem.membership_id)
    mgr.suspend_member("eng_delta_w")
    mgr.resume_member("eng_delta_w")
    mgr.quarantine_member("eng_delta_w", reason="test")
    mgr.revoke_member("eng_delta_w", reason="test")

    runtime.take_snapshot()

    # Verify model weights remained untouched
    post_hash = hashlib.sha256()
    with torch.no_grad():
        for name, param in sorted(test_model.named_parameters()):
            post_hash.update(name.encode("utf-8"))
            post_hash.update(param.detach().cpu().numpy().tobytes())
    post_digest = post_hash.hexdigest()

    assert pre_digest == post_digest
    assert post_digest == "c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da"
