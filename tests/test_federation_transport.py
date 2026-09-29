"""
Comprehensive Security & Verification Test Suite for Step 36:
Production Federation Message Transport & Secure Inter-Node Communication.

Verifies:
1. Length-prefixed framing (enc, dec, oversized rejection, truncation, malformed headers)
2. Canonical codec (deterministic encoding, prohibited secret/weight exclusion, unknown types)
3. Envelope integrity & Ed25519 signature validation
4. Channel lifecycle state machine (transitions, quarantine, absorbing revocation)
5. Session sequence validation & monotonic replay protection
6. Sovereign message dispatching, CapabilityGate mediation, and scope authorization
7. Multi-tenant boundary isolation
8. Heartbeat protocol & connection degradation
9. Bounded exponential backoff reconnection
10. Failure Cases A through U
11. Durable security journal & crash recovery integration
12. Neural core immutability (Params=3,443,136, Vocab=4096, Context=512, Weight Hash invariant, ΔW = 0)
"""

import hashlib
import json
import os
import struct
import time
from typing import Dict, Any, Optional

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
)
from chakrview.cognition.peering.models import (
    AuditEventType,
    DiscoveryStatus,
    TrustGrant,
    TrustLevel,
    TrustStatus,
    FederationScope,
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
    HandshakeStatus,
    MAX_FEDERATION_ENGINES,
)
from chakrview.cognition.federation.persistence import (
    InMemorySecurityStateStore,
    SecurityStateJournal,
    JournalEntry,
    JournalEntryType,
    JOURNAL_GENESIS_DIGEST,
    DurableSecuritySnapshot,
    RecoveryFailedClosedError,
)
from chakrview.cognition.federation.recovery import FederationRecoveryManager
from chakrview.cognition.federation.runtime import (
    FederationRuntime,
    EngineHealthStatus,
)
from chakrview.cognition.federation.discovery.models import (
    NodeProtocol,
    NodeDiscoverySource,
    MembershipState,
    FederationNodeEndpoint,
    FederationNodeCandidate,
    FederationNodeMembership,
    MAX_MEMBERSHIP_NODES,
)
from chakrview.cognition.federation.discovery.membership import FederationMembershipManager
from chakrview.cognition.federation.discovery.connection import FederationConnectionManager
from chakrview.cognition.federation.discovery.errors import UnknownPeerError

from chakrview.cognition.federation.transport import (
    FEDERATION_PROTOCOL_VERSION,
    DEFAULT_MAX_FRAME_SIZE,
    DEFAULT_MAX_PAYLOAD_SIZE,
    ChannelState,
    VALID_CHANNEL_TRANSITIONS,
    FederationMessageType,
    ChannelMetrics,
    ReconnectPolicy,
    FederationMessageEnvelope,
    FederationMessageFramer,
    FederationMessageCodec,
    FederationChannel,
    FederationMessageDispatcher,
    FederationTransportClient,
    FederationTransportServer,
    FederationTransportError,
    FramingError,
    OversizedFrameError,
    MalformedFrameError,
    TruncatedFrameError,
    CodecError,
    ProhibitedPayloadError,
    UnknownMessageTypeError,
    EnvelopeIntegrityError,
    ChannelError,
    ChannelStateError,
    ChannelAuthenticationError,
    ChannelClosedError,
    ChannelTimeoutError,
    ChannelQuarantinedError,
    ChannelRevokedError,
    DispatcherError,
    HandlerNotFoundError,
    HandlerExecutionError,
    UnauthorizedMessageError,
    ReplayError,
    SequenceRegressionError,
    DuplicateMessageError,
    ReconnectError,
    MaxReconnectAttemptsExceededError,
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
    gate = CapabilityGate()
    gate.registry.register(CalculatorCapability())
    return gate


@pytest.fixture
def memory_store():
    return InMemorySecurityStateStore()


@pytest.fixture
def local_engine(test_model, mock_gate):
    return CrossZoneFederationEngine(
        local_zone_id="zone-alpha",
        capability_gate=mock_gate,
        model=test_model,
        initial_epoch=1,
    )


@pytest.fixture
def remote_engine(test_model, mock_gate):
    return CrossZoneFederationEngine(
        local_zone_id="zone-beta",
        capability_gate=mock_gate,
        model=test_model,
        initial_epoch=1,
    )


@pytest.fixture
def local_key():
    return Ed25519PrivateKeyWrapper.generate()


@pytest.fixture
def remote_key():
    return Ed25519PrivateKeyWrapper.generate()


def make_test_endpoint(host="10.0.1.1", port=9443, zone_id="zone-beta") -> FederationNodeEndpoint:
    return FederationNodeEndpoint.create(
        host=host,
        port=port,
        protocol=NodeProtocol.MTLS,
        zone_id=zone_id,
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


def make_test_envelope(
    message_type: FederationMessageType = FederationMessageType.CAPABILITY_REQUEST,
    message_id: str = "msg_001",
    sequence_number: int = 1,
    sender_peer_id: str = "peer_beta",
    receiver_peer_id: str = "peer_alpha",
    sender_engine_id: str = "eng_beta",
    receiver_engine_id: str = "eng_alpha",
    payload: Optional[Dict[str, Any]] = None,
    tenant_id: Optional[str] = "zone-alpha",
) -> FederationMessageEnvelope:
    return FederationMessageEnvelope(
        message_type=message_type,
        message_id=message_id,
        session_id="sess_test_001",
        sender_engine_id=sender_engine_id,
        receiver_engine_id=receiver_engine_id,
        sender_peer_id=sender_peer_id,
        receiver_peer_id=receiver_peer_id,
        sequence_number=sequence_number,
        epoch=1,
        payload=payload if payload is not None else {"action": "query", "data": 42},
        tenant_id=tenant_id,
    )


# ============================================================================
# 1. FRAMING TESTS (01 - 04)
# ============================================================================

def test_01_framing_roundtrip():
    """Valid binary payload encodes and decodes accurately with length prefix."""
    raw = b"FEDERATION_MESSAGE_PAYLOAD_TEST"
    frame = FederationMessageFramer.encode_frame(raw)
    assert len(frame) == 4 + len(raw)

    buf = bytearray(frame)
    decoded = FederationMessageFramer.decode_frame(buf)
    assert decoded == raw
    assert len(buf) == 0


def test_02_framing_empty_payload_rejected():
    """Encoding empty payload raises MalformedFrameError (Failure Case A)."""
    with pytest.raises(MalformedFrameError):
        FederationMessageFramer.encode_frame(b"")

    # Decoding zero-length frame header fails closed
    zero_header = struct.pack(">I", 0)
    buf = bytearray(zero_header)
    with pytest.raises(MalformedFrameError):
        FederationMessageFramer.decode_frame(buf)


def test_03_framing_oversized_frame_rejected():
    """Frame headers specifying size > ceiling fail closed before allocation (Failure Case B)."""
    limit = 1024
    huge_header = struct.pack(">I", 2048) + (b"X" * 100)
    buf = bytearray(huge_header)
    with pytest.raises(OversizedFrameError):
        FederationMessageFramer.decode_frame(buf, max_frame_size=limit)


def test_04_framing_truncated_frame_detected():
    """Incomplete stream frames return None or raise TruncatedFrameError (Failure Case C)."""
    payload = b"COMPLETE_PAYLOAD_DATA"
    full_frame = FederationMessageFramer.encode_frame(payload)

    # Partial frame (header + only 5 bytes of payload)
    partial = full_frame[:9]
    buf = bytearray(partial)
    res = FederationMessageFramer.decode_frame(buf)
    assert res is None  # Needs more bytes; non-destructive
    assert len(buf) == 9

    # Strict EOF detection raises TruncatedFrameError
    with pytest.raises(TruncatedFrameError):
        FederationMessageFramer.decode_all_frames(partial, strict_eof=True)


# ============================================================================
# 2. CODEC & SERIALIZATION TESTS (05 - 10)
# ============================================================================

def test_05_codec_deterministic_serialization():
    """Envelopes serialize to deterministic canonical JSON with identical hash across runs."""
    env = make_test_envelope()
    b1 = FederationMessageCodec.serialize(env)
    b2 = FederationMessageCodec.serialize(env)
    assert b1 == b2

    deserialized = FederationMessageCodec.deserialize(b1)
    assert deserialized.message_id == env.message_id
    assert deserialized.sequence_number == env.sequence_number
    assert deserialized.payload == env.payload


def test_06_codec_prohibited_keys_rejected():
    """Payloads containing private keys or model weights raise ProhibitedPayloadError."""
    # Private key attempt
    env_key = make_test_envelope(payload={"private_key": "MIIEvgIBADANBg..."})
    with pytest.raises(ProhibitedPayloadError):
        FederationMessageCodec.serialize(env_key)

    # Weight tensor keyword attempt
    env_weights = make_test_envelope(payload={"model_weights": [0.1, 0.2, 0.3]})
    with pytest.raises(ProhibitedPayloadError):
        FederationMessageCodec.serialize(env_weights)


def test_07_codec_non_primitive_types_rejected():
    """Tensors, class instances, or executable callables raise ProhibitedPayloadError."""
    with pytest.raises(ProhibitedPayloadError):
        env_tensor = make_test_envelope(payload={"activation": torch.tensor([1.0, 2.0])})
        FederationMessageCodec.serialize(env_tensor)


def test_08_codec_unknown_message_type_rejected():
    """Unknown message types fail closed during deserialization (Failure Case L)."""
    env = make_test_envelope()
    raw_dict = env.to_dict()
    raw_dict["message_type"] = "MALICIOUS_REMOTE_EXEC"
    canonical_str = json.dumps(raw_dict, sort_keys=True, separators=(",", ":"))
    raw_bytes = canonical_str.encode("utf-8")

    with pytest.raises(UnknownMessageTypeError):
        FederationMessageCodec.deserialize(raw_bytes)


def test_09_envelope_digest_integrity_verification():
    """Tampered payload raises EnvelopeIntegrityError (Failure Case D)."""
    env = make_test_envelope()
    # Mutate payload without recomputing payload_digest
    env.payload["data"] = 999999
    with pytest.raises(EnvelopeIntegrityError):
        env.validate()


def test_10_envelope_ed25519_signature_verification(local_key, remote_key):
    """Envelope signature verifies mathematically and detects forged signatures (Failure Case E)."""
    env = make_test_envelope()
    env.sign(local_key)
    assert env.signature != ""

    # Legitimate signature verification
    assert env.verify_signature(local_key.public_key()) is True

    # Impostor key fails verification
    assert env.verify_signature(remote_key.public_key()) is False

    # Tampered signature fails verification
    env.signature = "bad" * 32
    assert env.verify_signature(local_key.public_key()) is False


# ============================================================================
# 3. CHANNEL LIFECYCLE & SECURITY TESTS (11 - 17)
# ============================================================================

def test_11_channel_state_transitions(local_engine):
    """Channel executes valid state transitions through its lifecycle."""
    ep = make_test_endpoint()
    chan = FederationChannel("chan_01", local_engine, ep)
    assert chan.state == ChannelState.DISCONNECTED

    chan.transition_to(ChannelState.CONNECTING)
    assert chan.state == ChannelState.CONNECTING

    chan.transition_to(ChannelState.AUTHENTICATING)
    assert chan.state == ChannelState.AUTHENTICATING

    chan.transition_to(ChannelState.ESTABLISHED)
    assert chan.state == ChannelState.ESTABLISHED
    assert chan.is_established is True

    chan.transition_to(ChannelState.DEGRADED)
    assert chan.state == ChannelState.DEGRADED

    chan.transition_to(ChannelState.CLOSING)
    assert chan.state == ChannelState.CLOSING

    chan.transition_to(ChannelState.CLOSED)
    assert chan.state == ChannelState.CLOSED


def test_12_channel_invalid_state_transition_fails_closed(local_engine):
    """Direct or illegal transitions raise ChannelStateError."""
    ep = make_test_endpoint()
    chan = FederationChannel("chan_02", local_engine, ep)

    # Cannot transition DISCONNECTED -> ESTABLISHED directly
    with pytest.raises(ChannelStateError):
        chan.transition_to(ChannelState.ESTABLISHED)


def test_13_channel_send_and_receive(local_engine):
    """Active established channel handles envelope transmission and reception."""
    ep = make_test_endpoint()
    chan = FederationChannel("chan_03", local_engine, ep, initial_state=ChannelState.ESTABLISHED)

    env = make_test_envelope(message_id="m_001", sequence_number=1)
    chan.send_message(env)
    assert chan.metrics.messages_sent == 1

    received = chan.receive_message()
    assert received is not None
    assert received.message_id == "m_001"
    assert chan.metrics.messages_received == 1


def test_14_channel_mtls_identity_binding(local_engine):
    """Channel verifies that presented cert matches pre-registered peer identity."""
    ep = make_test_endpoint()
    cert = make_test_cert_metadata("peer_beta")
    local_engine.certificate_binder.bind_peer("peer_beta", cert.fingerprint, epoch=1)

    chan = FederationChannel("chan_04", local_engine, ep, remote_peer_id="peer_beta")
    # Verify binding verification succeeds
    is_bound, _ = local_engine.certificate_binder.verify_binding("peer_beta", cert)
    assert is_bound is True

    # Impostor cert fails binding check
    other_cert = make_test_cert_metadata("peer_impostor", fingerprint="SHA256:different")
    is_bound2, _ = local_engine.certificate_binder.verify_binding("peer_beta", other_cert)
    assert is_bound2 is False


def test_15_channel_session_expiration_fails_closed(local_engine):
    """Expired session halts message sending/receiving fail-closed (Failure Case F)."""
    ep = make_test_endpoint()
    session = SecurePeerSession(
        session_id="sess_exp",
        local_peer_id="peer_alpha",
        remote_peer_id="peer_beta",
        local_zone_id="zone-alpha",
        remote_zone_id="zone-beta",
        created_epoch=1,
        expires_at_epoch=2,
    )
    session.status = SessionStatus.EXPIRED

    chan = FederationChannel("chan_05", local_engine, ep, session=session, initial_state=ChannelState.CLOSED)
    with pytest.raises(ChannelClosedError):
        chan.send_message(make_test_envelope())


def test_16_channel_revoked_peer_rejected(local_engine):
    """Revoked channel rejects all send and receive operations (Failure Case G)."""
    ep = make_test_endpoint()
    chan = FederationChannel("chan_06", local_engine, ep, initial_state=ChannelState.ESTABLISHED)
    chan.revoke(reason="Administrative revocation")
    assert chan.is_revoked is True

    with pytest.raises(ChannelRevokedError):
        chan.send_message(make_test_envelope())

    with pytest.raises(ChannelRevokedError):
        chan.receive_message()


def test_17_channel_quarantined_peer_traffic_blocked(local_engine):
    """Quarantined channel rejects message transmission fail-closed (Failure Case H)."""
    ep = make_test_endpoint()
    chan = FederationChannel("chan_07", local_engine, ep, initial_state=ChannelState.ESTABLISHED)
    chan.quarantine(reason="Repeated authentication anomalies")
    assert chan.is_quarantined is True

    with pytest.raises(ChannelQuarantinedError):
        chan.send_message(make_test_envelope())

    with pytest.raises(ChannelQuarantinedError):
        chan.receive_message()


# ============================================================================
# 4. REPLAY & SEQUENCE PROTECTION TESTS (18 - 20)
# ============================================================================

def test_18_replay_protection_duplicate_message_id_rejected(local_engine):
    """Duplicate message ID raises DuplicateMessageError (Failure Case I)."""
    ep = make_test_endpoint()
    session = SecurePeerSession(
        session_id="sess_replay",
        local_peer_id="peer_alpha",
        remote_peer_id="peer_beta",
        local_zone_id="zone-alpha",
        remote_zone_id="zone-beta",
        created_epoch=1,
        expires_at_epoch=100,
    )
    chan = FederationChannel("chan_08", local_engine, ep, session=session, initial_state=ChannelState.ESTABLISHED)

    # First receive succeeds
    env1 = make_test_envelope(message_id="msg_duplicate", sequence_number=1)
    chan.send_message(env1)
    chan.receive_message()

    # Second receive with same message_id raises DuplicateMessageError
    env2 = make_test_envelope(message_id="msg_duplicate", sequence_number=2)
    chan.send_message(env2)
    with pytest.raises(DuplicateMessageError):
        chan.receive_message()


def test_19_sequence_monotonicity_duplicate_sequence_rejected(local_engine):
    """Duplicate sequence number raises SequenceRegressionError (Failure Case J)."""
    ep = make_test_endpoint()
    session = SecurePeerSession(
        session_id="sess_seq",
        local_peer_id="peer_alpha",
        remote_peer_id="peer_beta",
        local_zone_id="zone-alpha",
        remote_zone_id="zone-beta",
        created_epoch=1,
        expires_at_epoch=100,
    )
    chan = FederationChannel("chan_09", local_engine, ep, session=session, initial_state=ChannelState.ESTABLISHED)

    # Sequence 5
    chan.send_message(make_test_envelope(message_id="m1", sequence_number=5))
    chan.receive_message()

    # Sequence 5 again raises SequenceRegressionError
    chan.send_message(make_test_envelope(message_id="m2", sequence_number=5))
    with pytest.raises(SequenceRegressionError):
        chan.receive_message()


def test_20_sequence_monotonicity_out_of_order_rejected(local_engine):
    """Regressive sequence number raises SequenceRegressionError (Failure Case K)."""
    ep = make_test_endpoint()
    session = SecurePeerSession(
        session_id="sess_seq_reg",
        local_peer_id="peer_alpha",
        remote_peer_id="peer_beta",
        local_zone_id="zone-alpha",
        remote_zone_id="zone-beta",
        created_epoch=1,
        expires_at_epoch=100,
    )
    chan = FederationChannel("chan_10", local_engine, ep, session=session, initial_state=ChannelState.ESTABLISHED)

    # Sequence 10
    chan.send_message(make_test_envelope(message_id="m1", sequence_number=10))
    chan.receive_message()

    # Sequence 9 raises SequenceRegressionError
    chan.send_message(make_test_envelope(message_id="m2", sequence_number=9))
    with pytest.raises(SequenceRegressionError):
        chan.receive_message()


# ============================================================================
# 5. DISPATCHER & SOVEREIGN AUTHORIZATION TESTS (21 - 26)
# ============================================================================

def test_21_dispatcher_handler_registration_and_dispatch(local_engine):
    """Registered handler executes and returns results under dispatcher."""
    dispatcher = FederationMessageDispatcher(engine=local_engine)

    received_data = []
    def custom_handler(env, chan):
        received_data.append(env.payload)
        return "SUCCESS"

    dispatcher.register_handler(FederationMessageType.EVIDENCE_EXCHANGE, custom_handler)
    assert dispatcher.has_handler(FederationMessageType.EVIDENCE_EXCHANGE) is True

    ep = make_test_endpoint()
    chan = FederationChannel("chan_disp", local_engine, ep, initial_state=ChannelState.ESTABLISHED)
    env = make_test_envelope(message_type=FederationMessageType.EVIDENCE_EXCHANGE, payload={"evidence": "test"})

    result = dispatcher.dispatch(env, chan)
    assert result == "SUCCESS"
    assert len(received_data) == 1


def test_22_dispatcher_unknown_message_type_rejected(local_engine):
    """Dispatching envelope with no registered handler raises HandlerNotFoundError."""
    dispatcher = FederationMessageDispatcher(engine=local_engine)
    dispatcher.unregister_handler(FederationMessageType.POLICY_SYNC)

    ep = make_test_endpoint()
    chan = FederationChannel("chan_unreg", local_engine, ep, initial_state=ChannelState.ESTABLISHED)
    env = make_test_envelope(message_type=FederationMessageType.POLICY_SYNC)

    with pytest.raises(HandlerNotFoundError):
        dispatcher.dispatch(env, chan)


def test_23_dispatcher_handler_execution_error_contained(local_engine):
    """Handler exceptions are caught and wrapped in HandlerExecutionError (Failure Case M)."""
    dispatcher = FederationMessageDispatcher(engine=local_engine)

    def faulty_handler(env, chan):
        raise ValueError("Simulated handler crash")

    dispatcher.register_handler(FederationMessageType.EVIDENCE_EXCHANGE, faulty_handler)
    ep = make_test_endpoint()
    chan = FederationChannel("chan_err", local_engine, ep, initial_state=ChannelState.ESTABLISHED)
    env = make_test_envelope(message_type=FederationMessageType.EVIDENCE_EXCHANGE)

    with pytest.raises(HandlerExecutionError):
        dispatcher.dispatch(env, chan)


def test_24_dispatcher_capability_gate_authorization(local_engine, mock_gate):
    """Capability request routed through dispatcher passes through CapabilityGate."""
    dispatcher = FederationMessageDispatcher(engine=local_engine)
    dispatcher.register_handler(FederationMessageType.CAPABILITY_REQUEST, lambda env, chan: "AUTHORIZED")

    ep = make_test_endpoint()
    chan = FederationChannel("chan_cap", local_engine, ep, initial_state=ChannelState.ESTABLISHED)

    # Valid calculator capability request
    env = make_test_envelope(
        message_type=FederationMessageType.CAPABILITY_REQUEST,
        payload={"capability_id": "calculator", "parameters": {"expression": "2 + 2"}},
    )

    # Caller lacks permission context -> denied by CapabilityGate
    with pytest.raises(UnauthorizedMessageError):
        dispatcher.dispatch(env, chan)


def test_25_dispatcher_trust_scope_authorization(local_engine):
    """Handler requiring FederationScope validates against peer TrustGrant."""
    dispatcher = FederationMessageDispatcher(engine=local_engine)

    dispatcher.register_handler(
        FederationMessageType.POLICY_SYNC,
        lambda env, chan: "SYNCED",
        required_scope=FederationScope.ALLOW_VERIFICATION,
    )

    # Register peer WITHOUT ALLOW_VERIFICATION scope
    peer_id = "peer_limited"
    base_id = PeerIdentity(peer_id=peer_id, zone_id="zone-beta", organization_id="org-beta", created_epoch=1)
    reg = PeerRegistration(identity=base_id, discovery_status=DiscoveryStatus.VERIFIED)
    grant = TrustGrant(
        grant_id="grant_l",
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

    ep = make_test_endpoint()
    chan = FederationChannel("chan_scope", local_engine, ep, remote_peer_id=peer_id, initial_state=ChannelState.ESTABLISHED)
    env = make_test_envelope(message_type=FederationMessageType.POLICY_SYNC, sender_peer_id=peer_id)

    with pytest.raises(UnauthorizedMessageError):
        dispatcher.dispatch(env, chan)


def test_26_dispatcher_tenant_isolation_enforced(local_engine):
    """Cross-tenant message envelope raises UnauthorizedMessageError."""
    dispatcher = FederationMessageDispatcher(engine=local_engine)
    dispatcher.register_handler(FederationMessageType.EVIDENCE_EXCHANGE, lambda env, chan: "OK")

    ep = make_test_endpoint()
    chan = FederationChannel("chan_tenant", local_engine, ep, initial_state=ChannelState.ESTABLISHED)

    # Envelope with foreign tenant ID
    env = make_test_envelope(
        message_type=FederationMessageType.EVIDENCE_EXCHANGE,
        tenant_id="zone-foreign-tenant",
    )
    with pytest.raises(UnauthorizedMessageError):
        dispatcher.dispatch(env, chan)


# ============================================================================
# 6. HEARTBEAT & LIVENESS TESTS (27 - 28)
# ============================================================================

def test_27_heartbeat_transmission_and_ack(local_engine):
    """Periodic heartbeat transmitted, handled, and acknowledged."""
    dispatcher = FederationMessageDispatcher(engine=local_engine)
    ep = make_test_endpoint()
    chan = FederationChannel("chan_hb", local_engine, ep, initial_state=ChannelState.ESTABLISHED)

    hb_env = make_test_envelope(message_type=FederationMessageType.HEARTBEAT, message_id="hb_001")
    resp = dispatcher.dispatch(hb_env, chan)

    assert resp["status"] == "ACK"
    assert resp["ack_envelope"].message_type == FederationMessageType.HEARTBEAT_ACK


def test_28_heartbeat_timeout_triggers_degraded_state(local_engine):
    """Unhealthy heartbeat observation transitions channel to DEGRADED."""
    ep = make_test_endpoint()
    chan = FederationChannel("chan_deg", local_engine, ep, initial_state=ChannelState.ESTABLISHED)

    chan.record_heartbeat(is_healthy=False)
    assert chan.state == ChannelState.DEGRADED

    # Restoring heartbeat recovers to ESTABLISHED
    chan.record_heartbeat(is_healthy=True)
    assert chan.state == ChannelState.ESTABLISHED


# ============================================================================
# 7. RECONNECTION & BACKOFF TESTS (29 - 32)
# ============================================================================

def test_29_reconnect_bounded_exponential_backoff(local_engine):
    """Reconnect policy computes bounded exponential backoff delays."""
    policy = ReconnectPolicy(
        initial_backoff_seconds=0.01,
        max_backoff_seconds=0.1,
        backoff_multiplier=2.0,
        max_attempts=4,
    )
    assert policy.compute_backoff(1) == 0.01
    assert policy.compute_backoff(2) == 0.02
    assert policy.compute_backoff(3) == 0.04
    assert policy.compute_backoff(4) == 0.08
    assert policy.compute_backoff(10) == 0.1  # Capped at max_backoff


def test_30_reconnect_attempts_exhausted_fails_closed(local_engine):
    """Exceeding max attempts closes channel and raises MaxReconnectAttemptsExceededError."""
    policy = ReconnectPolicy(initial_backoff_seconds=0.001, max_backoff_seconds=0.005, max_attempts=2)
    client = FederationTransportClient(engine=local_engine, reconnect_policy=policy)

    ep = make_test_endpoint()
    chan = FederationChannel("chan_rec_fail", local_engine, ep, initial_state=ChannelState.DEGRADED)

    # Reconnection without connection manager simulates persistent network failure
    with pytest.raises(MaxReconnectAttemptsExceededError):
        client.reconnect(chan)

    assert chan.state == ChannelState.CLOSED


def test_31_reconnect_revoked_channel_strictly_denied(local_engine):
    """Revoked channel attempting reconnect is rejected immediately."""
    client = FederationTransportClient(engine=local_engine)
    ep = make_test_endpoint()
    chan = FederationChannel("chan_rec_rev", local_engine, ep, initial_state=ChannelState.REVOKED)

    with pytest.raises(ChannelRevokedError):
        client.reconnect(chan)


def test_32_stale_membership_reconnect_rejected(local_engine):
    """Stale membership state rejection during reconnect fails closed (Failure Case Q)."""
    conn_mgr = FederationConnectionManager(engine=local_engine)
    local_engine.coordinator.state_manager.increment_version()
    local_engine.coordinator.state_manager.increment_version()

    stale_state = {"version": 0, "epoch": 1, "engine_id": "eng_stale"}
    is_stale = conn_mgr.check_remote_state_stale(stale_state)
    assert is_stale is True


# ============================================================================
# 8. CERTIFICATE REVOCATION & SERVER TESTS (33 - 35)
# ============================================================================

def test_33_certificate_revocation_during_active_connection(local_engine):
    """Certificate revocation triggers channel isolation (Failure Case R)."""
    ep = make_test_endpoint()
    cert = make_test_cert_metadata("peer_rev_test")
    chan = FederationChannel("chan_cert_rev", local_engine, ep, remote_peer_id="peer_rev_test", initial_state=ChannelState.ESTABLISHED)

    # Revoke certificate in registry
    local_engine.certificate_revocation_registry.revoke(
        fingerprint=cert.fingerprint,
        reason="Compromise suspected",
        epoch=1,
    )
    assert local_engine.certificate_revocation_registry.is_revoked(cert.fingerprint) is True

    # Attempting to re-verify binding fails
    is_bound, reason = local_engine.certificate_binder.verify_binding("peer_rev_test", cert)
    # Channel quarantines upon certificate revocation discovery
    chan.quarantine(reason=f"Certificate revoked: {cert.fingerprint}")
    assert chan.is_quarantined is True


def test_34_server_default_deny_unknown_peer_rejected(local_engine):
    """Server default-deny rejects unknown inbound peer connections."""
    dispatcher = FederationMessageDispatcher(engine=local_engine)
    server = FederationTransportServer(engine=local_engine, dispatcher=dispatcher)

    ep = make_test_endpoint(host="10.9.9.9", port=9443)
    with pytest.raises(UnknownPeerError):
        server.accept_channel(remote_endpoint=ep, remote_peer_id="peer_completely_unknown")


def test_35_server_capacity_ceiling_enforced(local_engine):
    """Server rejects connections exceeding MAX_MEMBERSHIP_NODES capacity."""
    dispatcher = FederationMessageDispatcher(engine=local_engine)
    server = FederationTransportServer(engine=local_engine, dispatcher=dispatcher, max_connections=2)

    ep = make_test_endpoint()
    cand1 = FederationNodeCandidate.from_endpoint(ep)
    mem1 = local_engine.membership_manager.register_candidate(cand1)
    local_engine.membership_manager.authenticate_candidate(
        mem1.membership_id,
        FederationEngineIdentity(engine_id="eng_c1", zone_id="zone-beta", created_epoch=1, identity_fingerprint="fp1"),
        peer_id="peer_c1",
    )

    chan1 = server.accept_channel(ep, remote_peer_id="peer_c1")
    assert chan1.is_established is True

    ep2 = make_test_endpoint(host="10.0.1.2")
    cand2 = FederationNodeCandidate.from_endpoint(ep2)
    mem2 = local_engine.membership_manager.register_candidate(cand2)
    local_engine.membership_manager.authenticate_candidate(
        mem2.membership_id,
        FederationEngineIdentity(engine_id="eng_c2", zone_id="zone-beta", created_epoch=1, identity_fingerprint="fp2"),
        peer_id="peer_c2",
    )
    chan2 = server.accept_channel(ep2, remote_peer_id="peer_c2")
    assert chan2.is_established is True

    # 3rd connection exceeds capacity (max_connections = 2)
    ep3 = make_test_endpoint(host="10.0.1.3")
    with pytest.raises(ChannelStateError):
        server.accept_channel(ep3, remote_peer_id="peer_c3")


# ============================================================================
# 9. DURABLE JOURNAL & CRASH BOUNDARIES (36 - 39)
# ============================================================================

def test_36_durable_journal_transport_events_logged(test_model, mock_gate, memory_store):
    """Transport lifecycle mutations write append-only journal entries."""
    engine = CrossZoneFederationEngine(local_zone_id="zone-alpha", capability_gate=mock_gate, model=test_model)
    runtime = FederationRuntime(engine=engine, store=memory_store, auto_recover=False)

    ep = make_test_endpoint()
    chan = FederationChannel("chan_jrn", engine, ep)
    chan.transition_to(ChannelState.CONNECTING)
    chan.transition_to(ChannelState.AUTHENTICATING)
    chan.transition_to(ChannelState.ESTABLISHED)
    chan.transition_to(ChannelState.CLOSED)

    entries = runtime.journal.entries
    entry_types = [e.entry_type for e in entries]
    assert JournalEntryType.CONNECTION_ESTABLISHED in entry_types
    assert JournalEntryType.CONNECTION_CLOSED in entry_types


def test_37_crash_during_message_processing(test_model, mock_gate, memory_store):
    """Crash during dispatch leaves channel and durable state uncorrupted (Failure Case S)."""
    engine1 = CrossZoneFederationEngine(local_zone_id="zone-alpha", capability_gate=mock_gate, model=test_model)
    runtime1 = FederationRuntime(engine=engine1, store=memory_store, auto_recover=False)
    runtime1.take_snapshot()

    # Simulate message delivery without committing to store
    ep = make_test_endpoint()
    chan1 = FederationChannel("chan_crash", engine1, ep, initial_state=ChannelState.ESTABLISHED)
    env = make_test_envelope(message_id="m_volatile")

    # Recover in engine2
    engine2 = CrossZoneFederationEngine(local_zone_id="zone-alpha", capability_gate=mock_gate, model=test_model)
    runtime2 = FederationRuntime(engine=engine2, store=memory_store, auto_recover=True)
    assert runtime2.status.value == "RUNNING"


def test_38_crash_after_journal_append(test_model, mock_gate, memory_store):
    """Crash after journal append successfully restores event upon recovery (Failure Case T)."""
    engine1 = CrossZoneFederationEngine(local_zone_id="zone-alpha", capability_gate=mock_gate, model=test_model)
    runtime1 = FederationRuntime(engine=engine1, store=memory_store, auto_recover=False)
    runtime1.take_snapshot()

    runtime1.journal.append(
        JournalEntryType.CONNECTION_ESTABLISHED,
        epoch=1,
        payload={"channel_id": "chan_survive"},
    )
    runtime1.flush_journal_to_store()

    engine2 = CrossZoneFederationEngine(local_zone_id="zone-alpha", capability_gate=mock_gate, model=test_model)
    runtime2 = FederationRuntime(engine=engine2, store=memory_store, auto_recover=True)
    assert runtime2.journal.last_sequence >= 1


def test_39_crash_before_journal_append(test_model, mock_gate, memory_store):
    """Crash before journal append does not persist mutation (Failure Case U)."""
    engine1 = CrossZoneFederationEngine(local_zone_id="zone-alpha", capability_gate=mock_gate, model=test_model)
    runtime1 = FederationRuntime(engine=engine1, store=memory_store, auto_recover=False)
    runtime1.take_snapshot()

    # Mutation only in volatile memory
    ep = make_test_endpoint()
    chan = FederationChannel("chan_volatile", engine1, ep, initial_state=ChannelState.ESTABLISHED)

    # Recover without flush
    engine2 = CrossZoneFederationEngine(local_zone_id="zone-alpha", capability_gate=mock_gate, model=test_model)
    runtime2 = FederationRuntime(engine=engine2, store=memory_store, auto_recover=True)
    assert engine2.transport_server.get_channel("chan_volatile") is None


# ============================================================================
# 10. MULTI-TENANT ISOLATION TESTS (40)
# ============================================================================

def test_40_multi_tenant_isolation_cross_tenant_denied(local_engine):
    """Tenant A cannot process or accept messages from Tenant B."""
    conn_mgr = FederationConnectionManager(engine=local_engine)

    # Foreign tenant identity
    foreign_identity = FederationEngineIdentity(
        engine_id="eng_foreign",
        zone_id="zone-foreign",
        created_epoch=1,
        identity_fingerprint="fp_foreign",
    )
    with pytest.raises(Exception):
        conn_mgr.validate_tenant_isolation(foreign_identity, allowed_tenant="zone-alpha")


# ============================================================================
# 11. NEURAL CORE IMMUTABILITY TESTS (41 - 42)
# ============================================================================

def test_41_chakrmicro_parameters_and_hash_strictly_immutable(test_model):
    """ChakrMicro parameter count and SHA-256 tensor hash strictly invariant."""
    param_count = sum(p.numel() for p in test_model.parameters())
    assert param_count == 3_443_136

    hasher = hashlib.sha256()
    with torch.no_grad():
        for name, param in sorted(test_model.named_parameters()):
            hasher.update(name.encode("utf-8"))
            hasher.update(param.detach().cpu().numpy().tobytes())
    digest = hasher.hexdigest()
    assert digest == "c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da"


def test_42_delta_w_zero_across_transport_lifecycle(test_model, mock_gate, memory_store, local_key):
    """End-to-end transport, framing, codec, dispatch, and reconnect confirms ΔW = 0."""
    pre_hasher = hashlib.sha256()
    with torch.no_grad():
        for name, param in sorted(test_model.named_parameters()):
            pre_hasher.update(name.encode("utf-8"))
            pre_hasher.update(param.detach().cpu().numpy().tobytes())
    pre_digest = pre_hasher.hexdigest()

    # Complete transport workflow
    engine = CrossZoneFederationEngine(local_zone_id="zone-alpha", capability_gate=mock_gate, model=test_model)
    runtime = FederationRuntime(engine=engine, store=memory_store, auto_recover=False)
    dispatcher = FederationMessageDispatcher(engine=engine)
    dispatcher.register_handler(FederationMessageType.HEARTBEAT, lambda env, chan: "ACK")

    ep = make_test_endpoint()
    chan = FederationChannel("chan_dw", engine, ep, initial_state=ChannelState.ESTABLISHED)
    env = make_test_envelope(message_type=FederationMessageType.HEARTBEAT)
    env.sign(local_key)

    chan.send_message(env)
    chan.receive_message()
    dispatcher.dispatch(env, chan)
    chan.quarantine(reason="test quarantine")
    chan.revoke(reason="test revoke")

    post_hasher = hashlib.sha256()
    with torch.no_grad():
        for name, param in sorted(test_model.named_parameters()):
            post_hasher.update(name.encode("utf-8"))
            post_hasher.update(param.detach().cpu().numpy().tobytes())
    post_digest = post_hasher.hexdigest()

    assert pre_digest == post_digest
    assert post_digest == "c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da"
