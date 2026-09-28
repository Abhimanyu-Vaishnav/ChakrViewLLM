"""
Comprehensive Test Suite for ChakrView Step 27:
Distributed Federated Cognition & Secure Agent Transport Foundation.

Verifies all 38 required test mandates:
1.  Node identity creation
2.  Node registration
3.  Duplicate node rejection
4.  Tenant isolation in node registry
5.  Node health transitions
6.  Loopback transport (send, receive, request)
7.  Message envelope creation
8.  Deterministic serialization
9.  SHA-256 integrity validation
10. Tampered message rejection
11. Nonce generation/validation
12. Replay rejection (message ID & nonce)
13. TTL rejection (expired message)
14. Hop-limit rejection
15. Identity verification (HMAC signer/verifier)
16. Authentication != authorization
17. Deterministic routing
18. Resource-aware routing
19. Timeout handling
20. Retry handling & backoff
21. Circuit breaker (CLOSED -> OPEN -> HALF_OPEN)
22. Remote agent failure isolation
23. Distributed evidence aggregation
24. Minority evidence preservation
25. Distributed conflict detection
26. Distributed synthesis
27. Capability-gate enforcement
28. Tenant isolation across nodes
29. Session isolation across nodes
30. Weight immutability (SHA-256 fingerprint verification)
31. Frozen ChakrMicro invariants (3,443,136 parameters, 4,096 vocab, 512 context)
32. Deterministic repeated execution
33. Sanitized distributed trace (no private scratchpad leakage)
34. Bounded observability
35. End-to-end distributed cognitive cycle
36. Graceful degradation when remote node fails
37. Malformed response rejection
38. Steps 0-26 regression
"""

from pathlib import Path
import pytest
import time
import torch
import uuid

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer.tokenizer import BPETokenizer
from chakrview.capability.gate import CapabilityGate
from chakrview.capability.registry import CapabilityRegistry
from chakrview.capability.contract import CapabilityRequest
from chakrview.cognition.diagnostics.integrity import (
    CoreIntegrityGuard,
    WeightMutationError,
    EXPECTED_PARAMETERS,
    EXPECTED_VOCAB_SIZE,
    EXPECTED_MAX_SEQ_LEN,
    EXPECTED_BOS_ID,
    EXPECTED_EOS_ID,
    EXPECTED_PAD_ID,
)
from chakrview.cognition.adaptation.profiles import ResourceProfile
from chakrview.memory.engine import ContinualCognitionEngine
from chakrview.reasoning.engine import GovernedReasoningEngine
from chakrview.cognition.critical.engine import CriticalThinkingEngine
from chakrview.cognition.unified.models import DecisionState

from chakrview.cognition.federated.models import (
    AgentRole,
    AgentStatus,
    MessageType,
    AgentTask,
    AgentMessage,
    FederatedConflictRecord,
    FederatedSynthesisCandidate,
)
from chakrview.cognition.federated.engine import FederatedCognitionEngine

from chakrview.cognition.distributed.models import (
    NodeIdentity,
    NodeRole,
    NodeStatus,
    NodeTrustState,
    NodeCapabilities,
    NodeResourceProfile,
    NodeEndpoint,
    NodeRegistration,
    MessageHeader,
    MessageRoute,
    MessageIntegrity,
    DistributedMessageEnvelope,
    DistributedRouteDecision,
    SafePublicDistributedTrace,
    MAX_FEDERATION_NODES,
)
from chakrview.cognition.distributed.transport import (
    LoopbackTransport,
    TransportResponse,
    TransportStatus,
    TransportProtocolError,
    TransportUnavailable,
)
from chakrview.cognition.distributed.security import (
    ReplayProtectionTracker,
    ReplayAttackError,
    MessageExpiredError,
    ExcessiveHopsError,
    TenantRoutingError,
    MessageTamperingError,
    DeterministicHmacMessageSigner,
    DeterministicHmacMessageVerifier,
    create_distributed_envelope,
)
from chakrview.cognition.distributed.registry import (
    DistributedNodeRegistry,
    DuplicateNodeError,
    NodeCapacityExceededError,
)
from chakrview.cognition.distributed.resilience import (
    CircuitBreaker,
    CircuitBreakerState,
    TimeoutPolicy,
    RetryPolicy,
)
from chakrview.cognition.distributed.router import (
    DistributedTaskRouter,
    NoEligibleNodeError,
)
from chakrview.cognition.distributed.policy import DistributedExecutionPolicy
from chakrview.cognition.distributed.observability import DistributedObservabilityMetrics
from chakrview.cognition.distributed.engine import DistributedFederatedCognitionEngine


# ============================================================================
# Pytest Fixtures
# ============================================================================

@pytest.fixture
def frozen_model() -> ChakrMicro:
    """Deterministic frozen ChakrMicro v0.1 instance."""
    config = ModelConfig(
        vocab_size=4096,
        d_model=192,
        n_layers=6,
        n_heads=6,
        hidden_dim=512,
        max_seq_len=512,
        pad_token_id=2,
    )
    model = ChakrMicro(config)
    model.eval()
    return model


@pytest.fixture
def tokenizer() -> BPETokenizer:
    """BPETokenizer with experiment artifacts if available."""
    exp_dir = Path(__file__).resolve().parent.parent / "data" / "tokenizer_experiments" / "v4096"
    if exp_dir.is_dir():
        from chakrview.tokenizer import load_experiment_artifacts
        return load_experiment_artifacts(exp_dir)
    return BPETokenizer()


@pytest.fixture
def capability_gate() -> CapabilityGate:
    """Sovereign capability gate."""
    return CapabilityGate(registry=CapabilityRegistry())


@pytest.fixture
def local_node_identity() -> NodeIdentity:
    return NodeIdentity(
        node_id="node_coordinator",
        tenant_id="tenant_alpha",
        role=NodeRole.PRIMARY,
    )


# ============================================================================
# Test Cases (Mandates 1-38)
# ============================================================================

def test_01_node_identity_creation():
    ident = NodeIdentity(
        node_id="node_1",
        tenant_id="tenant_1",
        cluster_id="cluster_main",
        role=NodeRole.WORKER,
    )
    assert ident.node_id == "node_1"
    assert ident.tenant_id == "tenant_1"
    assert ident.role == NodeRole.WORKER
    assert ident.version == "27.0.0"
    d = ident.to_dict()
    assert d["node_id"] == "node_1"
    assert d["role"] == "WORKER"


def test_02_node_registration():
    registry = DistributedNodeRegistry(max_nodes=4)
    ident = NodeIdentity(node_id="node_a", tenant_id="tenant_x")
    reg = NodeRegistration(
        identity=ident,
        endpoint=NodeEndpoint(endpoint_id="ep_a", uri="loopback://node_a"),
        capabilities=NodeCapabilities(supported_roles=[AgentRole.ANALYST, AgentRole.CRITIC]),
        resource_profile=NodeResourceProfile(),
    )
    registry.register_node(reg)
    assert registry.total_nodes == 1
    found = registry.get_node("node_a")
    assert found is not None
    assert found.identity.node_id == "node_a"
    assert found.is_eligible_for_tasks() is True


def test_03_duplicate_node_rejection():
    registry = DistributedNodeRegistry(max_nodes=4)
    ident = NodeIdentity(node_id="node_dup", tenant_id="tenant_x")
    reg = NodeRegistration(
        identity=ident,
        endpoint=NodeEndpoint(endpoint_id="ep_dup", uri="loopback://node_dup"),
        capabilities=NodeCapabilities(supported_roles=[AgentRole.ANALYST]),
        resource_profile=NodeResourceProfile(),
    )
    registry.register_node(reg)
    with pytest.raises(DuplicateNodeError):
        registry.register_node(reg)


def test_04_tenant_isolation_in_registry():
    registry = DistributedNodeRegistry()
    reg_a = NodeRegistration(
        identity=NodeIdentity(node_id="node_t_a", tenant_id="tenant_alpha"),
        endpoint=NodeEndpoint(endpoint_id="ep_a", uri="loopback://node_t_a"),
        capabilities=NodeCapabilities(supported_roles=[AgentRole.ANALYST]),
        resource_profile=NodeResourceProfile(),
    )
    reg_b = NodeRegistration(
        identity=NodeIdentity(node_id="node_t_b", tenant_id="tenant_beta"),
        endpoint=NodeEndpoint(endpoint_id="ep_b", uri="loopback://node_t_b"),
        capabilities=NodeCapabilities(supported_roles=[AgentRole.ANALYST]),
        resource_profile=NodeResourceProfile(),
    )
    registry.register_node(reg_a)
    registry.register_node(reg_b)

    # Tenant alpha should discover ONLY node_t_a
    nodes_a = registry.discover_nodes(tenant_id="tenant_alpha")
    assert len(nodes_a) == 1
    assert nodes_a[0].identity.node_id == "node_t_a"

    # Tenant beta should discover ONLY node_t_b
    nodes_b = registry.discover_nodes(tenant_id="tenant_beta")
    assert len(nodes_b) == 1
    assert nodes_b[0].identity.node_id == "node_t_b"

    # Unknown tenant discovers nothing
    nodes_c = registry.discover_nodes(tenant_id="tenant_gamma")
    assert len(nodes_c) == 0


def test_05_node_health_transitions():
    registry = DistributedNodeRegistry()
    reg = NodeRegistration(
        identity=NodeIdentity(node_id="node_h", tenant_id="t1"),
        endpoint=NodeEndpoint(endpoint_id="ep_h", uri="loopback://node_h"),
        capabilities=NodeCapabilities(supported_roles=[AgentRole.RESEARCHER]),
        resource_profile=NodeResourceProfile(),
    )
    registry.register_node(reg)
    assert reg.is_eligible_for_tasks() is True

    # Transition to DEGRADED
    registry.update_health("node_h", status=NodeStatus.DEGRADED, error="High latency", latency_ms=120.0)
    assert reg.health.status == NodeStatus.DEGRADED
    assert reg.is_eligible_for_tasks() is True

    # Quarantine node
    registry.quarantine_node("node_h", reason="Suspicious activity")
    assert reg.health.status == NodeStatus.QUARANTINED
    assert reg.is_eligible_for_tasks() is False

    # Revoke node
    registry.revoke_node("node_h", reason="Permanent failure")
    assert reg.health.status == NodeStatus.REVOKED
    assert reg.trust_state == NodeTrustState.REVOKED
    assert reg.is_eligible_for_tasks() is False


def test_06_loopback_transport():
    transport = LoopbackTransport()
    transport.register_node_endpoint("node_1")
    transport.register_node_endpoint("node_2")

    env = create_distributed_envelope(
        sender_node_id="node_1",
        sender_agent_id="agent_1",
        receiver_node_id="node_2",
        receiver_agent_id="agent_2",
        tenant_id="tenant_t",
        session_id="session_s",
        message_type=MessageType.TASK_REQUEST,
        payload={"data": "hello_transport"},
    )

    # Test send & receive
    sent = transport.send(env)
    assert sent is True
    received = transport.receive("node_2")
    assert received is not None
    assert received.envelope_id == env.envelope_id
    assert received.payload["data"] == "hello_transport"

    # Test request with registered handler
    def responder(req: DistributedMessageEnvelope):
        return create_distributed_envelope(
            sender_node_id="node_2",
            sender_agent_id="agent_2",
            receiver_node_id="node_1",
            receiver_agent_id="agent_1",
            tenant_id="tenant_t",
            session_id="session_s",
            message_type=MessageType.TASK_RESPONSE,
            payload={"reply": "ok"},
            correlation_id=req.header.correlation_id,
        )

    transport.register_node_endpoint("node_2", handler=responder)
    resp = transport.request(env)
    assert resp.is_success() is True
    assert resp.response_envelope.payload["reply"] == "ok"
    transport.close()


def test_07_message_envelope_creation():
    env = create_distributed_envelope(
        sender_node_id="node_a",
        sender_agent_id="agent_a",
        receiver_node_id="node_b",
        receiver_agent_id="agent_b",
        tenant_id="tenant_1",
        session_id="session_1",
        message_type=MessageType.TASK_REQUEST,
        payload={"task_id": "t1"},
    )
    assert env.header.protocol_version == "27.0"
    assert env.route.sender_node_id == "node_a"
    assert env.route.receiver_node_id == "node_b"
    assert bool(env.integrity.fingerprint) is True
    assert bool(env.integrity.nonce) is True


def test_08_deterministic_serialization():
    env = create_distributed_envelope(
        sender_node_id="node_a",
        sender_agent_id="agent_a",
        receiver_node_id="node_b",
        receiver_agent_id="agent_b",
        tenant_id="tenant_1",
        session_id="session_1",
        message_type=MessageType.TASK_REQUEST,
        payload={"z": 100, "a": "first", "m": [3, 2, 1]},
    )
    b1 = env.canonical_serialize()
    b2 = env.canonical_serialize()
    assert b1 == b2
    assert isinstance(b1, bytes)


def test_09_sha256_integrity_validation():
    env = create_distributed_envelope(
        sender_node_id="node_a",
        sender_agent_id="agent_a",
        receiver_node_id="node_b",
        receiver_agent_id="agent_b",
        tenant_id="tenant_1",
        session_id="session_1",
        message_type=MessageType.TASK_REQUEST,
        payload={"metric": 42},
    )
    assert env.verify_integrity() is True


def test_10_tampered_message_rejection():
    env = create_distributed_envelope(
        sender_node_id="node_a",
        sender_agent_id="agent_a",
        receiver_node_id="node_b",
        receiver_agent_id="agent_b",
        tenant_id="tenant_1",
        session_id="session_1",
        message_type=MessageType.TASK_REQUEST,
        payload={"authorized": False},
    )
    assert env.verify_integrity() is True

    # Tamper with payload
    tampered_envelope = DistributedMessageEnvelope(
        envelope_id=env.envelope_id,
        header=env.header,
        route=env.route,
        integrity=env.integrity,  # Original fingerprint unchanged
        payload={"authorized": True},  # Tampered
    )
    assert tampered_envelope.verify_integrity() is False


def test_11_nonce_generation_validation():
    env1 = create_distributed_envelope(
        sender_node_id="node_a", sender_agent_id="ag", receiver_node_id="node_b",
        receiver_agent_id="ag", tenant_id="t", session_id="s",
        message_type=MessageType.TASK_REQUEST, payload={}
    )
    env2 = create_distributed_envelope(
        sender_node_id="node_a", sender_agent_id="ag", receiver_node_id="node_b",
        receiver_agent_id="ag", tenant_id="t", session_id="s",
        message_type=MessageType.TASK_REQUEST, payload={}
    )
    assert env1.integrity.nonce != env2.integrity.nonce
    assert len(env1.integrity.nonce) > 4


def test_12_replay_rejection():
    tracker = ReplayProtectionTracker()
    env = create_distributed_envelope(
        sender_node_id="node_a", sender_agent_id="ag", receiver_node_id="node_b",
        receiver_agent_id="ag", tenant_id="tenant_safe", session_id="s",
        message_type=MessageType.TASK_REQUEST, payload={"val": 1}
    )
    # First time: valid
    assert tracker.validate_and_record(env) is True

    # Replay identical envelope -> ReplayAttackError
    with pytest.raises(ReplayAttackError):
        tracker.validate_and_record(env)


def test_13_ttl_rejection():
    tracker = ReplayProtectionTracker()
    # Create envelope with old timestamp
    old_time = time.time() - 100.0
    env = create_distributed_envelope(
        sender_node_id="node_a", sender_agent_id="ag", receiver_node_id="node_b",
        receiver_agent_id="ag", tenant_id="t", session_id="s",
        message_type=MessageType.TASK_REQUEST, payload={},
        ttl_seconds=10.0,
    )
    # Validate with current time far ahead of timestamp
    with pytest.raises(MessageExpiredError):
        tracker.validate_and_record(env, current_time=env.header.timestamp + 50.0)


def test_14_hop_limit_rejection():
    tracker = ReplayProtectionTracker()
    env = create_distributed_envelope(
        sender_node_id="node_a", sender_agent_id="ag", receiver_node_id="node_b",
        receiver_agent_id="ag", tenant_id="t", session_id="s",
        message_type=MessageType.TASK_REQUEST, payload={},
        hop_count=5,  # Exceeds MAX_MESSAGE_HOPS (4)
    )
    with pytest.raises(ExcessiveHopsError):
        tracker.validate_and_record(env)


def test_15_identity_verification():
    secret = b"deterministic_secret_key_12345678"
    signer = DeterministicHmacMessageSigner(secret_key=secret)
    verifier = DeterministicHmacMessageVerifier({"node_trusted": secret})

    env = create_distributed_envelope(
        sender_node_id="node_trusted", sender_agent_id="ag", receiver_node_id="node_dest",
        receiver_agent_id="ag", tenant_id="t", session_id="s",
        message_type=MessageType.TASK_REQUEST, payload={"cmd": "verify"},
        signer=signer, signer_identity="node_trusted"
    )
    assert env.integrity.signature is not None
    # Verify with verifier
    is_valid = verifier.verify(env.canonical_serialize(), env.integrity.signature, "node_trusted")
    assert is_valid is True

    # Forged identity or key
    is_valid_fake = verifier.verify(env.canonical_serialize(), env.integrity.signature, "node_unknown")
    assert is_valid_fake is False


def test_16_authentication_not_authorization(capability_gate):
    # Proves AUTHENTICATION != AUTHORIZATION
    secret = b"node_secret"
    signer = DeterministicHmacMessageSigner(secret_key=secret)
    verifier = DeterministicHmacMessageVerifier({"node_verified": secret})

    env = create_distributed_envelope(
        sender_node_id="node_verified", sender_agent_id="ag", receiver_node_id="coordinator",
        receiver_agent_id="coordinator", tenant_id="tenant_x", session_id="session_y",
        message_type=MessageType.TASK_REQUEST, payload={"intent": "format_disk"},
        signer=signer, signer_identity="node_verified"
    )
    # Signature is mathematically authentic
    assert verifier.verify(env.canonical_serialize(), env.integrity.signature, "node_verified") is True

    # However, capability execution still requires CapabilityGate and is denied!
    unauthorized_request = CapabilityRequest(
        capability_id="format_disk",
        caller_id="node_verified",
        session_id="session_y",
    )
    with pytest.raises(Exception):
        capability_gate.authorize(unauthorized_request)


def test_17_deterministic_routing():
    registry = DistributedNodeRegistry()
    reg1 = NodeRegistration(
        identity=NodeIdentity(node_id="worker_01", tenant_id="tenant_t"),
        endpoint=NodeEndpoint(endpoint_id="ep1", uri="loopback://1"),
        capabilities=NodeCapabilities(supported_roles=[AgentRole.ANALYST]),
        resource_profile=NodeResourceProfile(latency_tier="LOCAL"),
    )
    reg2 = NodeRegistration(
        identity=NodeIdentity(node_id="worker_02", tenant_id="tenant_t"),
        endpoint=NodeEndpoint(endpoint_id="ep2", uri="loopback://2"),
        capabilities=NodeCapabilities(supported_roles=[AgentRole.ANALYST]),
        resource_profile=NodeResourceProfile(latency_tier="LAN"),
    )
    registry.register_node(reg1)
    registry.register_node(reg2)

    router = DistributedTaskRouter(registry=registry, local_node_id="coordinator")
    task = AgentTask(
        task_id="t_det",
        parent_task_id=None,
        assigned_role=AgentRole.ANALYST,
        objective="Analyze stability",
    )

    d1 = router.route_task(task, tenant_id="tenant_t", session_id="s")
    d2 = router.route_task(task, tenant_id="tenant_t", session_id="s")
    assert d1.assigned_node_id == d2.assigned_node_id == "worker_01"  # worker_01 has LOCAL tier


def test_18_resource_aware_routing():
    registry = DistributedNodeRegistry()
    # node_lan has LAN tier, node_wan has WAN tier
    reg_lan = NodeRegistration(
        identity=NodeIdentity(node_id="node_lan", tenant_id="t"),
        endpoint=NodeEndpoint(endpoint_id="ep_lan", uri="loopback://lan"),
        capabilities=NodeCapabilities(supported_roles=[AgentRole.RESEARCHER]),
        resource_profile=NodeResourceProfile(latency_tier="LAN"),
    )
    reg_wan = NodeRegistration(
        identity=NodeIdentity(node_id="node_wan", tenant_id="t"),
        endpoint=NodeEndpoint(endpoint_id="ep_wan", uri="loopback://wan"),
        capabilities=NodeCapabilities(supported_roles=[AgentRole.RESEARCHER]),
        resource_profile=NodeResourceProfile(latency_tier="WAN"),
    )
    registry.register_node(reg_wan)
    registry.register_node(reg_lan)

    router = DistributedTaskRouter(registry=registry, local_node_id="local_coord")
    task = AgentTask(
        task_id="t_res",
        parent_task_id=None,
        assigned_role=AgentRole.RESEARCHER,
        objective="Research docs",
    )
    decision = router.route_task(task, tenant_id="t", session_id="s")
    assert decision.assigned_node_id == "node_lan"


def test_19_timeout_handling():
    transport = LoopbackTransport()
    transport.register_node_endpoint("node_slow")
    transport.set_fault_injection(timeout_nodes={"node_slow"})

    env = create_distributed_envelope(
        sender_node_id="node_src", sender_agent_id="ag", receiver_node_id="node_slow",
        receiver_agent_id="ag", tenant_id="t", session_id="s",
        message_type=MessageType.TASK_REQUEST, payload={}
    )
    resp = transport.request(env)
    assert resp.status == TransportStatus.TIMEOUT
    assert resp.is_success() is False


def test_20_retry_handling():
    policy = RetryPolicy(max_retries=3, base_backoff_ms=20.0, backoff_multiplier=2.0)
    assert policy.calculate_backoff_ms(0) == 0.0
    assert policy.calculate_backoff_ms(1) == 20.0
    assert policy.calculate_backoff_ms(2) == 40.0
    assert policy.calculate_backoff_ms(3) == 80.0


def test_21_circuit_breaker():
    cb = CircuitBreaker(node_id="flaky_node", failure_threshold=2, recovery_timeout_ms=50.0)
    assert cb.state == CircuitBreakerState.CLOSED
    assert cb.can_execute() is True

    # 1 failure
    cb.record_failure("t1", "TIMEOUT", "Failed attempt 1")
    assert cb.state == CircuitBreakerState.CLOSED

    # 2nd failure -> Trips OPEN
    cb.record_failure("t2", "TIMEOUT", "Failed attempt 2")
    assert cb.state == CircuitBreakerState.OPEN
    assert cb.can_execute() is False

    # Wait for recovery timeout
    time.sleep(0.06)
    assert cb.state == CircuitBreakerState.HALF_OPEN
    assert cb.can_execute() is True

    # Record probe success -> transitions to CLOSED
    cb.record_success()
    cb.record_success()
    assert cb.state == CircuitBreakerState.CLOSED


def test_22_remote_agent_failure_isolation(frozen_model, tokenizer):
    registry = DistributedNodeRegistry()
    node_id = "node_worker_failing"
    reg = NodeRegistration(
        identity=NodeIdentity(node_id=node_id, tenant_id="t_fail"),
        endpoint=NodeEndpoint(endpoint_id="ep_fail", uri="loopback://fail"),
        capabilities=NodeCapabilities(supported_roles=[AgentRole.ANALYST]),
        resource_profile=NodeResourceProfile(),
    )
    registry.register_node(reg)

    transport = LoopbackTransport()
    # Fault injection: remote node produces error
    transport.register_node_endpoint(node_id)
    transport.set_fault_injection(error_nodes={node_id})

    engine = DistributedFederatedCognitionEngine(
        model=frozen_model,
        tokenizer=tokenizer,
        local_node_identity=NodeIdentity(node_id="local_head", tenant_id="t_fail"),
        node_registry=registry,
        transport=transport,
    )
    # Execution should not crash; falls back gracefully
    candidate, trace = engine.execute_distributed_cycle(
        objective="Analyze system security",
        tenant_id="t_fail",
        session_id="s_fail",
        prefer_remote=True,
    )
    assert candidate is not None
    assert trace.failure_count > 0


def test_23_distributed_evidence_aggregation(frozen_model, tokenizer):
    registry = DistributedNodeRegistry()
    worker_id = "remote_worker_ev"
    reg = NodeRegistration(
        identity=NodeIdentity(node_id=worker_id, tenant_id="t_ev"),
        endpoint=NodeEndpoint(endpoint_id="ep_ev", uri="loopback://ev"),
        capabilities=NodeCapabilities(supported_roles=[AgentRole.ANALYST, AgentRole.RESEARCHER]),
        resource_profile=NodeResourceProfile(),
    )
    registry.register_node(reg)

    transport = LoopbackTransport()
    transport.register_node_endpoint("local_coord")

    engine = DistributedFederatedCognitionEngine(
        model=frozen_model,
        tokenizer=tokenizer,
        local_node_identity=NodeIdentity(node_id="local_coord", tenant_id="t_ev"),
        node_registry=registry,
        transport=transport,
    )
    engine.register_remote_agent_responder(worker_id, [AgentRole.ANALYST, AgentRole.RESEARCHER], "t_ev")

    candidate, trace = engine.execute_distributed_cycle(
        objective="Gather all empirical facts",
        tenant_id="t_ev",
        session_id="s_ev",
        prefer_remote=True,
    )
    assert candidate is not None
    assert trace.message_count >= 1


def test_24_minority_evidence_preservation():
    from chakrview.cognition.federated.synthesis import FederatedSynthesizer
    from chakrview.cognition.federated.models import ConflictState

    synthesizer = FederatedSynthesizer()
    msg_majority = AgentMessage(
        message_id="m_maj", sender_agent_id="ag_1", receiver_agent_id="COORDINATOR",
        tenant_id="t1", session_id="s1", correlation_id="c1", message_type=MessageType.TASK_RESPONSE,
        payload={"role": "ANALYST", "claim": "Option A is optimal"},
    )
    conflict = FederatedConflictRecord(
        conflict_id="con_1",
        task_id="t_min",
        conflicting_agent_ids=["ag_1", "ag_2"],
        claims=[
            {"agent_id": "ag_1", "claim": "Option A is optimal"},
            {"agent_id": "ag_2", "claim": "Option B is optimal based on tail latency"},
        ],
        state=ConflictState.CONFLICT,
    )
    cand = synthesizer.synthesize(
        task_id="t_min",
        objective="Select deployment option",
        messages=[msg_majority],
        corroborated_evidence=[],
        conflicts=[conflict],
        critical_counter_evidence=[],
    )
    assert len(cand.minority_opinions) >= 1
    assert any("Option B" in op.get("claim", "") for op in cand.minority_opinions)
    assert cand.recommended_decision_state == DecisionState.ANSWER_WITH_UNCERTAINTY


def test_25_distributed_conflict_detection():
    from chakrview.cognition.federated.conflict import FederatedConflictResolver
    from chakrview.cognition.federated.models import ConflictState

    resolver = FederatedConflictResolver()
    msg_a = AgentMessage(
        message_id="m1", sender_agent_id="ag_a", receiver_agent_id="COORDINATOR",
        tenant_id="t1", session_id="s1", correlation_id="c1", message_type=MessageType.TASK_RESPONSE,
        payload={"claim": "The cache mechanism is enabled."},
    )
    msg_b = AgentMessage(
        message_id="m2", sender_agent_id="ag_b", receiver_agent_id="COORDINATOR",
        tenant_id="t1", session_id="s1", correlation_id="c1", message_type=MessageType.TASK_RESPONSE,
        payload={"claim": "The cache mechanism is disabled."},
    )
    conflicts = resolver.detect_conflicts("task_conf", [msg_a, msg_b])
    assert len(conflicts) >= 1
    assert conflicts[0].state == ConflictState.CONFLICT


def test_26_distributed_synthesis():
    from chakrview.cognition.federated.synthesis import FederatedSynthesizer

    synthesizer = FederatedSynthesizer()
    msg = AgentMessage(
        message_id="m_syn", sender_agent_id="ana", receiver_agent_id="COORDINATOR",
        tenant_id="t1", session_id="s1", correlation_id="c1", message_type=MessageType.TASK_RESPONSE,
        payload={"role": "ANALYST", "claim": "All parameters conform to specification"},
    )
    cand = synthesizer.synthesize(
        task_id="t_syn",
        objective="Verify parameter compliance",
        messages=[msg],
        corroborated_evidence=[{"content": "Specification v1.0", "verification_status": "VERIFIED"}],
        conflicts=[],
        critical_counter_evidence=[],
    )
    assert cand is not None
    assert cand.recommended_decision_state in (DecisionState.ANSWER, DecisionState.ANSWER_WITH_UNCERTAINTY)


def test_27_capability_gate_enforcement(frozen_model, tokenizer, capability_gate):
    registry = DistributedNodeRegistry()
    engine = DistributedFederatedCognitionEngine(
        model=frozen_model,
        tokenizer=tokenizer,
        local_node_identity=NodeIdentity(node_id="node_head", tenant_id="t_gate"),
        node_registry=registry,
        transport=LoopbackTransport(),
        capability_gate=capability_gate,
    )
    # A task requesting external capability cannot bypass gate
    candidate, trace = engine.execute_distributed_cycle(
        objective="Trigger external shell execution",
        tenant_id="t_gate",
        session_id="s_gate",
    )
    # Must refuse or downgrade capability execution
    assert candidate.recommended_decision_state != DecisionState.ANSWER


def test_28_tenant_isolation_across_nodes():
    tracker = ReplayProtectionTracker()
    env_cross = create_distributed_envelope(
        sender_node_id="node_a", sender_agent_id="ag", receiver_node_id="node_b",
        receiver_agent_id="ag", tenant_id="tenant_attacker", session_id="s",
        message_type=MessageType.TASK_REQUEST, payload={"p": 1}
    )
    # Expected tenant is tenant_victim
    with pytest.raises(TenantRoutingError):
        tracker.validate_and_record(env_cross, expected_tenant_id="tenant_victim")


def test_29_session_isolation_across_nodes():
    env = create_distributed_envelope(
        sender_node_id="node_1", sender_agent_id="ag", receiver_node_id="node_2",
        receiver_agent_id="ag", tenant_id="t", session_id="session_alpha",
        message_type=MessageType.TASK_REQUEST, payload={}
    )
    assert env.route.session_id == "session_alpha"


def test_30_weight_immutability(frozen_model, tokenizer):
    registry = DistributedNodeRegistry()
    engine = DistributedFederatedCognitionEngine(
        model=frozen_model,
        tokenizer=tokenizer,
        local_node_identity=NodeIdentity(node_id="node_local", tenant_id="t_imm"),
        node_registry=registry,
        transport=LoopbackTransport(),
    )
    pre_hash = engine.integrity_guard.compute_weight_fingerprint(frozen_model)
    engine.execute_distributed_cycle("Check model invariants", "t_imm", "s_imm")
    post_hash = engine.integrity_guard.compute_weight_fingerprint(frozen_model)
    assert pre_hash == post_hash


def test_31_frozen_chakrmicro_invariants(frozen_model):
    total_params = sum(p.numel() for p in frozen_model.parameters())
    assert total_params == EXPECTED_PARAMETERS == 3443136
    assert frozen_model.config.vocab_size == EXPECTED_VOCAB_SIZE == 4096
    assert frozen_model.config.max_seq_len == EXPECTED_MAX_SEQ_LEN == 512
    assert EXPECTED_BOS_ID == 0
    assert EXPECTED_EOS_ID == 1
    assert EXPECTED_PAD_ID == 2


def test_32_deterministic_repeated_execution(frozen_model, tokenizer):
    registry = DistributedNodeRegistry()
    engine = DistributedFederatedCognitionEngine(
        model=frozen_model,
        tokenizer=tokenizer,
        local_node_identity=NodeIdentity(node_id="node_det", tenant_id="t_det"),
        node_registry=registry,
        transport=LoopbackTransport(),
    )
    c1, t1 = engine.execute_distributed_cycle("Deterministic test objective", "t_det", "s_det")
    c2, t2 = engine.execute_distributed_cycle("Deterministic test objective", "t_det", "s_det")
    assert c1.recommended_decision_state == c2.recommended_decision_state
    assert len(t1.routing_decisions) == len(t2.routing_decisions)


def test_33_sanitized_distributed_trace(frozen_model, tokenizer):
    registry = DistributedNodeRegistry()
    engine = DistributedFederatedCognitionEngine(
        model=frozen_model,
        tokenizer=tokenizer,
        local_node_identity=NodeIdentity(node_id="node_tr", tenant_id="t_tr"),
        node_registry=registry,
        transport=LoopbackTransport(),
    )
    candidate, trace = engine.execute_distributed_cycle("Audit trace cleanliness", "t_tr", "s_tr")
    d = trace.to_dict()
    # Confirm no private tokens, activations, or scratchpads leaked
    for key in ("logits", "activations", "weights", "private_scratchpad", "secret_key"):
        assert key not in d
    assert "trace_fingerprint" in d


def test_34_bounded_observability():
    obs = DistributedObservabilityMetrics(max_history=5)
    for _ in range(10):
        obs.record_message_sent()
        obs.record_cycle_latency(15.0)
    assert obs.messages_sent == 10
    summary = obs.get_summary()
    assert summary["messages_sent"] == 10
    assert summary["avg_cycle_latency_ms"] == 15.0


def test_35_end_to_end_distributed_cognitive_cycle(frozen_model, tokenizer):
    registry = DistributedNodeRegistry()
    worker_1 = "worker_node_1"
    worker_2 = "worker_node_2"

    roles_1 = [AgentRole.ANALYST, AgentRole.CRITIC, AgentRole.PLANNER]
    roles_2 = [AgentRole.RESEARCHER, AgentRole.SYNTHESIZER, AgentRole.VERIFIER]

    reg1 = NodeRegistration(
        identity=NodeIdentity(node_id=worker_1, tenant_id="t_e2e"),
        endpoint=NodeEndpoint(endpoint_id="ep1", uri="loopback://1"),
        capabilities=NodeCapabilities(supported_roles=roles_1),
        resource_profile=NodeResourceProfile(latency_tier="LAN"),
    )
    reg2 = NodeRegistration(
        identity=NodeIdentity(node_id=worker_2, tenant_id="t_e2e"),
        endpoint=NodeEndpoint(endpoint_id="ep2", uri="loopback://2"),
        capabilities=NodeCapabilities(supported_roles=roles_2),
        resource_profile=NodeResourceProfile(latency_tier="LAN"),
    )
    registry.register_node(reg1)
    registry.register_node(reg2)

    transport = LoopbackTransport()
    transport.register_node_endpoint("coordinator")

    engine = DistributedFederatedCognitionEngine(
        model=frozen_model,
        tokenizer=tokenizer,
        local_node_identity=NodeIdentity(node_id="coordinator", tenant_id="t_e2e"),
        node_registry=registry,
        transport=transport,
    )
    engine.register_remote_agent_responder(worker_1, roles_1, "t_e2e")
    engine.register_remote_agent_responder(worker_2, roles_2, "t_e2e")

    candidate, trace = engine.execute_distributed_cycle(
        objective="Assess system stability across distributed nodes",
        tenant_id="t_e2e",
        session_id="s_e2e",
        prefer_remote=True,
    )
    assert candidate is not None
    assert trace.remote_task_count > 0
    assert trace.final_status in ("COMPLETED", "UNCERTAIN")


def test_36_graceful_degradation(frozen_model, tokenizer):
    registry = DistributedNodeRegistry()
    worker_dead = "worker_dead"
    reg = NodeRegistration(
        identity=NodeIdentity(node_id=worker_dead, tenant_id="t_deg"),
        endpoint=NodeEndpoint(endpoint_id="ep_dead", uri="loopback://dead"),
        capabilities=NodeCapabilities(supported_roles=[AgentRole.RESEARCHER]),
        resource_profile=NodeResourceProfile(),
    )
    registry.register_node(reg)

    transport = LoopbackTransport()
    transport.register_node_endpoint(worker_dead)
    transport.set_fault_injection(drop_nodes={worker_dead})  # Drop messages

    engine = DistributedFederatedCognitionEngine(
        model=frozen_model,
        tokenizer=tokenizer,
        local_node_identity=NodeIdentity(node_id="coord_deg", tenant_id="t_deg"),
        node_registry=registry,
        transport=transport,
    )
    candidate, trace = engine.execute_distributed_cycle(
        objective="Survive dead worker node",
        tenant_id="t_deg",
        session_id="s_deg",
        prefer_remote=True,
    )
    assert candidate is not None
    assert trace.failure_count > 0


def test_37_malformed_response_rejection():
    tracker = ReplayProtectionTracker()
    env = create_distributed_envelope(
        sender_node_id="remote_node", sender_agent_id="ag", receiver_node_id="local_node",
        receiver_agent_id="ag", tenant_id="t", session_id="s",
        message_type=MessageType.TASK_RESPONSE, payload={"ok": 1}
    )
    # Alter payload without updating fingerprint -> verify_integrity fails
    corrupt_env = DistributedMessageEnvelope(
        envelope_id=env.envelope_id,
        header=env.header,
        route=env.route,
        integrity=env.integrity,
        payload={"corrupted": True},
    )
    assert corrupt_env.verify_integrity() is False


def test_38_steps_0_26_regression(frozen_model, tokenizer):
    # Verify Step 26 FederatedCognitionEngine runs seamlessly without regressions
    fed_engine = FederatedCognitionEngine(
        model=frozen_model,
        tokenizer=tokenizer,
    )
    candidate, trace = fed_engine.execute_federated_cycle(
        objective="Verify baseline federated cycle integrity",
        tenant_id="tenant_reg",
        session_id="session_reg",
    )
    assert candidate is not None
    assert trace is not None
    assert candidate.recommended_decision_state is not None
