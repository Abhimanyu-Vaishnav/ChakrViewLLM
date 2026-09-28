"""
Comprehensive Test Suite for ChakrView Step 26:
Multi-Agent Federated Cognition & Cooperative Intelligence Foundation.

Verifies all 38 required test mandates:
1.  Agent identity creation
2.  Agent contract validation
3.  Duplicate agent rejection
4.  Tenant isolation
5.  Session isolation
6.  Message creation
7.  Message validation
8.  Invalid sender rejection
9.  Invalid receiver rejection
10. Message integrity fingerprint
11. Registry lifecycle
12. Bounded agent count (hard ceiling <= 8)
13. Bounded execution rounds (hard ceiling <= 8)
14. Bounded message count (hard ceiling <= 128)
15. Task decomposition
16. Deterministic task assignment
17. Agent execution
18. Agent failure isolation
19. Retry/reassignment policy
20. Conflict detection
21. Minority evidence preservation
22. Evidence aggregation
23. Consensus/synthesis
24. Critical-thinking integration
25. Deliberation integration
26. Memory integration
27. Capability-gate enforcement
28. Unauthorized agent action rejection
29. Runtime weight immutability (SHA-256 fingerprint verification)
30. Frozen ChakrMicro invariants (3,443,136 parameters, 4,096 vocab, 512 context)
31. Deterministic repeated execution
32. Sanitized public trace (no private scratchpad leakage)
33. Corrupted message rejection
34. Cross-tenant message rejection
35. Delegation-depth limit (hard ceiling <= 4)
36. End-to-end federated cognitive cycle
37. Graceful degradation when an agent fails
38. Steps 0-25 regression
"""

from pathlib import Path
import pytest
import time
import torch
import uuid

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer.tokenizer import BPETokenizer
from chakrview.capability.gate import CapabilityGate, CapabilityAuthorizationError
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
from chakrview.memory.models import MemoryVerificationState
from chakrview.reasoning.engine import GovernedReasoningEngine
from chakrview.cognition.critical.engine import CriticalThinkingEngine
from chakrview.thinking.deliberation import DeliberationEngine
from chakrview.cognition.unified.models import (
    CognitiveTaskType,
    DecisionState,
)
from chakrview.cognition.federated.models import (
    AgentRole,
    AgentStatus,
    AgentCapability,
    AgentIdentity,
    AgentContract,
    MessageType,
    MessagePriority,
    AgentMessage,
    AgentTask,
    ConflictState,
    FederatedConflictRecord,
    FederatedSynthesisCandidate,
    SafePublicFederatedTrace,
)
from chakrview.cognition.federated.protocol import (
    FederatedProtocolValidator,
    MessageValidationError,
    MessageTamperingError,
    TenantRoutingError,
)
from chakrview.cognition.federated.registry import (
    AgentRegistry,
    DuplicateAgentIdentityError,
    RegistryCapacityExceededError,
    AgentNotFoundError,
)
from chakrview.cognition.federated.policy import (
    FederatedExecutionPolicy,
    HARD_CEILING_FEDERATED_AGENTS,
    HARD_CEILING_FEDERATED_ROUNDS,
    HARD_CEILING_FEDERATED_MESSAGES,
    HARD_CEILING_DELEGATION_DEPTH,
)
from chakrview.cognition.federated.decomposition import FederatedTaskDecomposer
from chakrview.cognition.federated.agents import (
    FederatedAgent,
    AnalystAgent,
    ResearcherAgent,
    CriticAgent,
    PlannerAgent,
    SynthesizerAgent,
    VerifierAgent,
    TenantIsolationError,
    AgentExecutionError,
)
from chakrview.cognition.federated.evidence import FederatedEvidenceAggregator, EvidenceCategory
from chakrview.cognition.federated.conflict import FederatedConflictResolver
from chakrview.cognition.federated.synthesis import FederatedSynthesizer
from chakrview.cognition.federated.engine import FederatedCognitionEngine


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
def federated_engine(frozen_model, tokenizer) -> FederatedCognitionEngine:
    """Fully provisioned FederatedCognitionEngine."""
    mem_engine = ContinualCognitionEngine()
    cap_gate = CapabilityGate(registry=CapabilityRegistry())
    return FederatedCognitionEngine(
        model=frozen_model,
        tokenizer=tokenizer,
        memory_engine=mem_engine,
        capability_gate=cap_gate,
    )


# ============================================================================
# 1. Identity, Contract & Registry Tests (Mandates 1-5, 11-12)
# ============================================================================

def test_01_agent_identity_creation():
    """Verify strongly typed AgentIdentity creation, fields, and defaults."""
    ident = AgentIdentity(
        agent_id="agent_analyst_01",
        role=AgentRole.ANALYST,
        tenant_id="tenant_alpha",
        session_id="sess_1",
        capabilities=[AgentCapability.REASONING],
    )
    assert ident.agent_id == "agent_analyst_01"
    assert ident.role == AgentRole.ANALYST
    assert ident.status == AgentStatus.READY
    assert AgentCapability.REASONING in ident.capabilities
    assert ident.version == "1.0.0"


def test_02_agent_contract_validation():
    """Verify AgentContract validation rules and ceiling bounds."""
    ident = AgentIdentity(
        agent_id="agent_res_01",
        role=AgentRole.RESEARCHER,
        tenant_id="tenant_alpha",
        session_id="sess_1",
    )
    # Valid contract
    contract = AgentContract(identity=ident)
    contract.validate()

    # Invalid: empty agent ID
    bad_ident = AgentIdentity(agent_id="", role=AgentRole.RESEARCHER, tenant_id="tenant_alpha", session_id="sess_1")
    bad_contract = AgentContract(identity=bad_ident)
    with pytest.raises(ValueError, match="Agent ID cannot be empty"):
        bad_contract.validate()

    # Invalid: token budget > 512
    overflow_contract = AgentContract(identity=ident, max_tokens_budget=1024)
    with pytest.raises(ValueError, match="exceeds ChakrMicro context ceiling"):
        overflow_contract.validate()

    # Invalid: gate requirement bypassed
    unauth_contract = AgentContract(identity=ident, requires_gate_for_capabilities=False)
    with pytest.raises(PermissionError, match="MUST require CapabilityGate"):
        unauth_contract.validate()


def test_03_duplicate_agent_rejection():
    """Verify AgentRegistry rejects duplicate agent IDs under same tenant."""
    reg = AgentRegistry(max_agents=5)
    ident = AgentIdentity(agent_id="dup_agent", role=AgentRole.ANALYST, tenant_id="tenant_alpha", session_id="sess_1")
    contract = AgentContract(identity=ident)

    reg.register(contract)
    with pytest.raises(DuplicateAgentIdentityError, match="already registered"):
        reg.register(contract)


def test_04_tenant_isolation_in_registry():
    """Verify registry strictly partitions agent discovery by tenant."""
    reg = AgentRegistry(max_agents=5)
    id_a = AgentIdentity(agent_id="agent_a", role=AgentRole.ANALYST, tenant_id="tenant_A", session_id="sess_1")
    id_b = AgentIdentity(agent_id="agent_b", role=AgentRole.ANALYST, tenant_id="tenant_B", session_id="sess_1")

    reg.register(AgentContract(identity=id_a))
    reg.register(AgentContract(identity=id_b))

    # Tenant A cannot see Tenant B's agent
    assert reg.get_agent("agent_b", tenant_id="tenant_A") is None
    assert len(reg.list_agents(tenant_id="tenant_A")) == 1
    assert reg.list_agents(tenant_id="tenant_A")[0].identity.agent_id == "agent_a"


def test_05_session_isolation_in_registry():
    """Verify agents with session constraints filter correctly."""
    reg = AgentRegistry(max_agents=5)
    id_s1 = AgentIdentity(agent_id="agent_s1", role=AgentRole.ANALYST, tenant_id="tenant_A", session_id="session_1")
    id_s2 = AgentIdentity(agent_id="agent_s2", role=AgentRole.ANALYST, tenant_id="tenant_A", session_id="session_2")

    reg.register(AgentContract(identity=id_s1))
    reg.register(AgentContract(identity=id_s2))

    s1_agents = reg.find_by_role(AgentRole.ANALYST, tenant_id="tenant_A", session_id="session_1")
    assert len(s1_agents) == 1
    assert s1_agents[0].identity.agent_id == "agent_s1"


def test_11_registry_lifecycle():
    """Verify complete registry lifecycle: register, status update, unregister, clear."""
    reg = AgentRegistry(max_agents=4)
    ident = AgentIdentity(agent_id="agent_life", role=AgentRole.PLANNER, tenant_id="tenant_alpha", session_id="sess_1")
    contract = AgentContract(identity=ident)

    reg.register(contract)
    assert reg.get_agent("agent_life", "tenant_alpha") is not None

    reg.update_status("agent_life", "tenant_alpha", AgentStatus.BUSY)
    assert reg.get_agent("agent_life", "tenant_alpha").identity.status == AgentStatus.BUSY

    unreg = reg.unregister("agent_life", "tenant_alpha")
    assert unreg is True
    assert reg.get_agent("agent_life", "tenant_alpha") is None


def test_12_bounded_agent_count():
    """Verify registry refuses registration beyond hard capacity ceiling."""
    reg = AgentRegistry(max_agents=3)
    for i in range(3):
        ident = AgentIdentity(agent_id=f"ag_{i}", role=AgentRole.ANALYST, tenant_id="t_bound", session_id="sess")
        reg.register(AgentContract(identity=ident))

    # 4th agent must fail
    overflow = AgentIdentity(agent_id="ag_overflow", role=AgentRole.ANALYST, tenant_id="t_bound", session_id="sess")
    with pytest.raises(RegistryCapacityExceededError):
        reg.register(AgentContract(identity=overflow))


# ============================================================================
# 2. Message Protocol, Validation & Cryptographic Fingerprinting (Mandates 6-10, 33-34)
# ============================================================================

def test_06_message_creation():
    """Verify AgentMessage initialization and automatic SHA-256 fingerprint."""
    msg = AgentMessage(
        message_id="msg_001",
        sender_agent_id="sender_01",
        receiver_agent_id="receiver_01",
        tenant_id="tenant_alpha",
        session_id="session_1",
        correlation_id="task_123",
        message_type=MessageType.TASK_RESPONSE,
        payload={"claim": "Quantum entropy evaluated"},
    )
    assert len(msg.integrity_hash) == 64
    assert msg.verify_integrity() is True


def test_07_message_validation():
    """Verify message protocol validation against tenant and registered senders."""
    msg = AgentMessage(
        message_id="msg_valid",
        sender_agent_id="agent_1",
        receiver_agent_id="COORDINATOR",
        tenant_id="tenant_alpha",
        session_id="session_1",
        correlation_id="task_01",
        message_type=MessageType.TASK_RESPONSE,
        payload={"result": "ok"},
    )
    # Valid call passes
    FederatedProtocolValidator.validate_message(
        message=msg,
        expected_tenant_id="tenant_alpha",
        expected_session_id="session_1",
        registered_sender_ids=["agent_1"],
    )


def test_08_invalid_sender_rejection():
    """Verify message from unregistered sender is rejected."""
    msg = AgentMessage(
        message_id="msg_rogue",
        sender_agent_id="unregistered_rogue_agent",
        receiver_agent_id="COORDINATOR",
        tenant_id="tenant_alpha",
        session_id="session_1",
        correlation_id="task_01",
        message_type=MessageType.TASK_RESPONSE,
    )
    with pytest.raises(MessageValidationError, match="Unknown sender"):
        FederatedProtocolValidator.validate_message(
            message=msg,
            expected_tenant_id="tenant_alpha",
            registered_sender_ids=["agent_valid"],
        )


def test_09_invalid_receiver_rejection():
    """Verify message targeted to unregistered recipient is rejected."""
    msg = AgentMessage(
        message_id="msg_bad_rcv",
        sender_agent_id="agent_valid",
        receiver_agent_id="unknown_target_agent",
        tenant_id="tenant_alpha",
        session_id="session_1",
        correlation_id="task_01",
        message_type=MessageType.TASK_REQUEST,
    )
    with pytest.raises(MessageValidationError, match="Unknown receiver"):
        FederatedProtocolValidator.validate_message(
            message=msg,
            expected_tenant_id="tenant_alpha",
            registered_receiver_ids=["agent_other"],
        )


def test_10_message_integrity_fingerprint():
    """Verify deterministic SHA-256 fingerprint generation."""
    msg_a = AgentMessage(
        message_id="msg_hash",
        sender_agent_id="s1",
        receiver_agent_id="r1",
        tenant_id="t1",
        session_id="sess",
        correlation_id="c1",
        message_type=MessageType.TASK_RESPONSE,
        payload={"val": 42},
    )
    msg_b = AgentMessage(
        message_id="msg_hash",
        sender_agent_id="s1",
        receiver_agent_id="r1",
        tenant_id="t1",
        session_id="sess",
        correlation_id="c1",
        message_type=MessageType.TASK_RESPONSE,
        payload={"val": 42},
    )
    assert msg_a.integrity_hash == msg_b.integrity_hash


def test_33_corrupted_message_rejection():
    """Verify that tampering with message payload invalidates hash and raises error."""
    msg = AgentMessage(
        message_id="msg_tampered",
        sender_agent_id="agent_1",
        receiver_agent_id="COORDINATOR",
        tenant_id="tenant_alpha",
        session_id="session_1",
        correlation_id="task_01",
        message_type=MessageType.TASK_RESPONSE,
        payload={"original": "data"},
    )
    # Illicit payload modification after hash generation
    msg.payload["malicious_injection"] = True
    assert msg.verify_integrity() is False

    with pytest.raises(MessageTamperingError, match="Message integrity violation"):
        FederatedProtocolValidator.validate_message(msg, expected_tenant_id="tenant_alpha")


def test_34_cross_tenant_message_rejection():
    """Verify message crossing tenant boundaries is blocked immediately."""
    msg = AgentMessage(
        message_id="msg_cross",
        sender_agent_id="agent_1",
        receiver_agent_id="COORDINATOR",
        tenant_id="tenant_ATTACKER",
        session_id="session_1",
        correlation_id="task_01",
        message_type=MessageType.TASK_RESPONSE,
    )
    with pytest.raises(TenantRoutingError, match="Tenant routing violation"):
        FederatedProtocolValidator.validate_message(msg, expected_tenant_id="tenant_VICTIM")


# ============================================================================
# 3. Policy & Execution Ceilings (Mandates 13-14, 35)
# ============================================================================

def test_13_bounded_execution_rounds():
    """Verify policy enforces hard ceiling of 8 rounds."""
    pol = FederatedExecutionPolicy(
        profile=ResourceProfile.HIGH_RESOURCE,
        max_agents=5,
        max_rounds=999,  # Attempting overflow
        max_messages_per_round=10,
        max_total_messages=50,
        max_delegation_depth=3,
        max_candidate_outputs=3,
        max_synthesis_attempts=2,
        timeout_ms=5000.0,
    )
    assert pol.max_rounds <= HARD_CEILING_FEDERATED_ROUNDS
    assert pol.max_rounds == 8


def test_14_bounded_message_count():
    """Verify policy enforces hard ceiling of 128 total messages."""
    pol = FederatedExecutionPolicy(
        profile=ResourceProfile.HIGH_RESOURCE,
        max_agents=5,
        max_rounds=4,
        max_messages_per_round=10,
        max_total_messages=500,  # Attempting overflow
        max_delegation_depth=3,
        max_candidate_outputs=3,
        max_synthesis_attempts=2,
        timeout_ms=5000.0,
    )
    assert pol.max_total_messages <= HARD_CEILING_FEDERATED_MESSAGES
    assert pol.max_total_messages == 128


def test_35_delegation_depth_limit():
    """Verify policy enforces hard ceiling of 4 delegation depth."""
    pol = FederatedExecutionPolicy(
        profile=ResourceProfile.HIGH_RESOURCE,
        max_agents=5,
        max_rounds=4,
        max_messages_per_round=10,
        max_total_messages=50,
        max_delegation_depth=10,  # Attempting overflow
        max_candidate_outputs=3,
        max_synthesis_attempts=2,
        timeout_ms=5000.0,
    )
    assert pol.max_delegation_depth <= HARD_CEILING_DELEGATION_DEPTH
    assert pol.max_delegation_depth == 4


# ============================================================================
# 4. Task Decomposition & Assignment (Mandates 15-16)
# ============================================================================

def test_15_task_decomposition():
    """Verify FederatedTaskDecomposer generates bounded, typed subtasks."""
    decomposer = FederatedTaskDecomposer()
    subtasks = decomposer.decompose(
        objective="What is the speed of light in vacuum?",
        task_type=CognitiveTaskType.FACTUAL,
        tenant_id="tenant_alpha",
        session_id="sess_1",
    )
    assert len(subtasks) >= 2
    roles = [t.assigned_role for t in subtasks]
    assert AgentRole.RESEARCHER in roles
    assert AgentRole.SYNTHESIZER in roles


def test_16_deterministic_task_assignment():
    """Verify identical input produces identical task decomposition structure."""
    decomposer = FederatedTaskDecomposer()
    st1 = decomposer.decompose("Analyze cache miss rates", CognitiveTaskType.ANALYTICAL, "t1", "s1")
    st2 = decomposer.decompose("Analyze cache miss rates", CognitiveTaskType.ANALYTICAL, "t1", "s1")

    assert len(st1) == len(st2)
    assert [t.assigned_role for t in st1] == [t.assigned_role for t in st2]
    assert [t.expected_output_type for t in st1] == [t.expected_output_type for t in st2]


# ============================================================================
# 5. Agent Execution, Sandboxing & Fault Isolation (Mandates 17-19, 28)
# ============================================================================

def test_17_agent_execution(frozen_model, tokenizer):
    """Verify single agent execution produces valid AgentMessage."""
    ident = AgentIdentity("test_ag", AgentRole.PLANNER, "t_alpha", "s_1")
    contract = AgentContract(ident)
    agent = PlannerAgent(ident, contract, frozen_model, tokenizer)

    task = AgentTask(
        task_id="task_pln_01",
        parent_task_id=None,
        assigned_role=AgentRole.PLANNER,
        objective="Plan deployment sequence",
        constraints={"tenant_id": "t_alpha", "session_id": "s_1"},
    )
    msg = agent.execute_task(task, context={})
    assert msg.sender_agent_id == "test_ag"
    assert msg.message_type == MessageType.TASK_RESPONSE
    assert "plan_steps" in msg.payload
    assert task.status == "COMPLETED"


def test_18_agent_failure_isolation(federated_engine):
    """Verify that a failing agent does not crash the federated cycle."""
    # Create custom agent set where one agent deliberately fails
    ident = AgentIdentity("broken_agent", AgentRole.RESEARCHER, "tenant_alpha", "sess_fail")
    contract = AgentContract(ident)
    broken_agent = ResearcherAgent(ident, contract, federated_engine.model, federated_engine.tokenizer)

    # Monkeypatch to force error
    def crash_process(*args, **kwargs):
        raise RuntimeError("Simulated agent runtime crash")
    broken_agent._process = crash_process

    custom_agents = {
        AgentRole.RESEARCHER: broken_agent,
        AgentRole.ANALYST: AnalystAgent(
            AgentIdentity("analyst_ok", AgentRole.ANALYST, "tenant_alpha", "sess_fail"),
            AgentContract(AgentIdentity("analyst_ok", AgentRole.ANALYST, "tenant_alpha", "sess_fail")),
            federated_engine.model,
            federated_engine.tokenizer,
        ),
        AgentRole.SYNTHESIZER: SynthesizerAgent(
            AgentIdentity("syn_ok", AgentRole.SYNTHESIZER, "tenant_alpha", "sess_fail"),
            AgentContract(AgentIdentity("syn_ok", AgentRole.SYNTHESIZER, "tenant_alpha", "sess_fail")),
            federated_engine.model,
            federated_engine.tokenizer,
        ),
    }

    # Cycle should complete with degradation rather than crashing
    cand, trace = federated_engine.execute_federated_cycle(
        objective="What is the boiling point of Water?",
        tenant_id="tenant_alpha",
        session_id="sess_fail",
        custom_agents=custom_agents,
    )
    assert trace.failures_count >= 1
    assert "COMPLETED" in trace.status


def test_19_retry_policy(federated_engine):
    """Verify engine records retry attempts upon agent failure."""
    trace_events = []
    # Test retry mechanism directly
    ident = AgentIdentity("flaky_ag", AgentRole.ANALYST, "t1", "s1")
    contract = AgentContract(ident)
    flaky = AnalystAgent(ident, contract, federated_engine.model, federated_engine.tokenizer)

    call_count = [0]
    def flaky_process(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            raise ValueError("Transient glitch")
        return {"role": "ANALYST", "claim": "Recovered on retry", "success": True}
    flaky._process = flaky_process

    task = AgentTask(
        task_id="t_flaky",
        parent_task_id=None,
        assigned_role=AgentRole.ANALYST,
        constraints={"tenant_id": "t1", "session_id": "s1"},
    )
    # First attempt fails
    with pytest.raises(AgentExecutionError):
        flaky.execute_task(task, {})

    # Retry succeeds
    msg = flaky.execute_task(task, {"retry": True})
    assert msg.payload["claim"] == "Recovered on retry"


def test_28_unauthorized_agent_action_rejection():
    """Verify an agent cannot directly bypass tenant boundary."""
    ident = AgentIdentity("agent_alpha", AgentRole.RESEARCHER, "tenant_ALPHA", "s1")
    agent = ResearcherAgent(ident, AgentContract(ident), None, None)

    foreign_task = AgentTask(
        task_id="t_foreign",
        parent_task_id=None,
        assigned_role=AgentRole.RESEARCHER,
        constraints={"tenant_id": "tenant_BETA"},  # Mismatch
    )
    with pytest.raises(TenantIsolationError):
        agent.execute_task(foreign_task, {})


# ============================================================================
# 6. Evidence Aggregation, Conflicts & Synthesis (Mandates 20-23)
# ============================================================================

def test_20_conflict_detection():
    """Verify FederatedConflictResolver detects opposing claims."""
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
    assert len(conflicts[0].claims) == 2


def test_21_minority_evidence_preservation():
    """Verify synthesizer preserves minority claim in synthesis candidate."""
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


def test_22_evidence_aggregation():
    """Verify evidence aggregator categorizes ground facts vs claims vs assumptions."""
    agg = FederatedEvidenceAggregator()
    msg_res = AgentMessage(
        message_id="m_res", sender_agent_id="researcher", receiver_agent_id="COORDINATOR",
        tenant_id="t1", session_id="s1", correlation_id="c1", message_type=MessageType.TASK_RESPONSE,
        payload={
            "role": "RESEARCHER",
            "retrieved_memories": [{"content": "Verified Fact 123", "verification_status": "VERIFIED", "confidence": 0.95}],
        },
    )
    msg_ana = AgentMessage(
        message_id="m_ana", sender_agent_id="analyst", receiver_agent_id="COORDINATOR",
        tenant_id="t1", session_id="s1", correlation_id="c1", message_type=MessageType.TASK_RESPONSE,
        payload={"role": "ANALYST", "claim": "Unproven assertion", "success": False},
    )
    agg.ingest_message(msg_res)
    agg.ingest_message(msg_ana)

    summary = agg.get_summary()
    assert summary["ground_evidence_count"] == 1
    assert summary["claims_count"] == 1
    assert summary["corroborated_evidence_count"] == 1


def test_23_consensus_synthesis():
    """Verify synthesizer produces valid FederatedSynthesisCandidate."""
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
    assert cand.recommended_decision_state == DecisionState.ANSWER
    assert "conformed" in cand.primary_synthesis.lower() or "parameters" in cand.primary_synthesis.lower()


# ============================================================================
# 7. Subsystem Integration & Authority Boundaries (Mandates 24-27)
# ============================================================================

def test_24_critical_thinking_integration(federated_engine):
    """Verify Critic agent integrates with CriticalThinkingEngine."""
    critic_agent = federated_engine.provision_standard_agents("t_alpha", "s1")[AgentRole.CRITIC]
    task = AgentTask(
        task_id="t_crit", parent_task_id=None, assigned_role=AgentRole.CRITIC,
        objective="Is absolute zero reachable via finite cooling cycles?",
        constraints={"tenant_id": "t_alpha", "session_id": "s1"},
    )
    msg = critic_agent.execute_task(task, {})
    assert "assumptions" in msg.payload
    assert "hypotheses" in msg.payload


def test_25_deliberation_integration(frozen_model, tokenizer):
    """Verify deliberation engine can be wired as cooperative substrate."""
    delib = DeliberationEngine(model=frozen_model, tokenizer=tokenizer)
    engine = FederatedCognitionEngine(
        model=frozen_model,
        tokenizer=tokenizer,
        deliberation_engine=delib,
    )
    assert engine.deliberation_engine is not None


def test_26_memory_integration(federated_engine):
    """Verify Researcher agent queries memory and returns verified facts."""
    federated_engine.memory_engine.add_semantic_fact(
        tenant_id="tenant_alpha",
        subject="AvogadroNumber",
        predicate="approx_value",
        object_value="6.022e23",
        confidence=1.0,
        verification_status=MemoryVerificationState.VERIFIED,
    )
    researcher = federated_engine.provision_standard_agents("tenant_alpha", "s1")[AgentRole.RESEARCHER]
    task = AgentTask(
        task_id="t_mem", parent_task_id=None, assigned_role=AgentRole.RESEARCHER,
        objective="What is AvogadroNumber approx_value?",
        constraints={"tenant_id": "tenant_alpha", "session_id": "s1"},
    )
    msg = researcher.execute_task(task, {})
    assert msg.payload["candidate_count"] >= 1
    contents = [m["content"] for m in msg.payload["retrieved_memories"]]
    assert any("6.022e23" in c for c in contents)


def test_27_capability_gate_enforcement(frozen_model, tokenizer):
    """Verify capability request is routed strictly through CapabilityGate and fails closed on denial."""
    gate = CapabilityGate(registry=CapabilityRegistry())
    engine = FederatedCognitionEngine(
        model=frozen_model,
        tokenizer=tokenizer,
        capability_gate=gate,
    )
    cand, trace = engine.execute_federated_cycle(
        objective="Execute sensor hardware device_read capability",
        tenant_id="tenant_alpha",
    )
    # Denied by CapabilityGate due to unregistered capability
    assert cand.recommended_decision_state == DecisionState.SAFE_STOP
    assert "denied" in cand.primary_synthesis.lower() or "prevented" in cand.primary_synthesis.lower()


# ============================================================================
# 8. Model Invariants, Immutability & Determinism (Mandates 29-32)
# ============================================================================

def test_29_runtime_weight_immutability(federated_engine, frozen_model):
    """Verify model SHA-256 weight fingerprint is identical before and after cycle."""
    guard = CoreIntegrityGuard()
    initial_fp = guard.compute_weight_fingerprint(frozen_model)

    cand, trace = federated_engine.execute_federated_cycle(
        objective="Run full federated cognitive cycle",
        tenant_id="tenant_alpha",
    )
    post_fp = guard.compute_weight_fingerprint(frozen_model)
    assert initial_fp == post_fp
    assert trace.weights_modified is False


def test_30_frozen_chakrmicro_invariants(frozen_model):
    """Verify strict preservation of frozen ChakrMicro v0.1 architectural invariants."""
    guard = CoreIntegrityGuard()
    res = guard.verify_model(frozen_model)
    assert res.passed is True
    total_params = sum(p.numel() for p in frozen_model.parameters())
    assert total_params == EXPECTED_PARAMETERS
    assert frozen_model.config.vocab_size == EXPECTED_VOCAB_SIZE
    assert frozen_model.config.max_seq_len == EXPECTED_MAX_SEQ_LEN
    assert frozen_model.config.pad_token_id == EXPECTED_PAD_ID


def test_31_deterministic_repeated_execution(federated_engine):
    """Verify repeated executions with same prompt yield identical structure and decision."""
    cand1, trace1 = federated_engine.execute_federated_cycle(
        objective="Evaluate consistency of immutable configuration",
        tenant_id="tenant_alpha",
    )
    cand2, trace2 = federated_engine.execute_federated_cycle(
        objective="Evaluate consistency of immutable configuration",
        tenant_id="tenant_alpha",
    )
    assert cand1.recommended_decision_state == cand2.recommended_decision_state
    assert trace1.execution_rounds == trace2.execution_rounds
    assert trace1.participating_agents == trace2.participating_agents


def test_32_sanitized_public_trace(federated_engine):
    """Verify public trace redacts private scratchpads, raw logits, and thoughts."""
    cand, trace = federated_engine.execute_federated_cycle(
        objective="Perform analytical reasoning on database transactions",
        tenant_id="tenant_alpha",
    )
    trace_dict = trace.to_dict()
    # Confirm required telemetry is exposed
    assert "trace_id" in trace_dict
    assert "execution_rounds" in trace_dict
    assert "trace_fingerprint" in trace_dict
    # Confirm private chain-of-thought is NOT exposed
    assert "scratchpad" not in trace_dict
    assert "logits" not in trace_dict
    assert "thoughts" not in trace_dict


# ============================================================================
# 9. End-to-End, Degradation & Backward Compatibility Regression (Mandates 36-38)
# ============================================================================

def test_36_end_to_end_federated_cognitive_cycle(federated_engine):
    """Verify complete end-to-end multi-agent cooperative cognitive cycle."""
    federated_engine.memory_engine.add_semantic_fact(
        tenant_id="tenant_e2e",
        subject="SpeedOfSound",
        predicate="value_m_per_s",
        object_value="343",
        confidence=1.0,
        verification_status=MemoryVerificationState.VERIFIED,
    )
    cand, trace = federated_engine.execute_federated_cycle(
        objective="What is SpeedOfSound value_m_per_s?",
        tenant_id="tenant_e2e",
    )
    assert cand.recommended_decision_state in (DecisionState.ANSWER, DecisionState.ANSWER_WITH_UNCERTAINTY)
    assert trace.execution_rounds >= 1
    assert trace.total_messages_exchanged >= 2
    assert trace.status == "COMPLETED"


def test_37_graceful_degradation_when_agent_fails(federated_engine):
    """Verify that execution degrades gracefully and marks status when an agent is absent."""
    # Custom agents with only Synthesizer (missing Researcher & Analyst)
    custom_agents = {
        AgentRole.SYNTHESIZER: SynthesizerAgent(
            AgentIdentity("syn_solo", AgentRole.SYNTHESIZER, "t_degrade", "s_1"),
            AgentContract(AgentIdentity("syn_solo", AgentRole.SYNTHESIZER, "t_degrade", "s_1")),
            federated_engine.model,
            federated_engine.tokenizer,
        )
    }
    cand, trace = federated_engine.execute_federated_cycle(
        objective="Synthesize partial information under degraded federation",
        tenant_id="t_degrade",
        custom_agents=custom_agents,
    )
    assert trace.failures_count >= 1
    assert trace.status == "COMPLETED_WITH_DEGRADATION"


def test_38_steps_0_25_regression():
    """Verify that earlier Steps 0-25 components remain fully operational."""
    from chakrview.cognition.unified.models import UnifiedCognitiveState
    from chakrview.memory.record import MemoryRecord, MemoryType
    from chakrview.thinking.workspace import ThinkingWorkspace

    # Step 25 Unified State
    ustate = UnifiedCognitiveState(
        cycle_id="c_reg", tenant_id="t_reg", session_id="s_reg",
        user_prompt="Regression test", task_type=CognitiveTaskType.FACTUAL,
        active_objective="Regression test",
    )
    assert ustate.cycle_id == "c_reg"

    # Step 24 Memory Record
    rec = MemoryRecord("m_reg", MemoryType.SEMANTIC, "Test", "owner")
    assert rec.content == "Test"

    # Step 21 Workspace
    ws = ThinkingWorkspace(task_id="t_reg", owner_id="user_reg")
    assert ws.task_id == "t_reg"
    assert ws.owner_id == "user_reg"
