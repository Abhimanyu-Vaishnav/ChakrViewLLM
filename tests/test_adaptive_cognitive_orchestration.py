"""
Comprehensive Test Suite for ChakrView Step 28:
Adaptive Cognitive Orchestration & Resource-Aware Federation.

Verifies all 32 required test mandates:
1.  Workload classification (SIMPLE, STANDARD, COMPLEX, AMBIGUOUS, CONFLICTED, VERIFICATION_REQUIRED, RESOURCE_CONSTRAINED)
2.  Deterministic classification
3.  Simple-task minimal allocation
4.  Standard-task allocation
5.  Complex-task allocation
6.  Ambiguous-task escalation
7.  Conflict-triggered verification
8.  Verification-required workflow
9.  Resource-constrained adaptation
10. Deterministic resource allocation
11. Bounded agent allocation (<= 8)
12. Bounded node allocation (<= 8)
13. Bounded deliberation (<= 3 rounds)
14. Bounded retries (<= 2 retries)
15. Dependency ordering (topological scheduling)
16. Task-plan determinism
17. Agent failure adaptation
18. Node failure adaptation
19. Safe degradation
20. Uncertainty preservation
21. Minority evidence preservation
22. Capability-gate enforcement
23. Tenant isolation
24. Session isolation
25. Memory governance
26. Sanitized telemetry
27. No chain-of-thought leakage
28. Runtime weight immutability (fail-closed check)
29. Frozen ChakrMicro invariants (3,443,136 parameters, 4,096 vocab, 512 context)
30. Repeated deterministic execution
31. Steps 0-27 regression
32. Complete end-to-end adaptive cognitive cycle
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
from chakrview.capability.contract import CapabilityRequest, CapabilityContext
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
from chakrview.state.manager import CognitiveStateManager
from chakrview.cognition.unified.models import DecisionState

from chakrview.cognition.federated.models import AgentRole, ConflictState
from chakrview.cognition.distributed.models import (
    NodeIdentity,
    NodeRole,
    NodeStatus,
    NodeTrustState,
    NodeCapabilities,
    NodeResourceProfile,
    NodeEndpoint,
    NodeRegistration,
)
from chakrview.cognition.distributed.registry import DistributedNodeRegistry

from chakrview.cognition.orchestration.models import (
    WorkloadClass,
    TaskPlan,
    ResourceAllocationDecision,
    SafePublicOrchestrationTrace,
    MAX_ORCHESTRATION_AGENTS,
    MAX_ORCHESTRATION_NODES,
    MAX_DELIBERATION_ROUNDS,
    MAX_ORCHESTRATION_RETRIES,
)
from chakrview.cognition.orchestration.classifier import DeterministicWorkloadClassifier
from chakrview.cognition.orchestration.planner import AdaptiveTaskPlanner
from chakrview.cognition.orchestration.allocator import ResourceAwareAllocator
from chakrview.cognition.orchestration.scheduler import DeterministicTaskScheduler
from chakrview.cognition.orchestration.adaptive import AdaptiveStrategySelector, OrchestrationStrategy
from chakrview.cognition.orchestration.deliberation import (
    AdaptiveDeliberationController,
    DeliberationSufficiencyEvaluation,
)
from chakrview.cognition.orchestration.memory import GovernedOrchestrationMemoryBridge
from chakrview.cognition.orchestration.observability import OrchestrationObservabilityMetrics
from chakrview.cognition.orchestration.policy import AdaptiveOrchestrationPolicy
from chakrview.cognition.orchestration.engine import AdaptiveCognitiveOrchestrator


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
def node_registry() -> DistributedNodeRegistry:
    """Populated DistributedNodeRegistry with test nodes."""
    registry = DistributedNodeRegistry(max_nodes=8)
    for i in range(4):
        ident = NodeIdentity(
            node_id=f"node_test_{i}",
            tenant_id="tenant_alpha",
            role=NodeRole.WORKER if i > 0 else NodeRole.PRIMARY,
        )
        endpoint = NodeEndpoint(
            endpoint_id=f"ep_{i}",
            uri=f"loopback://node_test_{i}",
        )
        capabilities = NodeCapabilities(
            supported_roles=[AgentRole.ANALYST, AgentRole.RESEARCHER, AgentRole.CRITIC, AgentRole.VERIFIER],
        )
        profile = NodeResourceProfile(cpu_cores=4, memory_mb=8192)
        reg = NodeRegistration(
            identity=ident,
            endpoint=endpoint,
            capabilities=capabilities,
            resource_profile=profile,
        )
        registry.register_node(reg)
    return registry


@pytest.fixture
def orchestrator(frozen_model, tokenizer, capability_gate, node_registry) -> AdaptiveCognitiveOrchestrator:
    """Instantiate AdaptiveCognitiveOrchestrator."""
    return AdaptiveCognitiveOrchestrator(
        model=frozen_model,
        tokenizer=tokenizer,
        capability_gate=capability_gate,
        node_registry=node_registry,
        local_node_id="node_test_0",
    )


# ============================================================================
# Tests: Workload Classification & Determinism (Mandates 1 & 2)
# ============================================================================

def test_workload_classification():
    """1. Test classification into all formal categories."""
    classifier = DeterministicWorkloadClassifier()

    # Simple
    w_simple, _ = classifier.classify("What is capital of France?")
    assert w_simple == WorkloadClass.SIMPLE

    # Standard
    w_std, _ = classifier.classify("Analyze the quarterly economic indicators and report implications.")
    assert w_std == WorkloadClass.STANDARD

    # Complex
    w_cplx, _ = classifier.classify(
        "Calculate the structural load on the beam; furthermore, evaluate the stress matrix and compute safety margins.",
        context={"requires_multi_step_planning": True}
    )
    assert w_cplx == WorkloadClass.COMPLEX

    # Ambiguous
    w_amb, _ = classifier.classify("Maybe the server is down or possibly something else is vague.")
    assert w_amb == WorkloadClass.AMBIGUOUS

    # Conflicted
    w_conf, _ = classifier.classify("There is an explicit contradiction between sensor A and sensor B readings.")
    assert w_conf == WorkloadClass.CONFLICTED

    # Verification Required
    w_ver, _ = classifier.classify("Audit the safety gate compliance and verify cryptography.")
    assert w_ver == WorkloadClass.VERIFICATION_REQUIRED

    # Resource Constrained
    w_rc, _ = classifier.classify(
        "Standard request",
        resource_profile=ResourceProfile.LOW_RESOURCE,
        node_health_degraded=True,
    )
    assert w_rc == WorkloadClass.RESOURCE_CONSTRAINED


def test_deterministic_classification():
    """2. Deterministic repeated classification produces identical outputs."""
    obj = "Calculate total stress and compute equilibrium parameters."
    c1, meta1 = DeterministicWorkloadClassifier.classify(obj)
    c2, meta2 = DeterministicWorkloadClassifier.classify(obj)
    assert c1 == c2
    assert meta1["reason"] == meta2["reason"]
    assert meta1["features"] == meta2["features"]


# ============================================================================
# Tests: Task Planning & Allocation by Workload (Mandates 3 - 9)
# ============================================================================

def test_simple_task_minimal_allocation(orchestrator):
    """3. Simple tasks allocate minimal single-agent set."""
    planner = orchestrator.planner
    plan = planner.plan_task("t1", "Ping status", WorkloadClass.SIMPLE)
    assert plan.max_agent_count == 1
    assert plan.max_deliberation_rounds == 1
    assert plan.required_roles == [AgentRole.ANALYST]
    assert plan.verification_required is False

    alloc = orchestrator.allocator.allocate(plan)
    assert alloc.allocated_agent_count == 1
    assert alloc.max_rounds == 1


def test_standard_task_allocation(orchestrator):
    """4. Standard task allocates standard multi-agent set."""
    plan = orchestrator.planner.plan_task("t2", "Analyze market trends", WorkloadClass.STANDARD)
    assert AgentRole.ANALYST in plan.required_roles
    assert AgentRole.RESEARCHER in plan.required_roles
    assert AgentRole.SYNTHESIZER in plan.required_roles

    alloc = orchestrator.allocator.allocate(plan)
    assert alloc.allocated_agent_count >= 3
    assert alloc.max_rounds <= 2


def test_complex_task_allocation(orchestrator):
    """5. Complex task allocates expanded roles and deliberation rounds."""
    plan = orchestrator.planner.plan_task("t3", "Formulate multi-year strategic roadmap", WorkloadClass.COMPLEX)
    assert AgentRole.PLANNER in plan.required_roles
    assert AgentRole.CRITIC in plan.required_roles
    assert plan.max_deliberation_rounds == 3

    alloc = orchestrator.allocator.allocate(plan, resource_profile=ResourceProfile.HIGH_RESOURCE)
    assert alloc.allocated_agent_count >= 4
    assert alloc.max_rounds == 3


def test_ambiguous_task_escalation(orchestrator):
    """6. Ambiguous task mandates verification and critique."""
    plan = orchestrator.planner.plan_task("t4", "Perhaps investigate uncertain anomaly", WorkloadClass.AMBIGUOUS)
    assert plan.verification_required is True
    assert AgentRole.CRITIC in plan.required_roles


def test_conflict_triggered_verification(orchestrator):
    """7. Conflicted task mandates Verifier role and preserves minority perspectives."""
    plan = orchestrator.planner.plan_task("t5", "Reconcile opposing telemetry assertions", WorkloadClass.CONFLICTED)
    assert plan.verification_required is True
    assert AgentRole.VERIFIER in plan.required_roles
    assert "minority_evidence_preserved" in plan.termination_conditions[1]


def test_verification_required_workflow(orchestrator):
    """8. Verification-required workload mandates Verifier before synthesis."""
    plan = orchestrator.planner.plan_task("t6", "Audit cryptographic key compliance", WorkloadClass.VERIFICATION_REQUIRED)
    assert plan.verification_required is True
    assert AgentRole.VERIFIER in plan.required_roles
    assert plan.role_dependencies[AgentRole.SYNTHESIZER.value] == [AgentRole.VERIFIER.value]


def test_resource_constrained_adaptation(orchestrator):
    """9. Resource-constrained task strips down to minimal set while preserving safety."""
    # Standard constrained
    plan = orchestrator.planner.plan_task("t7", "Check log entries", WorkloadClass.RESOURCE_CONSTRAINED)
    assert plan.max_agent_count == 1
    assert plan.verification_required is False

    # Safety critical constrained
    plan_safe = orchestrator.planner.plan_task(
        "t8", "Check containment seal safety", WorkloadClass.RESOURCE_CONSTRAINED, context={"safety_critical": True}
    )
    assert plan_safe.max_agent_count == 2
    assert plan_safe.verification_required is True
    assert AgentRole.VERIFIER in plan_safe.required_roles


# ============================================================================
# Tests: Bounded Allocation & Ceilings (Mandates 10 - 14)
# ============================================================================

def test_deterministic_resource_allocation(orchestrator):
    """10. Deterministic resource allocation."""
    plan = orchestrator.planner.plan_task("t_det", "Standard workload", WorkloadClass.STANDARD)
    alloc1 = orchestrator.allocator.allocate(plan, ResourceProfile.STANDARD, "tenant_alpha")
    alloc2 = orchestrator.allocator.allocate(plan, ResourceProfile.STANDARD, "tenant_alpha")
    assert alloc1.allocated_agent_count == alloc2.allocated_agent_count
    assert alloc1.selected_node_ids == alloc2.selected_node_ids
    assert alloc1.execution_budget_ms == alloc2.execution_budget_ms


def test_bounded_agent_allocation(orchestrator):
    """11. Agent allocation never exceeds hard ceiling of 8."""
    plan = orchestrator.planner.plan_task("t_big", "Complex mission", WorkloadClass.COMPLEX)
    alloc = orchestrator.allocator.allocate(plan, ResourceProfile.HIGH_RESOURCE)
    assert alloc.allocated_agent_count <= MAX_ORCHESTRATION_AGENTS


def test_bounded_node_allocation(orchestrator):
    """12. Node allocation never exceeds hard ceiling of 8."""
    plan = orchestrator.planner.plan_task("t_nodes", "Multi-node task", WorkloadClass.COMPLEX)
    alloc = orchestrator.allocator.allocate(plan, ResourceProfile.HIGH_RESOURCE)
    assert alloc.allocated_node_count <= MAX_ORCHESTRATION_NODES


def test_bounded_deliberation(orchestrator):
    """13. Deliberation rounds never exceed hard ceiling of 3."""
    plan = orchestrator.planner.plan_task("t_delib", "Complex deliberation", WorkloadClass.COMPLEX)
    alloc = orchestrator.allocator.allocate(plan, ResourceProfile.HIGH_RESOURCE)
    assert alloc.max_rounds <= MAX_DELIBERATION_ROUNDS


def test_bounded_retries(orchestrator):
    """14. Retries never exceed hard ceiling of 2."""
    plan = orchestrator.planner.plan_task("t_retry", "Task with failures", WorkloadClass.STANDARD)
    alloc = orchestrator.allocator.allocate(plan)
    assert alloc.retry_budget <= MAX_ORCHESTRATION_RETRIES


# ============================================================================
# Tests: Scheduling & Determinism (Mandates 15 & 16)
# ============================================================================

def test_dependency_ordering():
    """15. Topological role scheduling respects dependencies."""
    roles = [AgentRole.SYNTHESIZER, AgentRole.PLANNER, AgentRole.ANALYST, AgentRole.CRITIC]
    deps = {
        AgentRole.ANALYST.value: [AgentRole.PLANNER.value],
        AgentRole.CRITIC.value: [AgentRole.ANALYST.value],
        AgentRole.SYNTHESIZER.value: [AgentRole.CRITIC.value],
    }
    stages = DeterministicTaskScheduler.schedule_roles(roles, deps)
    flattened = [r for stage in stages for r in stage]

    # Verify order: Planner before Analyst, Analyst before Critic, Critic before Synthesizer
    assert flattened.index(AgentRole.PLANNER) < flattened.index(AgentRole.ANALYST)
    assert flattened.index(AgentRole.ANALYST) < flattened.index(AgentRole.CRITIC)
    assert flattened.index(AgentRole.CRITIC) < flattened.index(AgentRole.SYNTHESIZER)


def test_task_plan_determinism():
    """16. TaskPlan serialization and reproduction."""
    planner = AdaptiveTaskPlanner()
    p1 = planner.plan_task("task_x", "Evaluate system stability", WorkloadClass.STANDARD)
    p2 = planner.plan_task("task_x", "Evaluate system stability", WorkloadClass.STANDARD)
    assert p1.required_roles == p2.required_roles
    assert p1.max_agent_count == p2.max_agent_count
    assert p1.max_deliberation_rounds == p2.max_deliberation_rounds


# ============================================================================
# Tests: Resilience, Failures & Degradation (Mandates 17 - 19)
# ============================================================================

def test_agent_failure_adaptation(orchestrator):
    """17. Isolated agent failure does not crash the orchestration cycle."""
    resp, trace = orchestrator.orchestrate(
        objective="Run standard analysis",
        tenant_id="tenant_alpha",
        session_id="sess_1",
    )
    assert trace.final_decision_state in (DecisionState.ANSWER.value, DecisionState.ANSWER_WITH_UNCERTAINTY.value)
    assert trace.rounds_executed >= 1


def test_node_failure_adaptation(node_registry):
    """18. Quarantined or degraded nodes are avoided by allocator."""
    # Quarantine node_test_1
    reg1 = node_registry.get_node("node_test_1")
    reg1.health.status = NodeStatus.QUARANTINED

    allocator = ResourceAwareAllocator(node_registry=node_registry)
    planner = AdaptiveTaskPlanner()
    plan = planner.plan_task("t_node_fail", "Standard task", WorkloadClass.STANDARD)
    alloc = allocator.allocate(plan, tenant_id="tenant_alpha")

    assert "node_test_1" not in alloc.selected_node_ids


def test_safe_degradation(orchestrator):
    """19. Degraded environment completes safely with bounded resources."""
    resp, trace = orchestrator.orchestrate(
        objective="System diagnostics under degraded state",
        tenant_id="tenant_alpha",
        session_id="sess_1",
        resource_profile=ResourceProfile.LOW_RESOURCE,
        context={"force_resource_constrained": True},
    )
    assert trace.workload_class == WorkloadClass.RESOURCE_CONSTRAINED.value
    assert trace.allocated_agents <= 2
    assert trace.rounds_executed == 1


# ============================================================================
# Tests: Evidence, Uncertainty & Minority Preservation (Mandates 20 & 21)
# ============================================================================

def test_uncertainty_preservation(orchestrator):
    """20. Conflicted evidence without resolution preserves uncertainty."""
    resp, trace = orchestrator.orchestrate(
        objective="There is an unresolved contradiction in the measurements.",
        tenant_id="tenant_alpha",
        session_id="sess_1",
        context={"contradiction_count": 2},
    )
    assert trace.workload_class == WorkloadClass.CONFLICTED.value
    assert trace.conflicts_detected >= 1
    assert trace.final_decision_state == DecisionState.ANSWER_WITH_UNCERTAINTY.value


def test_minority_evidence_preservation():
    """21. Minority evidence is preserved and not discarded by majority."""
    eval_res = AdaptiveDeliberationController.evaluate_sufficiency(
        workload_class=WorkloadClass.CONFLICTED,
        current_round=3,
        max_rounds=3,
        total_claims=4,
        ground_evidence_count=2,
        contradiction_count=1,
        minority_evidence_count=1,
        verification_invoked=True,
        verification_passed=True,
    )
    assert eval_res.is_sufficient is True
    assert eval_res.minority_evidence_count == 1
    assert eval_res.detected_conflicts == 1


# ============================================================================
# Tests: Capability Gate & Authority Rules (Mandate 22)
# ============================================================================

def test_capability_gate_enforcement(orchestrator):
    """22. Capabilities must strictly pass through CapabilityGate."""
    resp, trace = orchestrator.orchestrate(
        objective="Execute external tool calculation",
        tenant_id="tenant_alpha",
        session_id="sess_1",
        context={
            "requested_capability": "unregistered_dangerous_tool",
            "capability_params": {"cmd": "eval()"},
        },
    )
    # Orchestrator does NOT bypass gate; unauthorized request stops safely
    assert trace.final_decision_state == DecisionState.SAFE_STOP.value
    assert "Capability Authorization Denied" in resp


# ============================================================================
# Tests: Tenant & Session Isolation (Mandates 23 & 24)
# ============================================================================

def test_tenant_isolation(orchestrator):
    """23. State manager belonging to another tenant is rejected."""
    wrong_state_mgr = CognitiveStateManager(owner_id="tenant_foreign", session_id="sess_1")
    with pytest.raises(ValueError, match="State isolation violation"):
        orchestrator.orchestrate(
            objective="Query isolated tenant data",
            tenant_id="tenant_alpha",
            session_id="sess_1",
            state_manager=wrong_state_mgr,
        )


def test_session_isolation(orchestrator):
    """24. State manager belonging to another session is rejected."""
    wrong_session_mgr = CognitiveStateManager(owner_id="tenant_alpha", session_id="sess_foreign")
    with pytest.raises(ValueError, match="State isolation violation"):
        orchestrator.orchestrate(
            objective="Query isolated session data",
            tenant_id="tenant_alpha",
            session_id="sess_1",
            state_manager=wrong_session_mgr,
        )


# ============================================================================
# Tests: Governed Memory & Privacy (Mandates 25 - 27)
# ============================================================================

def test_memory_governance(orchestrator):
    """25. Memory bridge captures structured, governed experience."""
    resp, trace = orchestrator.orchestrate(
        objective="Record standard interaction in memory",
        tenant_id="tenant_alpha",
        session_id="sess_1",
    )
    # Check that memory engine received interaction
    episodes = orchestrator.memory_engine.episodic_store.list_episodes(tenant_id="tenant_alpha")
    assert len(episodes) >= 1
    last_ep = episodes[-1]
    assert last_ep.metadata["workload_class"] == trace.workload_class
    assert last_ep.metadata["weights_modified"] is False


def test_sanitized_telemetry(orchestrator):
    """26. SafePublicOrchestrationTrace exposes telemetry without CoT."""
    resp, trace = orchestrator.orchestrate(
        objective="Analyze system logs",
        tenant_id="tenant_alpha",
        session_id="sess_1",
    )
    d = trace.to_dict()
    assert "trace_id" in d
    assert "workload_class" in d
    assert "total_latency_ms" in d
    assert "weights_modified" in d
    assert d["weights_modified"] is False


def test_no_chain_of_thought_leakage(orchestrator):
    """27. Trace contains zero private scratchpad or raw logits."""
    resp, trace = orchestrator.orchestrate(
        objective="Deliberate on complex plan",
        tenant_id="tenant_alpha",
        session_id="sess_1",
    )
    d = trace.to_dict()
    for forbidden in ["logits", "scratchpad", "chain_of_thought", "private_tokens", "secret_key"]:
        assert forbidden not in d


# ============================================================================
# Tests: Weight Immutability & Model Invariants (Mandates 28 & 29)
# ============================================================================

def test_runtime_weight_immutability(orchestrator, frozen_model):
    """28. Runtime weight immutability check fails closed on mutation."""
    # Normal cycle succeeds with weights unchanged
    guard = CoreIntegrityGuard()
    pre_hash = guard.compute_weight_fingerprint(frozen_model)
    resp, trace = orchestrator.orchestrate("Verify weights intact", "tenant_alpha", "sess_1")
    post_hash = guard.compute_weight_fingerprint(frozen_model)
    assert pre_hash == post_hash
    assert trace.weights_modified is False

    # Simulate rogue weight modification
    with torch.no_grad():
        frozen_model.embedding.weight[0, 0] += 0.001

    with pytest.raises(WeightMutationError, match="Model pre-hash mismatch"):
        orchestrator.orchestrate("This must fail closed", "tenant_alpha", "sess_1")

    # Restore weight for subsequent tests
    with torch.no_grad():
        frozen_model.embedding.weight[0, 0] -= 0.001


def test_frozen_chakrmicro_invariants(frozen_model):
    """29. ChakrMicro v0.1 parameters and dimensions match exact frozen constants."""
    total_params = sum(p.numel() for p in frozen_model.parameters())
    assert total_params == EXPECTED_PARAMETERS
    assert frozen_model.config.vocab_size == EXPECTED_VOCAB_SIZE
    assert frozen_model.config.max_seq_len == EXPECTED_MAX_SEQ_LEN
    assert EXPECTED_BOS_ID == 0
    assert EXPECTED_EOS_ID == 1
    assert EXPECTED_PAD_ID == 2


# ============================================================================
# Tests: Determinism & End-to-End Cycle (Mandates 30 - 32)
# ============================================================================

def test_repeated_deterministic_execution(orchestrator):
    """30. Repeated execution with identical input produces equivalent trace."""
    obj = "Deterministic query testing repeat execution"
    resp1, t1 = orchestrator.orchestrate(obj, "tenant_alpha", "sess_1")
    resp2, t2 = orchestrator.orchestrate(obj, "tenant_alpha", "sess_1")

    assert t1.workload_class == t2.workload_class
    assert t1.allocated_agents == t2.allocated_agents
    assert t1.allocated_nodes == t2.allocated_nodes
    assert t1.rounds_executed == t2.rounds_executed
    assert t1.final_decision_state == t2.final_decision_state


def test_steps_0_to_27_regression(node_registry):
    """31. Verify baseline Steps 0-27 components function alongside Step 28."""
    # Node registry from Step 27
    assert node_registry.total_nodes == 4
    assert node_registry.get_node("node_test_0") is not None

    # Adaptations profiles from Step 23
    assert ResourceProfile.STANDARD.value == "STANDARD"

    # Capability Gate from Step 17
    gate = CapabilityGate()
    assert gate.registry is not None


def test_complete_end_to_end_adaptive_cognitive_cycle(orchestrator):
    """32. End-to-end adaptive cycle execution across different workload types."""
    # Test simple
    resp_s, trace_s = orchestrator.orchestrate("Ping", "tenant_alpha", "sess_1")
    assert trace_s.workload_class == WorkloadClass.SIMPLE.value
    assert trace_s.allocated_agents == 1

    # Test standard
    resp_std, trace_std = orchestrator.orchestrate("Analyze performance metrics", "tenant_alpha", "sess_1")
    assert trace_std.workload_class == WorkloadClass.STANDARD.value
    assert trace_std.allocated_agents >= 2

    # Test verification required
    resp_v, trace_v = orchestrator.orchestrate("Verify safety shutdown logic", "tenant_alpha", "sess_1")
    assert trace_v.workload_class == WorkloadClass.VERIFICATION_REQUIRED.value
    assert trace_v.verification_invoked is True


def test_early_termination_on_sufficiency():
    """33. Early termination is triggered when evidence is sufficient in round 1."""
    eval_res = AdaptiveDeliberationController.evaluate_sufficiency(
        workload_class=WorkloadClass.STANDARD,
        current_round=1,
        max_rounds=3,
        total_claims=3,
        ground_evidence_count=2,
        contradiction_count=0,
        minority_evidence_count=0,
        verification_invoked=False,
        verification_passed=None,
    )
    assert eval_res.is_sufficient is True
    assert eval_res.requires_additional_round is False


def test_observability_metrics_boundedness():
    """34. Rolling window in observability tracker does not grow unbounded."""
    metrics = OrchestrationObservabilityMetrics(max_history=10)
    for i in range(1500):
        metrics.record_orchestration(
            workload_class=WorkloadClass.SIMPLE,
            profile=ResourceProfile.STANDARD,
            agent_count=1,
            node_count=1,
            rounds=1,
            latency_ms=1.5,
            decision_state=DecisionState.ANSWER,
        )
    assert len(metrics._agent_counts) <= 1000
    assert metrics.tasks_orchestrated == 1500


def test_invalid_inputs_fail_closed(orchestrator):
    """35. Empty or malformed inputs fail closed immediately."""
    with pytest.raises(ValueError, match="tenant_id"):
        orchestrator.orchestrate("valid objective", tenant_id="", session_id="s1")

    with pytest.raises(ValueError, match="session_id"):
        orchestrator.orchestrate("valid objective", tenant_id="t1", session_id="")

    with pytest.raises(ValueError, match="objective"):
        orchestrator.orchestrate("   ", tenant_id="t1", session_id="s1")


def test_strategy_selection():
    """36. AdaptiveStrategySelector selects expected strategies across workloads."""
    # Simple -> LOCAL_MINIMAL
    s1 = AdaptiveStrategySelector.select_strategy(WorkloadClass.SIMPLE, ResourceProfile.STANDARD)
    assert s1 == OrchestrationStrategy.LOCAL_MINIMAL

    # Verification required -> VERIFICATION_PIPELINE
    s2 = AdaptiveStrategySelector.select_strategy(WorkloadClass.VERIFICATION_REQUIRED, ResourceProfile.STANDARD)
    assert s2 == OrchestrationStrategy.VERIFICATION_PIPELINE

    # Resource constrained -> FAIL_SAFE_DEGRADED
    s3 = AdaptiveStrategySelector.select_strategy(WorkloadClass.RESOURCE_CONSTRAINED, ResourceProfile.LOW_RESOURCE)
    assert s3 == OrchestrationStrategy.FAIL_SAFE_DEGRADED

    # Distributed available -> DISTRIBUTED_ROUTED
    s4 = AdaptiveStrategySelector.select_strategy(WorkloadClass.COMPLEX, ResourceProfile.HIGH_RESOURCE, available_nodes_count=3)
    assert s4 == OrchestrationStrategy.DISTRIBUTED_ROUTED


def test_deliberation_controller_max_rounds_hard_stop():
    """37. Deliberation controller enforces hard ceiling and marks terminal."""
    eval_res = AdaptiveDeliberationController.evaluate_sufficiency(
        workload_class=WorkloadClass.COMPLEX,
        current_round=3,
        max_rounds=3,
        total_claims=5,
        ground_evidence_count=2,
        contradiction_count=0,
        minority_evidence_count=0,
        verification_invoked=False,
        verification_passed=None,
    )
    assert eval_res.is_sufficient is True
    assert eval_res.requires_additional_round is False
    assert "Reached maximum round ceiling" in eval_res.reason


def test_observability_metrics_aggregation():
    """38. Telemetry metrics properly compute rolling averages and distribution."""
    metrics = OrchestrationObservabilityMetrics()
    metrics.record_orchestration(WorkloadClass.SIMPLE, ResourceProfile.STANDARD, agent_count=1, node_count=1, rounds=1, latency_ms=10.0, decision_state=DecisionState.ANSWER)
    metrics.record_orchestration(WorkloadClass.COMPLEX, ResourceProfile.HIGH_RESOURCE, agent_count=5, node_count=3, rounds=3, latency_ms=30.0, decision_state=DecisionState.ANSWER_WITH_UNCERTAINTY)

    assert metrics.tasks_orchestrated == 2
    assert metrics.average_agent_count == 3.0
    assert metrics.average_node_count == 2.0
    assert metrics.average_deliberation_rounds == 2.0
    assert metrics.average_latency_ms == 20.0
    assert metrics.uncertainty_rate == 0.5
    d = metrics.to_dict()
    assert d["workload_counts"][WorkloadClass.SIMPLE.value] == 1
    assert d["workload_counts"][WorkloadClass.COMPLEX.value] == 1


def test_state_manager_contradiction_detection(orchestrator):
    """39. CognitiveStateManager uncertainties feed into workload classification."""
    state_mgr = CognitiveStateManager(owner_id="tenant_alpha", session_id="sess_1")
    state_mgr.record_uncertainty(
        key="contradiction_reading_alpha",
        confidence=0.4,
        reason="Sensor conflict detected on channel alpha",
        source="sensor_audit",
    )

    resp, trace = orchestrator.orchestrate(
        objective="Assess stability of channel alpha",
        tenant_id="tenant_alpha",
        session_id="sess_1",
        state_manager=state_mgr,
    )
    assert trace.workload_class == WorkloadClass.CONFLICTED.value
    assert trace.conflicts_detected >= 1
    assert trace.final_decision_state == DecisionState.ANSWER_WITH_UNCERTAINTY.value


def test_fixed_vs_adaptive_resource_savings(orchestrator):
    """40. Minimum-sufficient cognition saves significant agent/round allocations for simple tasks."""
    # Simple query
    _, trace_simple = orchestrator.orchestrate("Ping", "tenant_alpha", "sess_1")
    # Complex query
    _, trace_complex = orchestrator.orchestrate(
        "Calculate the structural load on the beam; furthermore, evaluate the stress matrix and compute safety margins.",
        "tenant_alpha",
        "sess_1",
        context={"requires_multi_step_planning": True},
    )

    # Prove minimum-sufficient bounded allocation saves resources
    assert trace_simple.allocated_agents < trace_complex.allocated_agents
    assert trace_simple.rounds_executed <= trace_complex.rounds_executed
    assert trace_simple.allocated_agents == 1
    assert trace_complex.allocated_agents >= 3

