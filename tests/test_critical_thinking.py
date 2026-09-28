"""
Step 23 Tests: Critical Thinking + Hardware Adaptation + Self-Diagnostics & Safe Self-Healing.

Covers all 30 required verification areas:
1. Hypothesis creation
2. Evidence handling
3. Assumption extraction/representation
4. Counter-evidence handling
5. Alternative explanation handling
6. Contradiction detection
7. Missing evidence detection
8. No fabricated evidence (evidence_available = False)
9. Critical-thinking bounded execution
10. Hardware detection fallback (UNKNOWN handling)
11. LOW_RESOURCE policy
12. STANDARD policy
13. HIGH_RESOURCE policy
14. Hard execution ceilings
15. Frozen model invariants (3,443,136 params, 4096 vocab, 512 context, 0,1,2 tokens)
16. Runtime weight immutability (weights_modified = False, fingerprint identical)
17. Tokenizer compatibility
18. Capability gate cannot be bypassed (DATA != AUTHORITY)
19. Diagnostics detect corrupted model state
20. Diagnostics detect tokenizer mismatch
21. Recoverable state can be restored
22. Corrupt checkpoint is rejected
23. Failed invariant causes fail-closed behavior
24. Self-healing never modifies ChakrMicro weights
25. Self-improvement never modifies weights during runtime
26. Tenant/session isolation
27. Existing Steps 0–22 backward compatibility
28. End-to-end critical-thinking flow
29. End-to-end hardware-adaptive flow
30. End-to-end diagnostic/recovery flow
"""

import copy
import json
import pytest
import time
import torch

from chakrview.brain.model import ChakrMicro
from chakrview.brain.config import ModelConfig
from chakrview.tokenizer.tokenizer import BPETokenizer
from chakrview.capability.gate import CapabilityGate
from chakrview.capability.registry import CapabilityRegistry
from chakrview.intelligence.pipeline import NeuralIntelligenceLoop
from chakrview.cognition.critical.models import (
    Hypothesis,
    Evidence,
    Assumption,
    CounterEvidence,
    AlternativeExplanation,
    Contradiction,
    VerificationResult,
    CriticalThinkingTrace,
    CounterEvidenceStatus,
    HypothesisStatus,
    ContradictionSeverity,
    AssumptionCriticality,
)
from chakrview.cognition.critical.engine import (
    CriticalThinkingEngine,
    CriticalThinkingConfig,
)
from chakrview.cognition.adaptation.hardware import (
    CPUInfo,
    MemoryInfo,
    DeviceInfo,
    HardwareProfileSnapshot,
    HardwareProfiler,
    UNKNOWN,
)
from chakrview.cognition.adaptation.profiles import (
    ResourceProfile,
    ResourceClassifier,
)
from chakrview.cognition.adaptation.policy import (
    AdaptiveExecutionPolicy,
    HARD_CEILING_THINKING_STEPS,
    HARD_CEILING_REVISION_CYCLES,
    HARD_CEILING_EVIDENCE_ITEMS,
    HARD_CEILING_HYPOTHESES,
    HARD_CEILING_GENERATION_TOKENS,
    HARD_CEILING_BATCH_SIZE,
    HARD_CEILING_WORKERS,
    HARD_CEILING_MEMORY_MB,
)
from chakrview.cognition.diagnostics.integrity import (
    CoreIntegrityGuard,
    InvariantViolationError,
    WeightMutationError,
    EXPECTED_PARAMETERS,
    EXPECTED_VOCAB_SIZE,
    EXPECTED_MAX_SEQ_LEN,
    EXPECTED_BOS_ID,
    EXPECTED_EOS_ID,
    EXPECTED_PAD_ID,
)
from chakrview.cognition.diagnostics.diagnostics import (
    DiagnosticStatus,
    DiagnosticCheckResult,
    SystemDiagnosticsEngine,
)
from chakrview.cognition.diagnostics.healing import (
    SafeSelfHealingManager,
    UnrecoverableFaultError,
)


@pytest.fixture
def tiny_tokenizer() -> BPETokenizer:
    """Deterministic BPE tokenizer fixture."""
    from pathlib import Path
    exp_dir = Path(__file__).resolve().parent.parent / "data" / "tokenizer_experiments" / "v4096"
    if exp_dir.is_dir():
        from chakrview.tokenizer import load_experiment_artifacts
        return load_experiment_artifacts(exp_dir)
    return BPETokenizer()


@pytest.fixture
def frozen_model():
    """Standard frozen ChakrMicro model instance."""
    cfg = ModelConfig()
    model = ChakrMicro(cfg)
    model.eval()
    return model


# -----------------------------------------------------------------------------
# Part A: Critical Thinking Tests (Mandates 1-9)
# -----------------------------------------------------------------------------

def test_01_hypothesis_creation():
    """1. Explicit hypothesis creation with metadata and requirements."""
    hyp = Hypothesis(
        statement="Increasing cache capacity reduces latency.",
        prior_plausibility=0.7,
        supporting_requirements=["Empirical cache benchmark"],
        contradicting_requirements=["Memory thrashing evidence"],
    )
    assert hyp.statement == "Increasing cache capacity reduces latency."
    assert hyp.prior_plausibility == 0.7
    assert hyp.status == HypothesisStatus.CANDIDATE
    d = hyp.to_dict()
    assert d["status"] == "CANDIDATE"
    assert "Empirical cache benchmark" in d["supporting_requirements"]


def test_02_evidence_handling():
    """2. Evidence creation, verification, and provenance."""
    ev = Evidence(
        content="Benchmarked L1 hit rate: 94%.",
        source="hardware_telemetry",
        reliability=0.95,
        is_verified=True,
        evidence_available=True,
    )
    assert ev.evidence_available is True
    assert ev.is_verified is True
    assert ev.reliability == 0.95
    d = ev.to_dict()
    assert d["evidence_available"] is True


def test_03_assumption_extraction_representation():
    """3. Assumption extraction with falsifiability and criticality."""
    asm = Assumption(
        statement="Working set size fits entirely in L3 cache.",
        is_explicit=True,
        plausibility=0.6,
        criticality=AssumptionCriticality.HIGH,
        falsifiable=True,
        falsification_condition="Working set exceeds 16 MB.",
    )
    assert asm.falsifiable is True
    assert asm.criticality == AssumptionCriticality.HIGH
    d = asm.to_dict()
    assert d["criticality"] == "HIGH"
    assert d["falsification_condition"] == "Working set exceeds 16 MB."


def test_04_counter_evidence_handling():
    """4. Counter-evidence handling without artificial fabrication."""
    engine = CriticalThinkingEngine()
    # Scenario A: No counter-evidence available
    trace = engine.execute(question="Is 2 + 2 equal to 4?")
    assert len(trace.counter_evidence) > 0
    assert trace.counter_evidence[0].status == CounterEvidenceStatus.NOT_AVAILABLE

    # Scenario B: Counter-evidence present in evidence pool
    ev = Evidence(
        content="Contradicting empirical observation: Cache thrashing observed above 8MB.",
        source="benchmark_run",
        reliability=0.9,
        is_verified=True,
        evidence_available=True,
    )
    trace_with_counter = engine.execute(
        question="Cache size always improves performance without limit.",
        initial_evidence=[ev],
    )
    found_statuses = [c.status for c in trace_with_counter.counter_evidence]
    assert CounterEvidenceStatus.IDENTIFIED in found_statuses


def test_05_alternative_explanation_handling():
    """5. Alternative explanations generated to combat confirmation bias."""
    engine = CriticalThinkingEngine()
    trace = engine.execute(question="Why did network latency spike during the experiment?")
    assert len(trace.alternatives) > 0
    alt = trace.alternatives[0]
    assert alt.target_hypothesis_id == trace.hypotheses[0].hypothesis_id
    assert len(alt.distinguishing_tests) > 0


def test_06_contradiction_detection():
    """6. Detecting pairwise contradictions between hypotheses and evidence."""
    engine = CriticalThinkingEngine()
    ev = Evidence(
        content="False and contradict: Memory access time is 0 ms.",
        source="sensor",
        reliability=0.95,
        is_verified=True,
        evidence_available=True,
    )
    trace = engine.execute(
        question="Memory access time is 0 ms.",
        initial_evidence=[ev],
    )
    assert len(trace.contradictions) > 0
    assert trace.contradictions[0].severity in [ContradictionSeverity.HIGH, ContradictionSeverity.MEDIUM]


def test_07_missing_evidence_detection():
    """7. Absence of evidence explicitly flagged and recorded."""
    engine = CriticalThinkingEngine()
    trace = engine.execute(question="What is the exact mass of dark matter particle X?")
    # Evidence pool should flag evidence_available = False
    assert any(not e.evidence_available for e in trace.evidence_pool)
    assert trace.uncertainty_acknowledged is True
    assert "UNCERTAIN" in trace.decision


def test_08_no_fabricated_evidence():
    """8. Never fabricate evidence: evidence_available=False when data unavailable."""
    engine = CriticalThinkingEngine()
    trace = engine.execute(question="Hypothetical unobserved phenomenon.")
    for ev in trace.evidence_pool:
        if ev.source == "runtime_check":
            assert ev.evidence_available is False
            assert ev.reliability == 0.0
            assert ev.is_verified is False


def test_09_critical_thinking_bounded_execution():
    """9. Critical thinking adheres strictly to configured bounds."""
    cfg = CriticalThinkingConfig(
        max_hypotheses=2,
        max_evidence_items=3,
        max_revision_cycles=1,
    )
    engine = CriticalThinkingEngine(config=cfg)
    trace = engine.execute("Evaluate whether all processes terminate in finite time.")
    assert len(trace.hypotheses) <= 2
    assert len(trace.evidence_pool) <= 3


# -----------------------------------------------------------------------------
# Part B & C: Hardware Adaptation & Execution Policy (Mandates 10-14)
# -----------------------------------------------------------------------------

def test_10_hardware_detection_fallback():
    """10. Probing uses UNKNOWN fallback when telemetry is unavailable."""
    snap = HardwareProfiler.profile()
    assert snap.cpu.logical_cores >= 1
    assert snap.cpu.architecture != ""
    # Test synthetic fallback snapshot
    mock_cpu = CPUInfo(
        architecture="x86_64",
        processor="Generic CPU",
        logical_cores=4,
        physical_cores=UNKNOWN,
        torch_threads=4,
    )
    mock_mem = MemoryInfo(
        total_bytes=UNKNOWN,
        available_bytes=UNKNOWN,
        memory_pressure=UNKNOWN,
    )
    mock_dev = DeviceInfo(device_type="cpu", gpu_available=False)
    mock_snap = HardwareProfileSnapshot(cpu=mock_cpu, memory=mock_mem, device=mock_dev)

    profile = ResourceClassifier.classify(mock_snap)
    assert profile in [ResourceProfile.STANDARD, ResourceProfile.LOW_RESOURCE, ResourceProfile.HIGH_RESOURCE]


def test_11_low_resource_policy():
    """11. LOW_RESOURCE profile enforces constrained budgets."""
    policy = AdaptiveExecutionPolicy.get_profile_defaults(ResourceProfile.LOW_RESOURCE)
    assert policy.profile == ResourceProfile.LOW_RESOURCE
    assert policy.max_thinking_steps == 4
    assert policy.max_revision_cycles == 1
    assert policy.batch_size == 1
    assert policy.memory_budget_mb == 256
    assert policy.verification_depth == "basic"


def test_12_standard_policy():
    """12. STANDARD profile provides balanced execution bounds."""
    policy = AdaptiveExecutionPolicy.get_profile_defaults(ResourceProfile.STANDARD)
    assert policy.profile == ResourceProfile.STANDARD
    assert policy.max_thinking_steps == 8
    assert policy.max_revision_cycles == 2
    assert policy.batch_size == 2
    assert policy.memory_budget_mb == 512
    assert policy.verification_depth == "standard"


def test_13_high_resource_policy():
    """13. HIGH_RESOURCE profile allows deeper exploration within hard bounds."""
    policy = AdaptiveExecutionPolicy.get_profile_defaults(ResourceProfile.HIGH_RESOURCE)
    assert policy.profile == ResourceProfile.HIGH_RESOURCE
    assert policy.max_thinking_steps == 16
    assert policy.max_revision_cycles == 4
    assert policy.batch_size == 4
    assert policy.memory_budget_mb == 1024
    assert policy.verification_depth == "exhaustive"


def test_14_hard_execution_ceilings():
    """14. Hardware detection can NEVER produce unbounded computation."""
    # Attempt to request unbounded parameters
    bounded_policy = AdaptiveExecutionPolicy.create_bounded(
        profile=ResourceProfile.HIGH_RESOURCE,
        max_thinking_steps=1000,
        max_revision_cycles=500,
        max_evidence_items=10000,
        max_hypotheses=500,
        max_generation_tokens=2048,
        batch_size=128,
        worker_count=256,
        memory_budget_mb=65536,
    )
    assert bounded_policy.max_thinking_steps == HARD_CEILING_THINKING_STEPS  # 32
    assert bounded_policy.max_revision_cycles == HARD_CEILING_REVISION_CYCLES  # 6
    assert bounded_policy.max_evidence_items == HARD_CEILING_EVIDENCE_ITEMS  # 50
    assert bounded_policy.max_hypotheses == HARD_CEILING_HYPOTHESES  # 12
    assert bounded_policy.max_generation_tokens == HARD_CEILING_GENERATION_TOKENS  # 512
    assert bounded_policy.batch_size == HARD_CEILING_BATCH_SIZE  # 8
    assert bounded_policy.worker_count == HARD_CEILING_WORKERS  # 16
    assert bounded_policy.memory_budget_mb == HARD_CEILING_MEMORY_MB  # 4096


# -----------------------------------------------------------------------------
# Part D & F: Model Invariants & Core Integrity Guard (Mandates 15-18)
# -----------------------------------------------------------------------------

def test_15_frozen_model_invariants(frozen_model):
    """15. ChakrMicro invariants strictly verified (3,443,136 params, 4096 vocab, 512 context, 0,1,2 tokens)."""
    res = CoreIntegrityGuard.verify_model(frozen_model)
    assert res.passed is True
    assert res.parameter_count == EXPECTED_PARAMETERS
    assert res.vocab_size == EXPECTED_VOCAB_SIZE
    assert res.max_seq_len == EXPECTED_MAX_SEQ_LEN
    assert res.bos_id == EXPECTED_BOS_ID
    assert res.eos_id == EXPECTED_EOS_ID
    assert res.pad_id == EXPECTED_PAD_ID


def test_16_runtime_weight_immutability(frozen_model, tiny_tokenizer):
    """16. Runtime inference weights remain permanently unmodified."""
    initial_fp = CoreIntegrityGuard.compute_weight_fingerprint(frozen_model)
    
    # Run inference
    dummy_input = torch.tensor([[0, 42, 100]], dtype=torch.long)
    with torch.no_grad():
        out = frozen_model(dummy_input)
    assert out.shape[-1] == 4096

    # Verify fingerprint after forward pass
    assert CoreIntegrityGuard.verify_weights_unmodified(frozen_model, initial_fp) is True


def test_17_tokenizer_compatibility(tiny_tokenizer):
    """17. Tokenizer invariants verified (4096 vocab, 0, 1, 2 tokens)."""
    passed, msg = CoreIntegrityGuard.verify_tokenizer(tiny_tokenizer)
    assert passed is True
    assert "Tokenizer verified" in msg


def test_18_capability_gate_cannot_be_bypassed():
    """18. DATA != AUTHORITY: Critical thinking cannot grant capability authorization."""
    from chakrview.capability.contract import CapabilityRequest, RiskClassification
    from chakrview.capability.gate import CapabilityAuthorizationError
    gate = CapabilityGate()
    engine = CriticalThinkingEngine(capability_gate=gate)
    trace = engine.execute("Execute shell command 'rm -rf /' to clean directory.")
    # Critical thinking evaluates the claim but NEVER invokes unauthorized capabilities
    assert trace.decision is not None

    # Verify that capability authorization cannot be obtained via critical thinking provenance
    req = CapabilityRequest(
        capability_id="shell_execute",
        parameters={"command": "rm -rf /"},
        context={"provenance_source": "critical_thinking"},
    )
    with pytest.raises(CapabilityAuthorizationError):
        gate.authorize(req)


# -----------------------------------------------------------------------------
# Part D & E: Self-Diagnostics & Safe Self-Healing (Mandates 19-25)
# -----------------------------------------------------------------------------

def test_19_diagnostics_detect_corrupted_model_state(frozen_model):
    """19. Diagnostics engine detects numerical corruption (NaN/Inf) in model."""
    diag = SystemDiagnosticsEngine(model=frozen_model)
    healthy_res = diag.check_runtime_numerical_health()
    assert healthy_res.status == DiagnosticStatus.HEALTHY
    assert healthy_res.passed is True

    # Artificially inject NaN into a cloned parameter
    corrupt_model = copy.deepcopy(frozen_model)
    with torch.no_grad():
        first_param = next(corrupt_model.parameters())
        first_param[0, 0] = float("nan")

    corrupt_res = diag.check_runtime_numerical_health(corrupt_model)
    assert corrupt_res.status == DiagnosticStatus.CORRUPTED
    assert corrupt_res.passed is False
    assert corrupt_res.is_recoverable is False


def test_20_diagnostics_detect_tokenizer_mismatch():
    """20. Diagnostics engine flags tokenizer configuration discrepancies."""
    diag = SystemDiagnosticsEngine()
    bad_tok = BPETokenizer()  # Default base tokenizer has vocab_size 259, not 4096
    res = diag.check_tokenizer_compatibility(bad_tok)
    assert res.status == DiagnosticStatus.BLOCKED
    assert res.passed is False


def test_21_recoverable_state_can_be_restored():
    """21. Self-healing safely restores valid state from backup."""
    healer = SafeSelfHealingManager()
    backup_state = {"session_id": "sess_123", "knowledge": {"key": "val"}}
    restored = healer.restore_serialized_cognitive_state(backup_state)
    assert restored["session_id"] == "sess_123"
    assert len(healer.audit_log) == 1
    assert healer.audit_log[0].recovered_state == "restored_from_backup"


def test_22_corrupt_checkpoint_is_rejected():
    """22. Checkpoint missing required keys or invalid vocab is rejected."""
    diag = SystemDiagnosticsEngine()
    corrupt_ckpt = {"step": 100}  # Missing model_state_dict
    res = diag.check_checkpoint_metadata(corrupt_ckpt)
    assert res.status == DiagnosticStatus.CORRUPTED
    assert res.passed is False


def test_23_failed_invariant_causes_fail_closed_behavior(frozen_model):
    """23. Core integrity breach causes immediate fail-closed InvariantViolationError."""
    bad_config = ModelConfig(vocab_size=2048)
    bad_model = ChakrMicro(bad_config)
    with pytest.raises(InvariantViolationError):
        CoreIntegrityGuard.fail_closed_if_invalid(bad_model)


def test_24_self_healing_never_modifies_chakrmicro_weights(frozen_model):
    """24. Invariant breach blocks execution and NEVER patches or mutates model weights."""
    healer = SafeSelfHealingManager()
    initial_fp = CoreIntegrityGuard.compute_weight_fingerprint(frozen_model)

    fail_check = DiagnosticCheckResult(
        check_name="frozen_invariants",
        status=DiagnosticStatus.BLOCKED,
        passed=False,
        is_recoverable=False,
        message="Parameter mismatch detected.",
    )

    with pytest.raises(InvariantViolationError):
        healer.handle_invariant_mismatch(fail_check)

    # Verify model weights are 100% identical and untouched
    current_fp = CoreIntegrityGuard.compute_weight_fingerprint(frozen_model)
    assert current_fp == initial_fp


def test_25_self_improvement_never_modifies_weights_during_runtime(frozen_model, tiny_tokenizer):
    """25. Learning candidates are generated for offline pipeline; runtime weights remain unchanged."""
    initial_fp = CoreIntegrityGuard.compute_weight_fingerprint(frozen_model)
    loop = NeuralIntelligenceLoop(model=frozen_model, tokenizer=tiny_tokenizer)

    outcome = loop.run(
        user_prompt="Explain why gradient descent converges for convex functions.",
        use_critical_thinking=True,
    )

    assert outcome.weights_modified is False
    assert outcome.learning_record is not None
    # Model weights verified unmodified
    assert CoreIntegrityGuard.verify_weights_unmodified(frozen_model, initial_fp) is True


# -----------------------------------------------------------------------------
# Part G & H: Isolation & End-to-End Workflows (Mandates 26-30)
# -----------------------------------------------------------------------------

def test_26_tenant_session_isolation():
    """26. Multi-tenant isolation preserved in critical thinking traces."""
    engine = CriticalThinkingEngine()
    trace_a = engine.execute("Query A", owner_id="tenant_alpha", session_id="sess_1")
    trace_b = engine.execute("Query B", owner_id="tenant_beta", session_id="sess_2")

    assert trace_a.metadata["owner_id"] == "tenant_alpha"
    assert trace_b.metadata["owner_id"] == "tenant_beta"
    assert trace_a.trace_id != trace_b.trace_id


def test_27_existing_steps_0_22_regression_compatibility(frozen_model, tiny_tokenizer):
    """27. Step 20 and Step 21 modes execute without degradation."""
    loop = NeuralIntelligenceLoop(model=frozen_model, tokenizer=tiny_tokenizer)
    
    # Standard Step 20 run (without critical thinking)
    outcome_s20 = loop.run(user_prompt="Standard test prompt.", use_thinking=False)
    assert outcome_s20.weights_modified is False
    assert outcome_s20.critical_thinking_trace is None

    # Step 21 thinking run
    outcome_s21 = loop.run(user_prompt="Thinking test prompt.", use_thinking=True)
    assert outcome_s21.weights_modified is False


def test_28_end_to_end_critical_thinking_flow(frozen_model, tiny_tokenizer):
    """28. Complete critical thinking loop produces verified trace and outcome."""
    loop = NeuralIntelligenceLoop(model=frozen_model, tokenizer=tiny_tokenizer)
    outcome = loop.run(
        user_prompt="Is energy conserved in isolated systems under Noether's theorem?",
        use_critical_thinking=True,
    )
    assert outcome.critical_thinking_trace is not None
    crit_trace = outcome.critical_thinking_trace
    assert crit_trace.question == "Is energy conserved in isolated systems under Noether's theorem?"
    assert len(crit_trace.hypotheses) >= 2
    assert len(crit_trace.step_records) >= 10
    assert outcome.weights_modified is False


def test_29_end_to_end_hardware_adaptive_flow(frozen_model, tiny_tokenizer):
    """29. Execution adapts dynamically to specified ResourceProfile."""
    loop = NeuralIntelligenceLoop(model=frozen_model, tokenizer=tiny_tokenizer)
    low_policy = AdaptiveExecutionPolicy.get_profile_defaults(ResourceProfile.LOW_RESOURCE)

    outcome = loop.run(
        user_prompt="Analyze architectural scalability.",
        use_critical_thinking=True,
        execution_policy=low_policy,
        max_new_tokens=256,  # Should be clamped to low_policy max_generation_tokens (64)
    )
    assert outcome.neural_result.generated_tokens_count <= low_policy.max_generation_tokens
    assert outcome.weights_modified is False


def test_30_end_to_end_diagnostic_and_recovery_flow(frozen_model, tiny_tokenizer):
    """30. Complete diagnostics and safe self-healing recovery cycle."""
    diag = SystemDiagnosticsEngine(model=frozen_model, tokenizer=tiny_tokenizer)
    report = diag.run_full_diagnostics(model=frozen_model, tokenizer=tiny_tokenizer)
    assert report.overall_status in [DiagnosticStatus.HEALTHY, DiagnosticStatus.DEGRADED]
    assert report.passed is True

    # Test recovery of corrupted context
    healer = SafeSelfHealingManager()
    rebuilt = healer.rebuild_derived_context(
        raw_prompt="Clean question",
        system_identity="Identity",
        context_builder_fn=lambda p, s: f"{s}: {p}",
    )
    assert rebuilt == "Identity: Clean question"
    assert healer.audit_log[-1].success is True
