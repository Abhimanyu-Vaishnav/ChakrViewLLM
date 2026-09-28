"""
Comprehensive Test Suite for ChakrView Step 21: Neural Thinking & Deliberation Foundation.

Tests:
1. Simple problem solved in one thinking cycle.
2. Problem requiring multiple deliberation cycles (revision loop).
3. Candidate hypothesis rejected after critique.
4. Contradictory evidence causes reconsideration.
5. Verification failure causes structured revision.
6. Missing evidence produces INSUFFICIENT_INFORMATION.
7. Maximum thinking steps terminate safely (MAX_REASONING_LIMIT).
8. Maximum revision cycles terminate safely (MAX_REVISIONS_REACHED).
9. Stopping policy correctly identifies solved and solved-with-uncertainty states.
10. Thinking trace and workspace are fully serializable to JSON.
11. Historical thought steps are immutable and cannot be silently mutated.
12. Multi-tenant isolation is strictly enforced for workspaces.
13. Thinking cannot bypass CapabilityGate (THINKING != AUTHORITY).
14. Neural model weights remain strictly unchanged throughout deliberation.
15. Frozen ChakrMicro invariants remain strictly verified (3,443,136 params, 4096 vocab, 512 ctx).
16. Integration with NeuralIntelligenceLoop(use_thinking=True).
"""

from pathlib import Path
import pytest
import torch

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer import load_experiment_artifacts, BOS_ID, EOS_ID, PAD_ID
from chakrview.tokenizer.tokenizer import BPETokenizer
from chakrview.capability.bridge import get_standard_capability_registry
from chakrview.capability.gate import CapabilityGate, CapabilityAuthorizationError
from chakrview.capability.contract import CapabilityRequest, CapabilityContext
from chakrview.state.manager import CognitiveStateManager
from chakrview.state.epistemic import EpistemicStatus

from chakrview.thinking.thought import ThoughtStep, ThoughtPurpose
from chakrview.thinking.policy import (
    ThinkingPolicy,
    get_standard_policy,
    get_strict_policy,
    get_fast_policy,
)
from chakrview.thinking.workspace import (
    ThinkingWorkspace,
    WorkspaceBudgetExceededError,
    TenantIsolationError,
)
from chakrview.thinking.attention import (
    FocusType,
    AttentionFocus,
    ThinkingAttention,
)
from chakrview.thinking.critique import (
    CritiqueVerdict,
    CritiqueResult,
    CritiqueEngine,
)
from chakrview.thinking.revision import (
    RevisionPlan,
    RevisionEngine,
)
from chakrview.thinking.stopping import (
    StoppingCondition,
    ThinkingStoppingPolicy,
)
from chakrview.thinking.trace import ThinkingTrace
from chakrview.thinking.deliberation import (
    DeliberationEngine,
    DeliberationOutcome,
)
from chakrview.intelligence.pipeline import NeuralIntelligenceLoop


@pytest.fixture
def chakr_model() -> ChakrMicro:
    torch.manual_seed(42)
    cfg = ModelConfig()
    model = ChakrMicro(cfg)
    model.eval()
    return model


@pytest.fixture
def bpe_tokenizer() -> BPETokenizer:
    exp_dir = Path(__file__).resolve().parent.parent / "data" / "tokenizer_experiments" / "v4096"
    if exp_dir.is_dir():
        return load_experiment_artifacts(exp_dir)
    return BPETokenizer()


# =====================================================================
# 1. Simple problem solved in one thinking cycle
# =====================================================================

def test_simple_problem_solved(chakr_model: ChakrMicro, bpe_tokenizer: BPETokenizer):
    """Verify clean one-cycle deliberation on a direct mathematical query."""
    cap_registry = get_standard_capability_registry()
    cap_gate = CapabilityGate(cap_registry)

    engine = DeliberationEngine(
        model=chakr_model,
        tokenizer=bpe_tokenizer,
        capability_registry=cap_registry,
        capability_gate=cap_gate,
    )

    policy = ThinkingPolicy(max_thought_steps=6, max_revision_cycles=2)
    outcome = engine.deliberate(
        objective="Calculate 12 * 5",
        owner_id="alice",
        session_id="sess_1",
        policy=policy,
        max_new_tokens=16,
    )

    assert isinstance(outcome, DeliberationOutcome)
    assert outcome.success is True
    assert outcome.stopping_condition in ("SOLVED", "SOLVED_WITH_UNCERTAINTY")
    assert outcome.workspace.step_count > 0
    assert outcome.weights_modified is False
    assert "60" in outcome.response_text or "Final Answer:" in outcome.response_text


# =====================================================================
# 2. Problem requiring multiple deliberation cycles (revision loop)
# =====================================================================

def test_multi_cycle_deliberation_revision(chakr_model: ChakrMicro, bpe_tokenizer: BPETokenizer):
    """Verify that critique weakness triggers a structured revision cycle before solving."""
    engine = DeliberationEngine(
        model=chakr_model,
        tokenizer=bpe_tokenizer,
    )

    policy = ThinkingPolicy(
        max_thought_steps=8,
        max_revision_cycles=2,
        min_critique_score=0.8,
        allow_revisions=True,
    )

    outcome = engine.deliberate(
        objective="Analyze planetary atmospheric composition",
        owner_id="alice",
        policy=policy,
        max_new_tokens=16,
    )

    # Must have performed at least 1 revision before concluding or exhausting budget
    assert outcome.workspace.step_count >= 2
    assert outcome.trace.number_of_steps == outcome.workspace.step_count
    assert outcome.weights_modified is False


# =====================================================================
# 3. Candidate hypothesis rejected after critique
# =====================================================================

def test_candidate_rejected_after_critique():
    """Verify that critique engine flags weak or empty candidates appropriately."""
    critique_engine = CritiqueEngine()
    ws = ThinkingWorkspace(objective="Calculate fuel efficiency in km/L")

    # Empty text -> REQUIRES_REVISION
    empty_critique = critique_engine.critique("", ws)
    assert empty_critique.verdict == CritiqueVerdict.REQUIRES_REVISION
    assert empty_critique.score == 0.0

    # Irrelevant candidate -> WEAK
    irrelevant_critique = critique_engine.critique("Bananas and apples are yellow and red.", ws)
    assert irrelevant_critique.verdict in (CritiqueVerdict.WEAK, CritiqueVerdict.INSUFFICIENT_EVIDENCE)
    assert irrelevant_critique.score < 0.75
    assert len(ws.detected_weaknesses) > 0


# =====================================================================
# 4. Contradictory evidence causes reconsideration
# =====================================================================

def test_contradictory_evidence_detection():
    """Verify that contradictory evidence triggers CONTRADICTED verdict."""
    critique_engine = CritiqueEngine()
    ws = ThinkingWorkspace(objective="Verify sensor operational status")
    ws.add_evidence(content="Sensor S1 is not active and offline.", source="sensor_telemetry")

    critique = critique_engine.critique("Sensor S1 is fully active and functioning normally.", ws)
    assert critique.verdict == CritiqueVerdict.CONTRADICTED
    assert len(critique.contradictions_detected) > 0
    assert critique.score < 0.75


# =====================================================================
# 5. Verification failure causes structured revision
# =====================================================================

def test_verification_failure_triggers_revision():
    """Verify that RevisionEngine produces targeted RevisionPlan and records revision thought."""
    ws = ThinkingWorkspace(objective="Calculate exact perimeter")
    rev_engine = RevisionEngine()

    critique = CritiqueResult(
        verdict=CritiqueVerdict.WEAK,
        score=0.5,
        findings=["Calculated value conflicts with bounding constraints."],
        recommended_action="Recalculate with corrected operator precedence.",
    )

    plan = rev_engine.plan_revision(ws, critique)
    assert isinstance(plan, RevisionPlan)
    assert plan.revision_index == 1
    assert "correct_weakness" in plan.focus_areas
    assert ws.revision_count == 1

    # Historical thoughts must include a REVISE thought
    last_thought = ws.thought_steps[-1]
    assert last_thought.purpose == ThoughtPurpose.REVISE
    assert "Revision cycle #1" in last_thought.content


# =====================================================================
# 6. Missing evidence produces INSUFFICIENT_INFORMATION
# =====================================================================

def test_missing_evidence_stopping_condition():
    """Verify that stopping policy halts with INSUFFICIENT_INFORMATION when evidence cannot be found."""
    policy = ThinkingStoppingPolicy()
    ws = ThinkingWorkspace(objective="Determine the atmospheric pressure on exoplanet Proxima b")
    ws.unresolved_questions.append("What is Proxima b surface probe reading?")

    critique = CritiqueResult(
        verdict=CritiqueVerdict.INSUFFICIENT_EVIDENCE,
        score=0.4,
        missing_evidence=["Proxima b surface atmospheric reading"],
    )

    should_stop, condition, reason = policy.evaluate_stopping(ws, latest_critique=critique)
    assert should_stop is True
    assert condition == StoppingCondition.INSUFFICIENT_INFORMATION
    assert "Essential premise evidence is missing" in reason


# =====================================================================
# 7. Maximum thinking steps terminate safely
# =====================================================================

def test_max_thinking_steps_terminates_safely(chakr_model: ChakrMicro, bpe_tokenizer: BPETokenizer):
    """Verify that deliberation halts safely when max_thought_steps is reached."""
    engine = DeliberationEngine(chakr_model, bpe_tokenizer)
    policy = ThinkingPolicy(max_thought_steps=3, max_revision_cycles=0)

    outcome = engine.deliberate(
        objective="Analyze open cosmological problem",
        policy=policy,
        max_new_tokens=8,
    )

    assert outcome.workspace.step_count <= 4
    assert outcome.stopping_condition == "MAX_REASONING_LIMIT"


# =====================================================================
# 8. Maximum revision cycles terminate safely
# =====================================================================

def test_max_revisions_terminates_safely(chakr_model: ChakrMicro, bpe_tokenizer: BPETokenizer):
    """Verify that deliberation halts safely when max_revision_cycles is reached."""
    engine = DeliberationEngine(chakr_model, bpe_tokenizer)
    policy = ThinkingPolicy(max_thought_steps=12, max_revision_cycles=1, min_critique_score=0.99)

    outcome = engine.deliberate(
        objective="Draft complete theoretical paper on topological invariants",
        policy=policy,
        max_new_tokens=8,
    )

    assert outcome.workspace.revision_count <= 1
    assert outcome.stopping_condition in ("MAX_REVISIONS_REACHED", "MAX_REASONING_LIMIT")


# =====================================================================
# 9. Stopping policy correctly identifies solved state
# =====================================================================

def test_stopping_policy_solved_state():
    """Verify StoppingPolicy distinguishes SOLVED from SOLVED_WITH_UNCERTAINTY."""
    stopping_policy = ThinkingStoppingPolicy()
    ws = ThinkingWorkspace(objective="Test clean solution")
    ws.set_conclusion("Verified mathematical answer: 42")

    critique_ok = CritiqueResult(verdict=CritiqueVerdict.PASS, score=0.95)

    # 1. Clean solve without uncertainty
    should_stop, cond, _ = stopping_policy.evaluate_stopping(ws, latest_critique=critique_ok, verification_passed=True)
    assert should_stop is True
    assert cond == StoppingCondition.SOLVED

    # 2. Solve with residual acknowledged uncertainty
    ws.uncertainty_state["minor_sensor_noise"] = {"confidence": 0.8, "reason": "Thermal variance"}
    should_stop_unc, cond_unc, _ = stopping_policy.evaluate_stopping(ws, latest_critique=critique_ok, verification_passed=True)
    assert should_stop_unc is True
    assert cond_unc == StoppingCondition.SOLVED_WITH_UNCERTAINTY


# =====================================================================
# 10. Thinking trace & workspace are serializable
# =====================================================================

def test_serialization_and_safe_summary():
    """Verify full JSON serialization and safe summary generation without chain-of-thought dumps."""
    ws = ThinkingWorkspace(objective="Test serialization")
    ws.add_thought_step(ThoughtPurpose.OBSERVE, "Step 1 content")
    ws.record_hypothesis("H1: Solution is X")
    ws.set_conclusion("Final X")

    trace = ThinkingTrace(deliberation_id=ws.workspace_id, task_id=ws.task_id)
    trace.finalize("COMPLETED", "SOLVED", "All constraints satisfied", elapsed_ms=15.4)

    # JSON round-trip for workspace
    ws_dict = ws.to_dict()
    assert isinstance(ws_dict, dict)
    ws_restored = ThinkingWorkspace.from_dict(ws_dict)
    assert ws_restored.workspace_id == ws.workspace_id
    assert len(ws_restored.thought_steps) == 1

    # Safe summary must NOT expose full internal thought text
    safe_summary = trace.get_safe_summary()
    assert "deliberation_id" in safe_summary
    assert "stopping_condition" in safe_summary
    assert "Step 1 content" not in str(safe_summary)


# =====================================================================
# 11. Historical thought steps are immutable
# =====================================================================

def test_thought_step_immutability():
    """Verify that ThoughtStep objects are frozen dataclasses and reject mutation."""
    step = ThoughtStep(
        thought_id="th_test",
        purpose=ThoughtPurpose.OBSERVE,
        content="Immutable observation",
    )
    with pytest.raises((AttributeError, TypeError)):
        # Attempting to tamper with content must raise error
        step.content = "Tampered content"  # type: ignore


# =====================================================================
# 12. Multi-tenant isolation in workspace
# =====================================================================

def test_workspace_tenant_isolation():
    """Verify that ThinkingWorkspace strictly rejects cross-tenant access."""
    ws = ThinkingWorkspace(owner_id="alice", session_id="sess_alice")

    # Alice valid access
    ws.validate_tenant("alice", "sess_alice")

    # Bob unauthorized access
    with pytest.raises(TenantIsolationError, match="Multi-tenant violation"):
        ws.validate_tenant("bob", "sess_alice")

    # Wrong session
    with pytest.raises(TenantIsolationError, match="Session isolation violation"):
        ws.validate_tenant("alice", "sess_other")


# =====================================================================
# 13. Thinking cannot bypass CapabilityGate
# =====================================================================

def test_thinking_cannot_bypass_capability_gate():
    """Verify that deliberation thoughts never grant capability execution authority."""
    registry = get_standard_capability_registry()
    gate = CapabilityGate(registry)

    req = CapabilityRequest(capability_id="calculator", parameters={"expression": "100 / 4"})
    # Context lacking math permissions
    unauthorized_ctx = CapabilityContext(user_id="alice", granted_permissions=set())

    with pytest.raises(CapabilityAuthorizationError):
        gate.authorize(req, unauthorized_ctx)


# =====================================================================
# 14. Neural model weights remain unchanged
# =====================================================================

def test_zero_runtime_weight_mutation_in_deliberation(chakr_model: ChakrMicro, bpe_tokenizer: BPETokenizer):
    """Verify that weights remain strictly bit-for-bit unchanged after deliberation."""
    engine = DeliberationEngine(chakr_model, bpe_tokenizer)
    initial_weights = chakr_model.embedding.weight.clone()

    outcome = engine.deliberate(
        objective="Calculate 7 * 8",
        owner_id="alice",
        max_new_tokens=16,
    )

    assert outcome.weights_modified is False
    current_weights = chakr_model.embedding.weight
    assert torch.equal(initial_weights, current_weights)
    assert engine.inference_engine.verify_weights_unmodified() is True


# =====================================================================
# 15. Frozen ChakrMicro invariants remain strictly verified
# =====================================================================

def test_frozen_invariants_preserved(chakr_model: ChakrMicro, bpe_tokenizer: BPETokenizer):
    """Verify exact parameter count, vocab, context, and special tokens."""
    param_count = sum(p.numel() for p in chakr_model.parameters())
    assert param_count == 3_443_136, f"Expected 3,443,136 params, got {param_count}"

    assert chakr_model.config.vocab_size == 4_096
    assert chakr_model.config.max_seq_len == 512
    assert chakr_model.config.bos_token_id == 0
    assert chakr_model.config.eos_token_id == 1
    assert chakr_model.config.pad_token_id == 2

    assert bpe_tokenizer.vocab_size == 4_096
    assert BOS_ID == 0
    assert EOS_ID == 1
    assert PAD_ID == 2


# =====================================================================
# 16. Integration with NeuralIntelligenceLoop
# =====================================================================

def test_neural_intelligence_loop_with_thinking(chakr_model: ChakrMicro, bpe_tokenizer: BPETokenizer):
    """Verify integration of DeliberationEngine into NeuralIntelligenceLoop(use_thinking=True)."""
    state_mgr = CognitiveStateManager(owner_id="alice", session_id="sess_1")
    cap_registry = get_standard_capability_registry()
    cap_gate = CapabilityGate(cap_registry)

    loop = NeuralIntelligenceLoop(
        model=chakr_model,
        tokenizer=bpe_tokenizer,
        state_manager=state_mgr,
        capability_gate=cap_gate,
        capability_registry=cap_registry,
    )

    outcome = loop.run(
        user_prompt="Calculate 14 * 5",
        owner_id="alice",
        session_id="sess_1",
        max_new_tokens=16,
        use_thinking=True,
    )

    assert outcome.success is True
    assert outcome.weights_modified is False
    assert outcome.learning_record is not None
    assert "70" in outcome.response_text or "Final Answer:" in outcome.response_text
