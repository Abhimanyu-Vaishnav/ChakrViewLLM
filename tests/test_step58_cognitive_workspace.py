"""
Step 58 Automated Tests: Cognitive Workspace, Multi-Turn Execution, Memory Consolidation & Safety.

Tests verify:
  01. CognitiveWorkingState schema, deterministic transition, and serialization roundtrip.
  02. MemoryConsolidator admission rules: rejects unverified episodes, admits verified ones.
  03. MemoryConsolidator evidence requirement: requires >= 2 independent verified experiences to promote semantic pattern.
  04. ExplainableMemoryRetriever deterministic scoring and explainable breakdown.
  05. StructuredReflector trajectory analysis, failure breakdown, and transfer boundary synthesis.
  06. CognitiveWorkspace multi-turn execution (UNDERSTAND -> PLAN -> ACT -> OBSERVE -> DIAGNOSE -> CORRECT -> VERIFY -> REMEMBER -> CONSOLIDATE -> REUSE -> REFLECT).
  07. Trajectory logs strictly adhere to Step 54 XML format.
  08. Memory Value metric: verifies positive transfer (attempts reduced from 2 to 1 with verified experience).
  09. Memory Ablation test: disabling memory forces re-encounter task to execute all repair attempts.
  10. Negative-transfer safety: unrelated / mismatched task receives zero or penalized memory score.
  11. Unrelated control: Task B executes cleanly without interference from arithmetic repair memory.
  12. Frozen baseline immutability invariant: model parameters and SHA-256 weight hash unchanged before and after execution.
"""

from __future__ import annotations

import pytest
import torch

from chakrview.arena.evaluator import ArenaEvaluator
from chakrview.learning.episode import Attempt, LearningEpisode
from chakrview.learning.experience import ExperienceRecord, ExperienceExtractor
from chakrview.runtime.interactive import (
    instantiate_frozen_baseline,
    compute_model_hash,
)
from chakrview.runtime.pipeline import EXPECTED_WEIGHT_HASH
from chakrview.cognition.workspace.state import (
    CognitiveWorkingState,
    CognitiveStatePhase,
    FailedAttemptSummary,
)
from chakrview.cognition.workspace.consolidation import (
    SemanticMemoryEntry,
    MemoryConsolidator,
)
from chakrview.cognition.workspace.retrieval import (
    ExplainableMemoryRetriever,
    MemoryRetrievalResult,
)
from chakrview.cognition.workspace.reflection import (
    StructuredReflector,
    EpisodeReflection,
)
from chakrview.cognition.workspace.workspace import CognitiveWorkspace
from scripts.experiment_step57_learning_loop import (
    TASK_A_SPEC,
    TASK_A_PRIME_SPEC,
    TASK_B_SPEC,
)


@pytest.fixture(scope="module")
def baseline_model():
    return instantiate_frozen_baseline()


# ---------------------------------------------------------------------------
# Test 01: CognitiveWorkingState schema & serialization
# ---------------------------------------------------------------------------
def test_01_working_state_serialization():
    state = CognitiveWorkingState(
        task_id="TASK_TEST_01",
        task_family="function_repair",
        objective="Repair defective add function",
        current_phase=CognitiveStatePhase.INITIALIZING,
    )
    state.transition_to(CognitiveStatePhase.PLANNING)
    state.current_plan = ["step 1", "step 2"]
    state.record_failure(
        action="def add(a, b): return a - b",
        observation="AssertionError",
        diagnosis="Expected addition",
        duration=0.05,
    )

    d = state.to_dict()
    assert d["task_id"] == "TASK_TEST_01"
    assert d["current_phase"] == "PLANNING"
    assert len(d["failure_history"]) == 1
    assert d["failure_history"][0]["diagnosis"] == "Expected addition"

    restored = CognitiveWorkingState.from_dict(d)
    assert restored.task_id == "TASK_TEST_01"
    assert restored.current_phase == CognitiveStatePhase.PLANNING
    assert len(restored.failure_history) == 1
    assert restored.failure_history[0].observation == "AssertionError"


# ---------------------------------------------------------------------------
# Test 02: MemoryConsolidator admission rules
# ---------------------------------------------------------------------------
def test_02_consolidator_admission_rules():
    consolidator = MemoryConsolidator(min_evidence_threshold=2)

    # 1. Unverified record -> Rejected
    unverified_rec = ExperienceRecord(
        experience_id="exp_fail_01",
        task_id="TASK_FAIL",
        task_family="function_repair",
        task_description="Failed repair",
        context={},
        attempt_count=2,
        initial_action="def f(): pass",
        failure_observation="AssertionError",
        diagnosis="Error",
        correction=None,
        verified_result="FAILURE",
        what_worked="Did not converge to verified solution",
        what_failed="AssertionError",
        reusable_pattern="Unresolved failure",
        verified=False,
        timestamp_utc="2026-09-30T00:00:00Z",
    )
    assert consolidator.admit_experience(unverified_rec) is False
    assert len(consolidator.episodic_store) == 0

    # 2. Verified record -> Admitted
    verified_rec = ExperienceRecord(
        experience_id="exp_succ_01",
        task_id="TASK_SUCC",
        task_family="function_repair",
        task_description="Successful repair",
        context={},
        attempt_count=2,
        initial_action="def add(a, b): return a - b",
        failure_observation="assert add(2, 3) == 5 failed",
        diagnosis="Expected addition",
        correction="def add(a, b): return a + b",
        verified_result="SUCCESS",
        what_worked="def add(a, b): return a + b",
        what_failed="assert add(2, 3) == 5 failed",
        reusable_pattern="Diagnosis: Expected addition; Solution: def add(a, b): return a + b",
        verified=True,
        timestamp_utc="2026-09-30T00:00:00Z",
    )
    assert consolidator.admit_experience(verified_rec) is True
    assert len(consolidator.episodic_store) == 1


# ---------------------------------------------------------------------------
# Test 03: Evidence requirement for semantic pattern promotion (>= 2)
# ---------------------------------------------------------------------------
def test_03_consolidator_evidence_threshold():
    consolidator = MemoryConsolidator(min_evidence_threshold=2)

    exp1 = ExperienceRecord(
        experience_id="exp_01",
        task_id="TASK_ADD",
        task_family="function_repair",
        task_description="Fix add",
        context={},
        attempt_count=2,
        initial_action="def add(a, b): return a - b",
        failure_observation="AssertionError",
        diagnosis="Operator error: assert failed",
        correction="def add(a, b): return a + b",
        verified_result="SUCCESS",
        what_worked="def add(a, b): return a + b",
        what_failed="assert failed",
        reusable_pattern="Invert - to +",
        verified=True,
        timestamp_utc="2026-09-30T00:00:00Z",
    )
    consolidator.admit_experience(exp1)
    promoted = consolidator.consolidate()
    # Only 1 episode: cannot promote yet
    assert len(promoted) == 0
    assert len(consolidator.semantic_store) == 0

    exp2 = ExperienceRecord(
        experience_id="exp_02",
        task_id="TASK_SUB",
        task_family="function_repair",
        task_description="Fix sub",
        context={},
        attempt_count=2,
        initial_action="def sub(a, b): return a + b",
        failure_observation="AssertionError",
        diagnosis="Operator error: assert failed",
        correction="def sub(a, b): return a - b",
        verified_result="SUCCESS",
        what_worked="def sub(a, b): return a - b",
        what_failed="assert failed",
        reusable_pattern="Invert + to -",
        verified=True,
        timestamp_utc="2026-09-30T00:00:00Z",
    )
    consolidator.admit_experience(exp2)
    promoted2 = consolidator.consolidate()
    # Now 2 independent verified episodes: promotes 1 consolidated semantic pattern
    assert len(promoted2) == 1
    assert len(consolidator.semantic_store) == 1
    pattern = promoted2[0]
    assert pattern.task_family == "function_repair"
    assert pattern.evidence_count == 2
    assert "exp_01" in pattern.supporting_evidence
    assert "exp_02" in pattern.supporting_evidence


# ---------------------------------------------------------------------------
# Test 04: ExplainableMemoryRetriever scoring & breakdown
# ---------------------------------------------------------------------------
def test_04_retriever_explainable_ranking():
    consolidator = MemoryConsolidator(min_evidence_threshold=2)
    exp1 = ExperienceRecord(
        experience_id="exp_add",
        task_id="TASK_ADD",
        task_family="function_repair",
        task_description="Fix add",
        context={},
        attempt_count=2,
        initial_action="def add(a, b): return a - b",
        failure_observation="AssertionError",
        diagnosis="Operator error in add",
        correction="def add(a, b): return a + b",
        verified_result="SUCCESS",
        what_worked="def add(a, b): return a + b",
        what_failed="assert failed",
        reusable_pattern="Successful Action: def add(a, b):\n    return a + b\n",
        verified=True,
        timestamp_utc="2026-09-30T00:00:00Z",
    )
    exp2 = ExperienceRecord(
        experience_id="exp_sub",
        task_id="TASK_SUB",
        task_family="function_repair",
        task_description="Fix sub",
        context={},
        attempt_count=2,
        initial_action="def sub(a, b): return a + b",
        failure_observation="AssertionError",
        diagnosis="Operator error in sub",
        correction="def sub(a, b): return a - b",
        verified_result="SUCCESS",
        what_worked="def sub(a, b): return a - b",
        what_failed="assert failed",
        reusable_pattern="Successful Action: def sub(a, b):\n    return a - b\n",
        verified=True,
        timestamp_utc="2026-09-30T00:00:00Z",
    )
    consolidator.admit_experience(exp1)
    consolidator.admit_experience(exp2)
    consolidator.consolidate()

    retriever = ExplainableMemoryRetriever()
    results = retriever.retrieve(
        task_family="function_repair",
        objective="Fix defect in add",
        current_diagnosis="Operator error in add",
        consolidator=consolidator,
        top_k=3,
    )

    assert len(results) > 0
    top = results[0]
    assert top.score > 0.5
    assert top.family_match_score == 1.0
    assert "family_match=1.00" in top.explanation
    assert top.reusable_content != ""


# ---------------------------------------------------------------------------
# Test 05: StructuredReflector analysis & boundary limits
# ---------------------------------------------------------------------------
def test_05_structured_reflector():
    ep = LearningEpisode(
        episode_id="ep_refl_01",
        task_id="TASK_REFL",
        task_spec={"category": "function_repair", "description": "Fix defect"},
        initial_context={},
        final_result="SUCCESS",
    )
    ep.add_attempt(Attempt(1, "EXECUTED", "def f(): return 1 - 2", "AssertionError", "FAIL", diagnosis="subtraction defect", correction="def f(): return 1 + 2"))
    ep.add_attempt(Attempt(2, "EXECUTED", "def f(): return 1 + 2", "PASSED", "PASS"))

    reflection = StructuredReflector.reflect(ep)
    assert reflection.task_id == "TASK_REFL"
    assert len(reflection.what_failed) == 1
    assert "subtraction defect" in reflection.what_failed[0]
    assert "Verified action" in reflection.what_worked
    assert reflection.transfer_confidence > 0.8
    assert len(reflection.boundary_limits) >= 3


# ---------------------------------------------------------------------------
# Test 06: CognitiveWorkspace multi-turn execution
# ---------------------------------------------------------------------------
def test_06_cognitive_workspace_execution():
    workspace = CognitiveWorkspace(max_attempts=3)
    episode = workspace.run_task(
        task_id=TASK_A_SPEC["task_id"],
        task_spec=TASK_A_SPEC,
        initial_action=TASK_A_SPEC["initial_defective_code"],
    )

    assert episode.final_result == "SUCCESS"
    assert len(episode.attempts) == 2
    assert episode.attempts[0].evaluation == "FAIL"
    assert episode.attempts[1].evaluation == "PASS"

    assert "UNDERSTAND" in episode.executed_stages
    assert "PLAN" in episode.executed_stages
    assert "ACT" in episode.executed_stages
    assert "OBSERVE" in episode.executed_stages
    assert "DIAGNOSE" in episode.executed_stages
    assert "CORRECT" in episode.executed_stages
    assert "VERIFY" in episode.executed_stages
    assert "REMEMBER" in episode.executed_stages
    assert "REFLECT" in episode.executed_stages

    # Working state reflects verified completion
    assert workspace.working_state is not None
    assert workspace.working_state.verified is True
    assert workspace.working_state.completed is True


# ---------------------------------------------------------------------------
# Test 07: Trajectory log conformity (Step 54)
# ---------------------------------------------------------------------------
def test_07_trajectory_log_format():
    workspace = CognitiveWorkspace(max_attempts=3)
    workspace.run_task(
        task_id=TASK_A_SPEC["task_id"],
        task_spec=TASK_A_SPEC,
        initial_action=TASK_A_SPEC["initial_defective_code"],
    )

    assert len(workspace.trajectory_logs) == 2
    log1 = workspace.trajectory_logs[0]
    assert "<TRAJECTORY>" in log1
    assert "<SPEC>" in log1
    assert "<STATE>" in log1
    assert "<ACTION>" in log1
    assert "<OBSERVATION>" in log1
    assert "<DIAGNOSIS>" in log1
    assert "<NEXT_ACTION>" in log1
    assert "<RESULT>\nFAILURE\n</RESULT>" in log1

    log2 = workspace.trajectory_logs[1]
    assert "<RESULT>\nSUCCESS\n</RESULT>" in log2


# ---------------------------------------------------------------------------
# Test 08: Memory Value metric (attempts reduced from 2 to 1)
# ---------------------------------------------------------------------------
def test_08_memory_value_positive_transfer():
    workspace = CognitiveWorkspace(max_attempts=3)

    # Run 1: Cold start (no memory) -> takes 2 attempts
    ep1 = workspace.run_task(
        task_id=TASK_A_SPEC["task_id"],
        task_spec=TASK_A_SPEC,
        initial_action=TASK_A_SPEC["initial_defective_code"],
    )
    attempts_cold = len(ep1.attempts)
    assert attempts_cold == 2

    # Run 2: Re-encounter Task A with verified experience admitted into consolidator
    ep2 = workspace.run_task(
        task_id=TASK_A_SPEC["task_id"],
        task_spec=TASK_A_SPEC,
        initial_action=TASK_A_SPEC["initial_defective_code"],
    )
    attempts_conditioned = len(ep2.attempts)
    assert attempts_conditioned == 1

    memory_value = attempts_cold - attempts_conditioned
    assert memory_value == 1  # Positive transfer


# ---------------------------------------------------------------------------
# Test 09: Memory Ablation test (disabling memory removes shortcut)
# ---------------------------------------------------------------------------
def test_09_memory_ablation():
    workspace = CognitiveWorkspace(max_attempts=3)

    # Populate memory
    workspace.run_task(
        task_id=TASK_A_SPEC["task_id"],
        task_spec=TASK_A_SPEC,
        initial_action=TASK_A_SPEC["initial_defective_code"],
    )

    # Re-encounter WITH memory disabled (ablation)
    ep_ablated = workspace.run_task(
        task_id=TASK_A_SPEC["task_id"],
        task_spec=TASK_A_SPEC,
        initial_action=TASK_A_SPEC["initial_defective_code"],
        enable_memory_retrieval=False,
    )
    # Without memory, initial action cannot be shortcutted; takes 2 attempts
    assert len(ep_ablated.attempts) == 2


# ---------------------------------------------------------------------------
# Test 10: Negative-transfer safety (mismatched family receives 0/penalized score)
# ---------------------------------------------------------------------------
def test_10_negative_transfer_safety():
    workspace = CognitiveWorkspace(max_attempts=3)

    # Populate arithmetic repair memory
    workspace.run_task(
        task_id=TASK_A_SPEC["task_id"],
        task_spec=TASK_A_SPEC,
        initial_action=TASK_A_SPEC["initial_defective_code"],
    )

    # Try to retrieve memory for an unrelated state machine task
    results = workspace.retriever.retrieve(
        task_family="state_machine",
        objective="Transition state from LOCKED to OPEN",
        current_diagnosis="Door is LOCKED",
        consolidator=workspace.consolidator,
    )

    # Incompatible family should receive score 0.0 or be penalized below useful threshold
    for r in results:
        assert r.family_match_score == 0.0
        assert r.score < 0.35


# ---------------------------------------------------------------------------
# Test 11: Unrelated control (Task B unaffected)
# ---------------------------------------------------------------------------
def test_11_unrelated_control_execution():
    workspace = CognitiveWorkspace(max_attempts=3)

    # Execute arithmetic task first
    workspace.run_task(
        task_id=TASK_A_SPEC["task_id"],
        task_spec=TASK_A_SPEC,
        initial_action=TASK_A_SPEC["initial_defective_code"],
    )

    # Execute unrelated state machine task B
    ep_b = workspace.run_task(
        task_id=TASK_B_SPEC["task_id"],
        task_spec=TASK_B_SPEC,
        initial_action=TASK_B_SPEC["initial_defective_code"],
    )

    assert ep_b.final_result == "SUCCESS"
    assert ep_b.task_id == TASK_B_SPEC["task_id"]
    assert len(ep_b.attempts) == 2


# ---------------------------------------------------------------------------
# Test 12: Frozen baseline immutability invariant (DeltaW_base == 0)
# ---------------------------------------------------------------------------
def test_12_frozen_baseline_immutability(baseline_model):
    hash_pre = compute_model_hash(baseline_model)
    assert hash_pre == EXPECTED_WEIGHT_HASH

    workspace = CognitiveWorkspace(max_attempts=3)
    workspace.run_task(
        task_id=TASK_A_SPEC["task_id"],
        task_spec=TASK_A_SPEC,
        initial_action=TASK_A_SPEC["initial_defective_code"],
    )
    workspace.run_task(
        task_id=TASK_A_PRIME_SPEC["task_id"],
        task_spec=TASK_A_PRIME_SPEC,
        initial_action=TASK_A_PRIME_SPEC["initial_defective_code"],
    )
    workspace.run_task(
        task_id=TASK_B_SPEC["task_id"],
        task_spec=TASK_B_SPEC,
        initial_action=TASK_B_SPEC["initial_defective_code"],
    )

    hash_post = compute_model_hash(baseline_model)
    assert hash_post == EXPECTED_WEIGHT_HASH
    assert hash_pre == hash_post
