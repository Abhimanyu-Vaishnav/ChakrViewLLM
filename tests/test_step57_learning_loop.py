"""
Step 57 Automated Tests: Controlled Cognitive Learning Loop & Experiential Replay.

Tests verify:
  01. LearningEpisode creation, attempt recording, and canonical serialization.
  02. ExperienceRecord extraction from verified episode (and rejection of unverified).
  03. CognitiveLearningLoop stores and retrieves experience by task ID and task family.
  04. CognitiveLearningLoop executes all stages: UNDERSTAND -> ACT -> OBSERVE -> EVALUATE -> DIAGNOSE -> CORRECT -> RETRY -> VERIFY -> REMEMBER -> LEARN.
  05. Memory-conditioned repeat: Task A resolves in 2 attempts on Run 1, but converges in 1 attempt on Run 2 with prior experience (Delta_exp = +1 attempt reduction).
  06. Transfer test: Related Task A' retrieves family-level experience.
  07. Unrelated control: Task B executes independently.
  08. PromotionGateController promotes valid candidate with DeltaW_base == 0.
  09. PromotionGateController rejects candidate if regression exceeds 2%.
  10. PromotionGateController rejects candidate if base core is mutated.
  11. Rollback test: mounting and unmounting adapter leaves base hash bit-exact identical.
  12. Frozen baseline immutability invariant strictly preserved throughout all loop executions.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
import pytest
import torch

from chakrview.brain.model import ChakrMicro
from chakrview.brain.adapter import NativeTaskAdapter
from chakrview.learning.episode import Attempt, LearningEpisode
from chakrview.learning.experience import ExperienceExtractor, ExperienceRecord
from chakrview.learning.loop import CognitiveLearningLoop
from chakrview.learning.promotion import PromotionGateController, PromotionDecision
from chakrview.runtime.interactive import (
    instantiate_frozen_baseline,
    compute_model_hash,
)
from chakrview.runtime.pipeline import EXPECTED_WEIGHT_HASH
from scripts.experiment_step57_learning_loop import (
    TASK_A_SPEC,
    TASK_A_PRIME_SPEC,
    TASK_B_SPEC,
)


@pytest.fixture(scope="module")
def baseline_model():
    return instantiate_frozen_baseline()


# ---------------------------------------------------------------------------
# Test 01: LearningEpisode creation and serialization
# ---------------------------------------------------------------------------
def test_01_learning_episode_serialization():
    ep = LearningEpisode(
        episode_id="ep_test_01",
        task_id="TASK_01",
        task_spec={"desc": "test task"},
        initial_context={"mode": "test"},
    )
    ep.add_stage("UNDERSTAND")
    ep.add_stage("ACT")
    ep.add_attempt(
        Attempt(
            attempt_id=1,
            state="INIT",
            action="def test(): return 1",
            observation="Passed",
            evaluation="PASS",
        )
    )
    ep.final_result = "SUCCESS"

    d = ep.to_dict()
    assert d["episode_id"] == "ep_test_01"
    assert len(d["attempts"]) == 1
    assert "UNDERSTAND" in d["executed_stages"]

    reconstructed = LearningEpisode.from_dict(d)
    assert reconstructed.episode_id == "ep_test_01"
    assert len(reconstructed.attempts) == 1
    assert reconstructed.attempts[0].action == "def test(): return 1"


# ---------------------------------------------------------------------------
# Test 02: ExperienceRecord extraction and verification gating
# ---------------------------------------------------------------------------
def test_02_experience_extraction_and_gating():
    # Failed episode -> verified is False
    ep_fail = LearningEpisode(
        episode_id="ep_fail",
        task_id="TASK_FAIL",
        task_spec={"category": "test"},
        initial_context={},
        final_result="FAILURE",
    )
    ep_fail.add_attempt(Attempt(1, "INIT", "x = 1", "syntax error", "FAIL"))
    exp_fail = ExperienceExtractor.extract_from_episode(ep_fail)
    assert exp_fail is not None
    assert exp_fail.verified is False

    # Successful episode -> verified is True
    ep_pass = LearningEpisode(
        episode_id="ep_pass",
        task_id="TASK_PASS",
        task_spec={"category": "test"},
        initial_context={},
        final_result="SUCCESS",
    )
    ep_pass.add_attempt(Attempt(1, "INIT", "def add(a, b): return a + b", "PASSED", "PASS"))
    exp_pass = ExperienceExtractor.extract_from_episode(ep_pass)
    assert exp_pass is not None
    assert exp_pass.verified is True


# ---------------------------------------------------------------------------
# Test 03: Store and retrieve experience
# ---------------------------------------------------------------------------
def test_03_store_and_retrieve_experience():
    loop = CognitiveLearningLoop()
    exp = ExperienceRecord(
        experience_id="exp_01",
        task_id="TASK_REPAIR",
        task_family="repair",
        task_description="Repair addition",
        context={},
        attempt_count=2,
        initial_action="return a - b",
        failure_observation="AssertionError",
        diagnosis="Wrong operator",
        correction="return a + b",
        verified_result="SUCCESS",
        what_worked="return a + b",
        what_failed="return a - b",
        reusable_pattern="Use '+' operator",
        verified=True,
        timestamp_utc="2026-09-30T00:00:00Z",
    )
    loop.store_experience(exp)

    retrieved_by_id = loop.retrieve_experience(task_id="TASK_REPAIR")
    assert retrieved_by_id is not None
    assert retrieved_by_id.reusable_pattern == "Use '+' operator"

    retrieved_by_family = loop.retrieve_experience(task_id="unknown", task_family="repair")
    assert retrieved_by_family is not None
    assert retrieved_by_family.task_id == "TASK_REPAIR"


# ---------------------------------------------------------------------------
# Test 04: CognitiveLearningLoop execution stages
# ---------------------------------------------------------------------------
def test_04_learning_loop_stages():
    loop = CognitiveLearningLoop(max_attempts=3)
    episode = loop.execute_episode(
        task_id=TASK_A_SPEC["task_id"],
        task_spec=TASK_A_SPEC,
        initial_action=TASK_A_SPEC["initial_defective_code"],
    )

    assert "UNDERSTAND" in episode.executed_stages
    assert "PLAN" in episode.executed_stages
    assert "ACT" in episode.executed_stages
    assert "OBSERVE" in episode.executed_stages
    assert "EVALUATE" in episode.executed_stages
    assert "DIAGNOSE" in episode.executed_stages
    assert "CORRECT" in episode.executed_stages
    assert "RETRY" in episode.executed_stages
    assert "VERIFY" in episode.executed_stages
    assert "REMEMBER" in episode.executed_stages
    assert "LEARN" in episode.executed_stages


# ---------------------------------------------------------------------------
# Test 05: Memory-conditioned repeat: Task A improvement (Delta_exp)
# ---------------------------------------------------------------------------
def test_05_memory_conditioned_repeat_improvement():
    loop = CognitiveLearningLoop(max_attempts=3)

    # Run 1: No prior experience -> Requires 2 attempts (fail -> fix -> pass)
    ep1 = loop.execute_episode(
        task_id=TASK_A_SPEC["task_id"],
        task_spec=TASK_A_SPEC,
        initial_action=TASK_A_SPEC["initial_defective_code"],
    )
    assert len(ep1.attempts) == 2
    assert ep1.final_result == "SUCCESS"

    stored_exp = loop.retrieve_experience(TASK_A_SPEC["task_id"])
    assert stored_exp is not None

    # Run 2: With stored experience -> Converges on Attempt 1
    ep2 = loop.execute_episode(
        task_id=TASK_A_SPEC["task_id"],
        task_spec=TASK_A_SPEC,
        initial_action=stored_exp.what_worked,
        retrieved_experience=stored_exp,
    )
    assert len(ep2.attempts) == 1
    assert ep2.final_result == "SUCCESS"

    delta_exp = len(ep1.attempts) - len(ep2.attempts)
    assert delta_exp == 1  # Exactly 1 attempt saved due to experience


# ---------------------------------------------------------------------------
# Test 06: Transfer to related Task A'
# ---------------------------------------------------------------------------
def test_06_transfer_to_related_task():
    loop = CognitiveLearningLoop(max_attempts=3)
    # Prime with Task A
    _ = loop.execute_episode(
        task_id=TASK_A_SPEC["task_id"],
        task_spec=TASK_A_SPEC,
        initial_action=TASK_A_SPEC["initial_defective_code"],
    )

    exp_family = loop.retrieve_experience("nonexistent", task_family="function_repair")
    assert exp_family is not None

    ep_prime = loop.execute_episode(
        task_id=TASK_A_PRIME_SPEC["task_id"],
        task_spec=TASK_A_PRIME_SPEC,
        initial_action=TASK_A_PRIME_SPEC["initial_defective_code"],
        retrieved_experience=exp_family,
    )
    assert ep_prime.final_result == "SUCCESS"


# ---------------------------------------------------------------------------
# Test 07: Unrelated control Task B
# ---------------------------------------------------------------------------
def test_07_unrelated_control_task_b():
    loop = CognitiveLearningLoop(max_attempts=3)
    ep_b = loop.execute_episode(
        task_id=TASK_B_SPEC["task_id"],
        task_spec=TASK_B_SPEC,
        initial_action=TASK_B_SPEC["initial_defective_code"],
    )
    assert ep_b.final_result == "SUCCESS"


# ---------------------------------------------------------------------------
# Test 08: PromotionGateController promotes valid candidate
# ---------------------------------------------------------------------------
def test_08_promotion_gate_promotes_valid(baseline_model):
    adapter = NativeTaskAdapter("valid_cand", rank=4)
    exp = ExperienceRecord(
        experience_id="exp_ok",
        task_id="T1",
        task_family="fam",
        task_description="desc",
        context={},
        attempt_count=1,
        initial_action="a",
        failure_observation=None,
        diagnosis=None,
        correction=None,
        verified_result="SUCCESS",
        what_worked="a",
        what_failed="none",
        reusable_pattern="pattern",
        verified=True,
        timestamp_utc="2026-09-30T00:00:00Z",
    )

    res = PromotionGateController.evaluate_candidate(
        base_model=baseline_model,
        experience=exp,
        candidate_adapter=adapter,
        anchor_pass_before=0.25,
        anchor_pass_after=0.25,
        target_pass_rate=1.0,
    )
    assert res["decision"] == PromotionDecision.PROMOTED


# ---------------------------------------------------------------------------
# Test 09: PromotionGateController rejects on regression
# ---------------------------------------------------------------------------
def test_09_promotion_gate_rejects_regression(baseline_model):
    adapter = NativeTaskAdapter("regress_cand", rank=4)
    exp = ExperienceRecord(
        experience_id="exp_ok",
        task_id="T1",
        task_family="fam",
        task_description="desc",
        context={},
        attempt_count=1,
        initial_action="a",
        failure_observation=None,
        diagnosis=None,
        correction=None,
        verified_result="SUCCESS",
        what_worked="a",
        what_failed="none",
        reusable_pattern="pattern",
        verified=True,
        timestamp_utc="2026-09-30T00:00:00Z",
    )

    # Anchor drops from 25% to 20% (drop = 5% > 2% max allowed)
    res = PromotionGateController.evaluate_candidate(
        base_model=baseline_model,
        experience=exp,
        candidate_adapter=adapter,
        anchor_pass_before=0.25,
        anchor_pass_after=0.20,
        target_pass_rate=1.0,
    )
    assert res["decision"] == PromotionDecision.REJECTED
    assert any("Anchor regression detected" in r for r in res["rejection_reasons"])


# ---------------------------------------------------------------------------
# Test 10: PromotionGateController rejects base mutation
# ---------------------------------------------------------------------------
def test_10_promotion_gate_rejects_base_mutation(baseline_model):
    adapter = NativeTaskAdapter("mut_cand", rank=4)
    exp = ExperienceRecord(
        experience_id="exp_ok",
        task_id="T1",
        task_family="fam",
        task_description="desc",
        context={},
        attempt_count=1,
        initial_action="a",
        failure_observation=None,
        diagnosis=None,
        correction=None,
        verified_result="SUCCESS",
        what_worked="a",
        what_failed="none",
        reusable_pattern="pattern",
        verified=True,
        timestamp_utc="2026-09-30T00:00:00Z",
    )

    # Base hash is tested against bogus hash
    res = PromotionGateController.evaluate_candidate(
        base_model=baseline_model,
        experience=exp,
        candidate_adapter=adapter,
        anchor_pass_before=0.25,
        anchor_pass_after=0.25,
        target_pass_rate=1.0,
        expected_base_hash="bogus_mutated_hash",
    )
    assert res["decision"] == PromotionDecision.REJECTED
    assert any("Base weight mutation detected" in r for r in res["rejection_reasons"])


# ---------------------------------------------------------------------------
# Test 11: Rollback test leaves base core 100% bit-exact
# ---------------------------------------------------------------------------
def test_11_rollback_leaves_base_core_intact(baseline_model):
    initial_hash = compute_model_hash(baseline_model)
    assert initial_hash == EXPECTED_WEIGHT_HASH

    adapter = NativeTaskAdapter("rollback_test", rank=4)
    adapter.mount(baseline_model)
    adapter.unmount(baseline_model)

    final_hash = compute_model_hash(baseline_model)
    assert final_hash == EXPECTED_WEIGHT_HASH


# ---------------------------------------------------------------------------
# Test 12: Frozen baseline immutability invariant
# ---------------------------------------------------------------------------
def test_12_frozen_baseline_immutability(baseline_model):
    cur_hash = compute_model_hash(baseline_model)
    assert cur_hash == EXPECTED_WEIGHT_HASH, "Frozen baseline hash mutated!"
