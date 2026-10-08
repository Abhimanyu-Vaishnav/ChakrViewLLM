import os
import sys
import json
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from chakrview.runtime.interactive import EXPECTED_WEIGHT_HASH
from chakrview.cognition.metacognitive_evaluator import MetaCognitiveEvaluator, ReasoningStrategy
from chakrview.cognition.wave496_503_meta_distributed_master_gate import run_meta_distributed_master_gate

def test_canonical_baseline_integrity():
    assert EXPECTED_WEIGHT_HASH == "c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da"

def test_wave496_metacognitive_eval():
    path = "artifacts/waves_496_503_execution/wave496_metacognitive_eval.json"
    assert os.path.exists(path)
    with open(path, "r") as f:
        data = json.load(f)
    assert data["discrimination_verified"] is True
    assert data["discrimination_separation"] > 0.40

def test_wave497_strategy_selection():
    path = "artifacts/waves_496_503_execution/wave497_strategy_selection.json"
    assert os.path.exists(path)
    with open(path, "r") as f:
        data = json.load(f)
    assert data["strategy_selection_verified"] is True
    assert data["accuracy_gain_over_fixed"] > 10.0

def test_wave498_counterfactual_eval():
    path = "artifacts/waves_496_503_execution/wave498_counterfactual_eval.json"
    assert os.path.exists(path)
    with open(path, "r") as f:
        data = json.load(f)
    assert data["counterfactual_reasoning_verified"] is True

def test_wave499_self_correction():
    path = "artifacts/waves_496_503_execution/wave499_self_correction.json"
    assert os.path.exists(path)
    with open(path, "r") as f:
        data = json.load(f)
    assert data["self_correction_verified"] is True
    assert data["recovery_rate"] >= 0.70

def test_wave500_distributed_peering():
    path = "artifacts/waves_496_503_execution/wave500_distributed_peering.json"
    assert os.path.exists(path)
    with open(path, "r") as f:
        data = json.load(f)
    assert data["peer_failure_does_not_cause_task_failure"] is True
    assert data["majority_vote_bypassed"] is True

def test_wave501_peer_disagreement():
    path = "artifacts/waves_496_503_execution/wave501_peer_disagreement.json"
    assert os.path.exists(path)
    with open(path, "r") as f:
        data = json.load(f)
    assert data["majority_voting_forbidden"] is True
    assert data["epistemic_verification_enforced"] is True

def test_wave502_meta_ril_integration():
    path = "artifacts/waves_496_503_execution/wave502_meta_ril_integration.json"
    assert os.path.exists(path)
    with open(path, "r") as f:
        data = json.load(f)
    assert data["unverified_peer_learning_blocked"] is True

def test_wave503_master_gate():
    rep = run_meta_distributed_master_gate()
    assert rep.baseline_params == 3443136
    assert rep.baseline_delta_w == 0
    assert rep.final_decision == "VERIFIED_UNDER_DEFINED_BENCHMARK"
