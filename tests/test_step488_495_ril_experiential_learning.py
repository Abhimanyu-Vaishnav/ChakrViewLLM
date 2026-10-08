import os
import sys
import json
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from chakrview.runtime.interactive import EXPECTED_WEIGHT_HASH
from chakrview.cognition.governed_learning.reasoning_experience import ReasoningExperience
from chakrview.cognition.wave488_495_ril_master_gate import run_ril_master_gate_evaluation

def test_canonical_baseline_integrity():
    assert EXPECTED_WEIGHT_HASH == "c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da"

def test_wave488_experience_schema():
    exp = ReasoningExperience.create(
        task_id="t1",
        task_family="2_hop",
        input_representation="A above B. B above C.",
        trajectory_summary=[{"cycle": 1}],
        intermediate_state_metadata={"s_1_norm": 0.3},
        reasoning_depth=1,
        outcome="C",
        expected_outcome="C",
        success=True,
        confidence=0.9,
        uncertainty=0.1,
        failure_category="NONE",
        resource_summary={"latency_ms": 1.2},
        model_version="v0.1",
        candidate_version="c1",
        evaluation_split="TRAINING_EXPERIENCE"
    )
    assert exp.experience_id is not None
    assert exp.success is True

def test_wave489_trajectory_capture():
    path = "artifacts/waves_488_495_execution/wave489_captured_experiences.json"
    assert os.path.exists(path)
    with open(path, "r") as f:
        data = json.load(f)
    assert len(data) == 20

def test_wave490_failure_attribution():
    path = "artifacts/waves_488_495_execution/wave490_failure_attribution.json"
    assert os.path.exists(path)
    with open(path, "r") as f:
        data = json.load(f)
    assert data["primary_failure_mode"] == "STATE_DEGRADATION"

def test_wave491_learning_signals():
    path = "artifacts/waves_488_495_execution/wave491_learning_signals.json"
    assert os.path.exists(path)
    with open(path, "r") as f:
        data = json.load(f)
    assert data["selected_signal"] == "ril_combined"

def test_wave492_candidate_update():
    path = "artifacts/waves_488_495_execution/wave492_candidate_update.json"
    assert os.path.exists(path)
    with open(path, "r") as f:
        data = json.load(f)
    assert data["ril_advantage_demonstrated"] is True

def test_wave493_anti_forgetting():
    path = "artifacts/waves_488_495_execution/wave493_anti_forgetting.json"
    assert os.path.exists(path)
    with open(path, "r") as f:
        data = json.load(f)
    assert data["verdict"] == "ZERO_REGRESSION_VERIFIED"

def test_wave494_repeated_experience():
    path = "artifacts/waves_488_495_execution/wave494_repeated_experience.json"
    assert os.path.exists(path)
    with open(path, "r") as f:
        data = json.load(f)
    assert data["net_unseen_improvement"] > 0.0

def test_wave495_master_gate():
    rep = run_ril_master_gate_evaluation()
    assert rep.final_decision == "RIL_ACHIEVED"
    assert rep.baseline_params == 3443136
    assert rep.baseline_delta_w == 0
