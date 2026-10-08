import os
import sys
import json
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from chakrview.runtime.interactive import EXPECTED_WEIGHT_HASH
from chakrview.cognition.wave480_487_i5_master_gate import run_i5_master_gate_evaluation

def test_canonical_baseline_integrity():
    assert EXPECTED_WEIGHT_HASH == "c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da"

def test_wave480_manifest():
    path = "artifacts/waves_480_487_execution/i5_benchmark_manifest.json"
    assert os.path.exists(path)
    with open(path, "r") as f:
        data = json.load(f)
    assert len(data["1_hop"]) > 0
    assert len(data["4_hop"]) > 0

def test_wave481_two_hop():
    path = "artifacts/waves_480_487_execution/wave481_two_hop.json"
    assert os.path.exists(path)
    with open(path, "r") as f:
        data = json.load(f)
    assert data["causal_state_dependence_verified"] is True

def test_wave482_three_hop():
    path = "artifacts/waves_480_487_execution/wave482_three_hop.json"
    assert os.path.exists(path)
    with open(path, "r") as f:
        data = json.load(f)
    assert data["g3_acc"] >= 0.85

def test_wave483_four_hop():
    path = "artifacts/waves_480_487_execution/wave483_four_hop.json"
    assert os.path.exists(path)
    with open(path, "r") as f:
        data = json.load(f)
    assert data["meaningful_4hop_performance"] is True
    assert data["accuracy_curve"]["hop_4"] >= 0.70

def test_wave484_adversarial_stress():
    path = "artifacts/waves_480_487_execution/wave484_adversarial_stress.json"
    assert os.path.exists(path)
    with open(path, "r") as f:
        data = json.load(f)
    assert data["anti_shortcut_verified"] is True
    assert data["adversarial_pass_rate"] == 1.0

def test_wave485_cross_language_i5():
    path = "artifacts/waves_480_487_execution/wave485_cross_language_i5.json"
    assert os.path.exists(path)
    with open(path, "r") as f:
        data = json.load(f)
    assert data["means"]["gateway_cognition"] >= 0.90

def test_wave486_adaptive_resources():
    path = "artifacts/waves_480_487_execution/wave486_adaptive_resources.json"
    assert os.path.exists(path)
    with open(path, "r") as f:
        data = json.load(f)
    assert data["monotonic_cycle_adaptation"] is True

def test_wave487_master_gate():
    rep = run_i5_master_gate_evaluation()
    assert rep.final_decision == "I5_ACHIEVED"
    assert rep.baseline_params == 3443136
    assert rep.baseline_delta_w == 0
