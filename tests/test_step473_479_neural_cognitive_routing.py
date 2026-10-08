import os
import sys
import json
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from chakrview.runtime.interactive import EXPECTED_WEIGHT_HASH
from chakrview.cognition.neural_routing_gateway import NeuralCognitiveRoutingGateway
from chakrview.cognition.wave473_479_master_gate import run_master_gate_evaluation

def test_canonical_baseline_intact():
    assert EXPECTED_WEIGHT_HASH == "c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da"

def test_gateway_parameter_budget():
    gw = NeuralCognitiveRoutingGateway()
    assert gw.total_param_count == 74016

def test_wave474_gateway_acquisition_artifacts():
    path = "artifacts/waves_473_479_execution/wave474_gateway_acquisition.json"
    assert os.path.exists(path)
    with open(path, "r") as f:
        data = json.load(f)
    assert data["trainable"] is True
    assert data["final_loss"] < 0.1

def test_wave475_causal_validation():
    path = "artifacts/waves_473_479_execution/wave475_causal_validation.json"
    assert os.path.exists(path)
    with open(path, "r") as f:
        data = json.load(f)
    assert data["causal_effect_demonstrated"] is True
    assert data["anti_shortcut_passed"] is True

def test_wave476_closed_pipeline():
    path = "artifacts/waves_473_479_execution/wave476_closed_pipeline.json"
    assert os.path.exists(path)
    with open(path, "r") as f:
        data = json.load(f)
    assert data["system_C_gateway_cognition_acc"] == 1.0

def test_wave477_verbalization():
    path = "artifacts/waves_473_479_execution/wave477_verbalization.json"
    assert os.path.exists(path)
    with open(path, "r") as f:
        data = json.load(f)
    assert data["gateway_reverse_transfer_functional"] is True

def test_wave478_bidirectional_closed_loop():
    path = "artifacts/waves_473_479_execution/wave478_bidirectional_closed_loop.json"
    assert os.path.exists(path)
    with open(path, "r") as f:
        data = json.load(f)
    assert data["bidirectional_gw_mean"] == 1.0

def test_wave479_master_gate():
    rep = run_master_gate_evaluation()
    assert rep.baseline_params == 3443136
    assert rep.baseline_delta_w == 0
    assert rep.final_decision == "NEURAL_COUPLING_EMERGING"
