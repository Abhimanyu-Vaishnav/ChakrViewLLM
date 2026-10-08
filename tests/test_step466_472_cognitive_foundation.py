import os
import sys
import json
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from chakrview.runtime.interactive import EXPECTED_WEIGHT_HASH
from chakrview.cognition.release_benchmark import run_release_benchmark

def test_canonical_baseline_integrity():
    assert EXPECTED_WEIGHT_HASH == "c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da"

def test_wave466_co_training_artifacts():
    path = "artifacts/waves_466_472_execution/wave466_co_training.json"
    assert os.path.exists(path)
    with open(path, "r") as f:
        data = json.load(f)
    assert "Candidate_B" in data
    assert data["Candidate_B"]["overall_loss"] < 3.60

def test_wave467_probes():
    path = "artifacts/waves_466_472_execution/wave467_probes.json"
    assert os.path.exists(path)
    with open(path, "r") as f:
        data = json.load(f)
    assert "role_sensitivity" in data

def test_wave468_memory_coupling():
    path = "artifacts/waves_466_472_execution/wave468_memory_coupling.json"
    assert os.path.exists(path)
    with open(path, "r") as f:
        data = json.load(f)
    assert data["memory_retrieval_acc"] == 1.0
    assert data["unknown_rejection"] is True
    assert data["conflict_rejection"] is True

def test_wave469_compositional_curriculum():
    path = "artifacts/waves_466_472_execution/wave469_curriculum.json"
    assert os.path.exists(path)
    with open(path, "r") as f:
        data = json.load(f)
    assert data["memory_assisted_acc"] == 1.0

def test_wave470_interference_matrix():
    path = "artifacts/waves_466_472_execution/wave470_interference.json"
    assert os.path.exists(path)
    with open(path, "r") as f:
        data = json.load(f)
    assert data["analysis"]["capabilities_status"] == "MODULAR_COEXISTENCE_VERIFIED"

def test_wave471_adaptive_computation():
    path = "artifacts/waves_466_472_execution/wave471_adaptive.json"
    assert os.path.exists(path)
    with open(path, "r") as f:
        data = json.load(f)
    assert data["resource_bounded"] is True
    assert data["easy_cycles"] < data["very_hard_cycles"]

def test_wave472_master_gate():
    path = "artifacts/waves_466_472_execution/wave472_master_gate.json"
    assert os.path.exists(path)
    with open(path, "r") as f:
        data = json.load(f)
    assert data["final_decision"] == "HOLD \u2014 RELATIONAL COUPLING DEVELOPING"
    assert data["baseline"]["sha"] == "c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da"
