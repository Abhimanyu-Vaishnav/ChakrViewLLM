import os
import sys
import json
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from chakrview.runtime.interactive import EXPECTED_WEIGHT_HASH
from chakrview.runtime.unified_cognitive_runtime import UnifiedCognitiveRuntime
from chakrview.cognition.v020_unified_runtime_master_gate import run_v020_master_gate

def test_canonical_baseline_integrity():
    assert EXPECTED_WEIGHT_HASH == "c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da"

def test_wave504_runtime_instantiation():
    runtime = UnifiedCognitiveRuntime()
    assert runtime.runtime_version == "v0.2.0_unified"
    assert len(runtime.memory) > 0

def test_wave505_task_lifecycle():
    path = "artifacts/waves_504_511_execution/wave505_task_lifecycle.json"
    assert os.path.exists(path)
    with open(path, "r") as f:
        data = json.load(f)
    assert len(data) == 12

def test_wave506_failure_recovery():
    path = "artifacts/waves_504_511_execution/wave506_failure_recovery.json"
    assert os.path.exists(path)
    with open(path, "r") as f:
        data = json.load(f)
    assert data["node_failure_does_not_cause_task_failure"] is True

def test_wave507_unified_governance():
    path = "artifacts/waves_504_511_execution/wave507_unified_governance.json"
    assert os.path.exists(path)
    with open(path, "r") as f:
        data = json.load(f)
    assert data["resource_permission_neq_requirement"] is True
    assert data["self_authorization_prevented"] is True

def test_wave508_long_horizon():
    path = "artifacts/waves_504_511_execution/wave508_long_horizon.json"
    assert os.path.exists(path)
    with open(path, "r") as f:
        data = json.load(f)
    assert data["overall_accuracy"] >= 0.90
    assert data["resource_leak_detected"] is False

def test_wave509_distributed_recovery():
    path = "artifacts/waves_504_511_execution/wave509_distributed_recovery.json"
    assert os.path.exists(path)
    with open(path, "r") as f:
        data = json.load(f)
    assert data["cognitive_state_recoverable"] is True
    assert data["task_restart_avoided"] is True

def test_wave510_benchmark_provenance():
    path = "artifacts/waves_504_511_execution/wave510_full_system_benchmark.json"
    assert os.path.exists(path)
    with open(path, "r") as f:
        data = json.load(f)
    assert "I5_4_HOP" in data
    assert data["I5_4_HOP"]["status"] == "VERIFIED"

def test_wave511_master_gate():
    rep = run_v020_master_gate()
    assert rep.final_decision == "V020_RELEASE_CANDIDATE_VERIFIED"
    assert rep.baseline_params == 3443136
    assert rep.baseline_delta_w == 0
