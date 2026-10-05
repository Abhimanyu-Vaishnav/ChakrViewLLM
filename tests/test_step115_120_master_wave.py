"""
Regression and automated tests for Steps 115-120 Master Development Wave:
- Step 115: Resource-Aware Worker Selection
- Step 116: Robust Transport & Node Abstraction
- Step 117: Distributed DAG Scheduler
- Step 118: Federation Fault Tolerance & Recovery
- Step 119: Extended Stage C Curriculum Integration
- Step 120: Unified Benchmark Integration
"""

import hashlib
import json
from pathlib import Path
import pytest

from chakrview.cognition.multi_agent.contracts import (
    WorkerRole,
    WorkerContract,
    WorkerPackage,
)
from chakrview.cognition.multi_agent.result import (
    WorkerExecutionStatus,
    WorkerResult,
)
from chakrview.cognition.multi_agent.resource_federation import (
    ResourceCapacityLevel,
    WorkerResourceProfile,
    TaskResourceRequirements,
    ResourceAwareWorkerSelector,
)
from chakrview.cognition.multi_agent.transport import (
    PROTOCOL_VERSION_V1,
    RequestEnvelope,
    ResponseEnvelope,
)
from chakrview.cognition.multi_agent.transport_channel import (
    SubprocessTransportChannel,
)
from chakrview.cognition.multi_agent.federation import (
    FederatedWorkerMetadata,
    FederatedScheduler,
)
from chakrview.cognition.multi_agent.fault_tolerance import (
    FederationFaultToleranceManager,
)
from chakrview.cognition.autonomous_benchmark import (
    build_benchmark_tool_gate,
    BenchmarkDiskProjectEngine,
)
from chakrview.runtime.interactive import (
    instantiate_frozen_baseline,
    compute_model_hash,
    EXPECTED_WEIGHT_HASH,
)


@pytest.fixture(scope="module")
def wave_fixture(tmp_path_factory):
    tmp_dir = tmp_path_factory.mktemp("step115_120_test_env")
    from scripts.setup_step113_fixture import setup_step113_repository
    repo_dir = setup_step113_repository(tmp_dir)
    db_path = tmp_dir / "test_wave_brain.db"
    tool_gate, _ = build_benchmark_tool_gate(repo_dir)
    scheduler = FederatedScheduler(db_path, repo_dir, tool_gate)

    workers = [
        FederatedWorkerMetadata("proc_analyst", WorkerRole.PROJECT_ANALYST, ["read_file"], timeout_seconds=10.0),
        FederatedWorkerMetadata("proc_impl", WorkerRole.IMPLEMENTER, ["write_file"], timeout_seconds=10.0),
        FederatedWorkerMetadata("proc_impl_fallback", WorkerRole.IMPLEMENTER, ["write_file"], timeout_seconds=10.0),
        FederatedWorkerMetadata("proc_test", WorkerRole.TEST_ENGINEER, ["write_file", "run_tests"], timeout_seconds=15.0),
    ]
    for w in workers:
        scheduler.register_federated_worker(w)

    return {
        "root": tmp_dir,
        "repo": repo_dir,
        "db": db_path,
        "gate": tool_gate,
        "scheduler": scheduler,
    }


def test_01_step115_resource_aware_selection():
    w_low = WorkerResourceProfile("w_low", WorkerRole.IMPLEMENTER, ResourceCapacityLevel.LOW_RESOURCE, memory_budget_mb=512)
    w_std = WorkerResourceProfile("w_std", WorkerRole.IMPLEMENTER, ResourceCapacityLevel.STANDARD, memory_budget_mb=1024)
    w_high = WorkerResourceProfile("w_high", WorkerRole.IMPLEMENTER, ResourceCapacityLevel.HIGH_RESOURCE, memory_budget_mb=4096)

    contract = WorkerContract("w_sel", WorkerRole.IMPLEMENTER, "t_req", ["write_file"], ["app/security.py"], 512)
    reqs = TaskResourceRequirements(minimum_capacity_level=ResourceCapacityLevel.STANDARD, minimum_memory_mb=1000)

    selected = ResourceAwareWorkerSelector.select_best_worker([w_low, w_std, w_high], contract, reqs)
    assert selected is not None
    assert selected.worker_id == "w_std"


def test_02_step116_transport_channel_contract(wave_fixture):
    repo_dir = wave_fixture["repo"]
    runtime_script = Path("chakrview/cognition/multi_agent/process_runtime.py")
    channel = SubprocessTransportChannel(runtime_script, repo_dir)
    assert channel.is_available() is True

    req = RequestEnvelope(
        protocol_version=PROTOCOL_VERSION_V1,
        worker_id="proc_analyst",
        task_id="t_chan",
        package_id="p_chan",
        role=WorkerRole.PROJECT_ANALYST.value,
        context_hash="hash1",
        context_token_count=50,
        allowed_tools=["read_file"],
        allowed_files=["app/models.py"],
        context_budget=512,
        package_payload={"task_id": "t_chan", "objective": "Read", "measured_context_tokens": 50},
    )
    resp = channel.send_request(req, timeout_seconds=10.0)
    assert resp.result_status == "SUCCESS"


def test_03_step118_fault_tolerance_manager(wave_fixture):
    sched = wave_fixture["scheduler"]
    fault_mgr = FederationFaultToleranceManager(sched)

    c = WorkerContract("proc_impl", WorkerRole.IMPLEMENTER, "t_fault_test", ["write_file"], ["app/security.py"], 512)
    p = WorkerPackage("t_fault_test", "Patch", dependency_outputs={"code_patch": {"app/security.py": "# test"}}, measured_context_tokens=60)

    res, audit = fault_mgr.execute_with_fault_tolerance(
        primary_worker_id="proc_impl",
        contract=c,
        package=p,
        fallback_worker_id="proc_impl_fallback",
        inject_crash=True,
    )
    assert res.status == WorkerExecutionStatus.SUCCESS
    assert res.worker_id == "proc_impl_fallback"
    assert audit.recovery_successful is True


def test_04_step119_curriculum_integration_and_baseline_immutability(wave_fixture):
    root = wave_fixture["root"]
    from chakrview.cognition.multi_agent.curriculum_integration import run_extended_curriculum_wave
    res = run_extended_curriculum_wave(root / "curriculum_out", steps=3)
    assert res["baseline_immutability_verified"] is True
    assert res["canonical_baseline_hash"] == EXPECTED_WEIGHT_HASH


def test_05_step120_canonical_baseline_exact():
    model = instantiate_frozen_baseline()
    h = compute_model_hash(model)
    assert h == EXPECTED_WEIGHT_HASH
