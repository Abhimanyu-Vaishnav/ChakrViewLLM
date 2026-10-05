"""
Unit and regression tests for Step 114 Distributed Multi-Node Worker Federation.

Verifies:
1. Normal process-isolated worker execution via OS subprocess
2. Transport envelope framing and protocol validation (hash checks)
3. Concurrent execution of independent worker processes
4. Worker capability mismatch rejection
5. Real OS process crash detection (nonzero returncode) and dynamic rerouting
6. Worker process timeout detection and graceful failure isolation
7. Malformed worker response envelope rejection
8. Duplicate task submission protection (Idempotency key)
9. Context isolation ceiling (<= 512 tokens) and payload isolation
10. GovernedToolGate authorization enforcement from isolated worker processes
11. PPB SQLite persistence & state reconstruction across fresh coordinators
12. Strict no-rescan disk behavior (0 rescanned on unchanged, 1 on delta)
13. Canonical neural baseline bit-exact immutability
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
from chakrview.cognition.multi_agent.transport import (
    PROTOCOL_VERSION_V1,
    RequestEnvelope,
    ResponseEnvelope,
)
from chakrview.cognition.multi_agent.federation import (
    FederatedWorkerMetadata,
    FederatedScheduler,
)
from chakrview.cognition.autonomous_benchmark import (
    BenchmarkDiskProjectEngine,
    build_benchmark_tool_gate,
)
from chakrview.runtime.interactive import (
    instantiate_frozen_baseline,
    compute_model_hash,
    EXPECTED_WEIGHT_HASH,
)


@pytest.fixture(scope="module")
def fed_fixture(tmp_path_factory):
    tmp_dir = tmp_path_factory.mktemp("step114_test_env")
    from scripts.setup_step113_fixture import setup_step113_repository
    repo_dir = setup_step113_repository(tmp_dir)
    db_path = tmp_dir / "test_fed_brain.db"
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


def test_01_transport_envelope_validation():
    # Valid envelope
    req = RequestEnvelope(
        protocol_version=PROTOCOL_VERSION_V1,
        worker_id="w1",
        task_id="t1",
        package_id="p1",
        role=WorkerRole.PROJECT_ANALYST.value,
        context_hash="hash1",
        context_token_count=100,
        allowed_tools=["read_file"],
        allowed_files=["app/models.py"],
        context_budget=512,
        package_payload={"k": "v"},
    )
    req.validate()

    # Incompatible protocol
    with pytest.raises(ValueError, match="Incompatible protocol_version"):
        req_bad = RequestEnvelope(
            protocol_version="9.9.9",
            worker_id="w1",
            task_id="t1",
            package_id="p1",
            role=WorkerRole.PROJECT_ANALYST.value,
            context_hash="hash1",
            context_token_count=100,
            allowed_tools=["read_file"],
            allowed_files=["app/models.py"],
            context_budget=512,
            package_payload={},
        )
        req_bad.validate()

    # Response hash mismatch
    res_bad = ResponseEnvelope(
        protocol_version=PROTOCOL_VERSION_V1,
        worker_id="w1",
        task_id="t1",
        package_id="p1",
        result_status="SUCCESS",
        result_payload={"a": 1},
        result_hash="fake_hash",
    )
    with pytest.raises(ValueError, match="Result hash mismatch"):
        res_bad.validate()


def test_02_normal_process_isolated_execution(fed_fixture):
    sched = fed_fixture["scheduler"]
    c = WorkerContract("proc_analyst", WorkerRole.PROJECT_ANALYST, "test_task_01", ["read_file"], ["app/models.py"], 512)
    p = WorkerPackage("test_task_01", "Inspect models", measured_context_tokens=80)
    res = sched.execute_in_isolated_process("proc_analyst", c, p)
    assert res.status == WorkerExecutionStatus.SUCCESS
    assert "app/models.py" in res.evidence


def test_03_concurrent_independent_workers(fed_fixture):
    sched = fed_fixture["scheduler"]
    c1 = WorkerContract("proc_analyst", WorkerRole.PROJECT_ANALYST, "par_t1", ["read_file"], ["app/models.py"], 512)
    p1 = WorkerPackage("par_t1", "Inspect 1", measured_context_tokens=50)
    c2 = WorkerContract("proc_analyst", WorkerRole.PROJECT_ANALYST, "par_t2", ["read_file"], ["app/validation.py"], 512)
    p2 = WorkerPackage("par_t2", "Inspect 2", measured_context_tokens=50)

    results = sched.execute_concurrent_tasks([
        ("proc_analyst", c1, p1),
        ("proc_analyst", c2, p2),
    ])
    assert len(results) == 2
    assert all(r.status == WorkerExecutionStatus.SUCCESS for r in results)


def test_04_capability_mismatch_rejected(fed_fixture):
    sched = fed_fixture["scheduler"]
    c = WorkerContract("proc_analyst", WorkerRole.IMPLEMENTER, "t_mismatch", ["write_file"], ["app/security.py"], 512)
    p = WorkerPackage("t_mismatch", "Mismatch", measured_context_tokens=50)
    res = sched.execute_in_isolated_process("proc_analyst", c, p)
    assert res.status == WorkerExecutionStatus.FAILED
    assert "capability mismatch" in (res.error or "").lower()


def test_05_process_crash_and_dynamic_rerouting(fed_fixture):
    sched = fed_fixture["scheduler"]
    c = WorkerContract("proc_impl", WorkerRole.IMPLEMENTER, "t_crash", ["write_file"], ["app/security.py"], 512)
    p = WorkerPackage("t_crash", "Reroute patch", dependency_outputs={"code_patch": {"app/security.py": "# patched"}}, measured_context_tokens=60)

    # Injects process crash with returncode 139 on primary
    res, was_rerouted = sched.execute_with_rerouting(
        primary_worker_id="proc_impl",
        contract=c,
        package=p,
        fallback_worker_id="proc_impl_fallback",
        inject_crash_on_primary=True,
    )
    assert was_rerouted is True
    assert res.status == WorkerExecutionStatus.SUCCESS
    assert res.worker_id == "proc_impl_fallback"


def test_06_worker_timeout_detection(fed_fixture):
    sched = fed_fixture["scheduler"]
    c = WorkerContract("proc_analyst", WorkerRole.PROJECT_ANALYST, "t_tout", ["read_file"], ["app/models.py"], 512)
    p = WorkerPackage("t_tout", "Timeout", measured_context_tokens=50)
    # Impose microscopic timeout
    res = sched.execute_in_isolated_process("proc_analyst", c, p, timeout=0.0001)
    assert res.status == WorkerExecutionStatus.FAILED
    assert "timed out" in (res.error or "").lower()


def test_07_duplicate_submission_protection(fed_fixture):
    sched = fed_fixture["scheduler"]
    c = WorkerContract("proc_analyst", WorkerRole.PROJECT_ANALYST, "test_task_01", ["read_file"], ["app/models.py"], 512)
    p = WorkerPackage("test_task_01", "Dup", measured_context_tokens=50)
    # test_task_01 was completed in test_02
    res = sched.execute_in_isolated_process("proc_analyst", c, p)
    assert res.status == WorkerExecutionStatus.FAILED
    assert "duplicate" in (res.error or "").lower()


def test_08_context_ceiling_and_tool_governance(fed_fixture):
    sched = fed_fixture["scheduler"]

    # Context overflow
    c_over = WorkerContract("proc_analyst", WorkerRole.PROJECT_ANALYST, "t_over", ["read_file"], ["app/models.py"], 200)
    p_over = WorkerPackage("t_over", "Over", measured_context_tokens=350)
    res_over = sched.execute_in_isolated_process("proc_analyst", c_over, p_over)
    assert res_over.status == WorkerExecutionStatus.FAILED
    assert "context ceiling exceeded" in (res_over.error or "").lower()

    # Tool governance: path traversal from isolated process
    c_sec = WorkerContract("proc_analyst", WorkerRole.PROJECT_ANALYST, "t_sec", ["read_file"], ["../../passwords.txt"], 512)
    p_sec = WorkerPackage("t_sec", "Sec", measured_context_tokens=50)
    res_sec = sched.execute_in_isolated_process("proc_analyst", c_sec, p_sec)
    assert res_sec.status == WorkerExecutionStatus.FAILED
    assert "security violation" in (res_sec.error or "").lower()


def test_09_persistence_and_no_rescan(fed_fixture):
    db_path = fed_fixture["db"]
    repo_dir = fed_fixture["repo"]

    # Reconstruct from fresh scheduler
    tool_gate = fed_fixture["gate"]
    fresh_sched = FederatedScheduler(db_path, repo_dir, tool_gate)
    persisted = fresh_sched.get_persisted_result("test_task_01")
    assert persisted is not None
    assert persisted["status"] == WorkerExecutionStatus.SUCCESS.value

    # Strict no-rescan
    engine = BenchmarkDiskProjectEngine(db_path, repo_dir)
    s1 = engine.scan_repository()
    s2 = engine.scan_repository()
    assert s2["files_scanned"] == 0
    assert s2["files_skipped_unchanged"] == s2["total_files"]

    (repo_dir / "app" / "models.py").write_text("# modified\n", encoding="utf-8")
    s3 = engine.scan_repository()
    assert s3["files_scanned"] == 1


def test_10_canonical_baseline_bit_exact():
    model = instantiate_frozen_baseline()
    h = compute_model_hash(model)
    assert h == EXPECTED_WEIGHT_HASH
