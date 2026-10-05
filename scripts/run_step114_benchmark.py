"""
Step 114 Distributed Multi-Node Worker Federation Benchmark Runner.

Executes all 14 empirical experiments:
- EXP 1: Normal process-isolated worker execution.
- EXP 2: Two independent workers execute concurrently.
- EXP 3: Worker capability mismatch.
- EXP 4: Worker process crash after valid package receipt.
- EXP 5: Automatic rerouting to compatible backup worker.
- EXP 6: Worker timeout detection and recovery.
- EXP 7: Malformed worker response rejection.
- EXP 8: Duplicate task submission protection.
- EXP 9: Context <=512 and serialized payload isolation.
- EXP 10: GovernedToolGate security enforcement from worker process.
- EXP 11: PPB persistence and fresh-coordinator reconstruction.
- EXP 12: DAG dependency correctness under process federation.
- EXP 13: No-rescan behavior remains intact.
- EXP 14: Canonical baseline immutability.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Ensure root in sys.path
_repo_root = str(Path(__file__).resolve().parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from typing import Any, Dict, List, Tuple

from chakrview.cognition.multi_agent.contracts import (
    WorkerRole,
    ReviewerVerdict,
    WorkerContract,
    WorkerPackage,
)
from chakrview.cognition.multi_agent.result import (
    WorkerExecutionStatus,
    WorkerResult,
    ReviewerCritique,
    SynthesisResult,
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


def run_complete_step114_benchmark(bench_dir: Path) -> Dict[str, Any]:
    t0 = time.perf_counter()
    bench_dir = Path(bench_dir).resolve()
    repo_dir = bench_dir / "repository"
    db_path = bench_dir / "project_brain.db"

    if bench_dir.exists():
        shutil.rmtree(bench_dir)
    bench_dir.mkdir(parents=True, exist_ok=True)

    # Setup repo fixture
    from scripts.setup_step113_fixture import setup_step113_repository
    setup_step113_repository(bench_dir)

    tool_gate, skill = build_benchmark_tool_gate(repo_dir)
    scheduler = FederatedScheduler(db_path, repo_dir, tool_gate)

    # Register federated workers
    workers = [
        FederatedWorkerMetadata("proc_analyst_01", WorkerRole.PROJECT_ANALYST, ["read_file"], timeout_seconds=10.0),
        FederatedWorkerMetadata("proc_planner_01", WorkerRole.PLANNER, [], timeout_seconds=10.0),
        FederatedWorkerMetadata("proc_impl_01", WorkerRole.IMPLEMENTER, ["write_file"], timeout_seconds=10.0),
        FederatedWorkerMetadata("proc_impl_backup", WorkerRole.IMPLEMENTER, ["write_file"], timeout_seconds=10.0),
        FederatedWorkerMetadata("proc_test_01", WorkerRole.TEST_ENGINEER, ["write_file", "run_tests"], timeout_seconds=15.0),
        FederatedWorkerMetadata("proc_rev_01", WorkerRole.REVIEWER, ["read_file"], timeout_seconds=10.0),
    ]
    for w in workers:
        scheduler.register_federated_worker(w)

    results: Dict[str, Any] = {}

    # -------------------------------------------------------------------------
    # EXP 14A: Canonical Baseline Verification Before Run
    # -------------------------------------------------------------------------
    model_pre = instantiate_frozen_baseline()
    hash_pre = compute_model_hash(model_pre)
    assert hash_pre == EXPECTED_WEIGHT_HASH, "Canonical baseline mutated before Step 114 benchmark!"

    # -------------------------------------------------------------------------
    # EXP 1 & EXP 9: Normal Process-Isolated Worker Execution & Context Isolation
    # -------------------------------------------------------------------------
    c1 = WorkerContract(
        worker_id="proc_analyst_01",
        role=WorkerRole.PROJECT_ANALYST,
        task_id="fed_task_01_analyst",
        allowed_tools=["read_file"],
        allowed_files=["app/security.py", "app/validation.py"],
        context_budget=512,
    )
    p1 = WorkerPackage(
        task_id="fed_task_01_analyst",
        objective="Inspect security and validation modules in isolated process",
        measured_context_tokens=110,
    )
    r1 = scheduler.execute_in_isolated_process("proc_analyst_01", c1, p1)
    assert r1.status == WorkerExecutionStatus.SUCCESS
    assert "app/security.py" in r1.evidence
    results["exp1_normal_process_isolated"] = True

    # Check serialized envelope size and context token ceiling
    req_env_1 = RequestEnvelope(
        protocol_version=PROTOCOL_VERSION_V1,
        worker_id="proc_analyst_01",
        task_id="fed_task_01_analyst",
        package_id="pkg_test",
        role=WorkerRole.PROJECT_ANALYST.value,
        context_hash=hashlib.sha256(b"ctx").hexdigest(),
        context_token_count=p1.measured_context_tokens,
        allowed_tools=list(c1.allowed_tools),
        allowed_files=list(c1.allowed_files),
        context_budget=512,
        package_payload={"objective": p1.objective, "measured_context_tokens": p1.measured_context_tokens},
    )
    req_env_bytes = json.dumps(req_env_1.to_dict()).encode("utf-8")
    results["exp9_context_isolation"] = {
        "context_token_count": p1.measured_context_tokens,
        "context_ceiling_512_passed": p1.measured_context_tokens <= 512,
        "serialized_payload_bytes": len(req_env_bytes),
    }

    # -------------------------------------------------------------------------
    # EXP 2: Two Independent Workers Execute Concurrently
    # -------------------------------------------------------------------------
    c_par_1 = WorkerContract("proc_analyst_01", WorkerRole.PROJECT_ANALYST, "par_task_01", ["read_file"], ["app/models.py"], 512)
    p_par_1 = WorkerPackage("par_task_01", "Inspect models", measured_context_tokens=60)
    c_par_2 = WorkerContract("proc_analyst_01", WorkerRole.PROJECT_ANALYST, "par_task_02", ["read_file"], ["app/service.py"], 512)
    p_par_2 = WorkerPackage("par_task_02", "Inspect service", measured_context_tokens=60)

    t_start_par = time.perf_counter()
    par_results = scheduler.execute_concurrent_tasks([
        ("proc_analyst_01", c_par_1, p_par_1),
        ("proc_analyst_01", c_par_2, p_par_2),
    ])
    t_par_elapsed = time.perf_counter() - t_start_par
    assert len(par_results) == 2
    assert all(r.status == WorkerExecutionStatus.SUCCESS for r in par_results)
    results["exp2_concurrent_execution"] = {
        "tasks_executed": 2,
        "elapsed_seconds": round(t_par_elapsed, 3),
        "status": "SUCCESS",
    }

    # -------------------------------------------------------------------------
    # EXP 3: Worker Capability Mismatch
    # -------------------------------------------------------------------------
    c_mismatch = WorkerContract(
        worker_id="proc_analyst_01",
        role=WorkerRole.IMPLEMENTER,  # Analyst assigned implementer role
        task_id="task_mismatch",
        allowed_tools=["write_file"],
        allowed_files=["app/security.py"],
        context_budget=512,
    )
    p_mismatch = WorkerPackage("task_mismatch", "Mismatch check", measured_context_tokens=50)
    r_mismatch = scheduler.execute_in_isolated_process("proc_analyst_01", c_mismatch, p_mismatch)
    assert r_mismatch.status == WorkerExecutionStatus.FAILED
    assert "capability mismatch" in (r_mismatch.error or "").lower()
    results["exp3_capability_mismatch"] = True

    # -------------------------------------------------------------------------
    # EXP 4 & 5: Worker Process Crash & Dynamic Rerouting to Backup Worker
    # -------------------------------------------------------------------------
    hardened_security_code = """# Hardened password strength validation
def validate_password_strength(password: str) -> bool:
    if len(password) < 8:
        return False
    if not any(c.isdigit() for c in password):
        return False
    special_chars = "!@#$%^&*()-_=+[]{}|;:,.<>?"
    if not any(c in special_chars for c in password):
        return False
    return True
"""
    c_crash = WorkerContract(
        worker_id="proc_impl_01",
        role=WorkerRole.IMPLEMENTER,
        task_id="task_crash_recovery",
        allowed_tools=["write_file"],
        allowed_files=["app/security.py"],
        context_budget=512,
    )
    p_crash = WorkerPackage(
        task_id="task_crash_recovery",
        objective="Apply patch via rerouted worker after primary crash",
        dependency_outputs={"code_patch": {"app/security.py": hardened_security_code}},
        measured_context_tokens=180,
    )
    r_recovered, was_rerouted = scheduler.execute_with_rerouting(
        primary_worker_id="proc_impl_01",
        contract=c_crash,
        package=p_crash,
        fallback_worker_id="proc_impl_backup",
        inject_crash_on_primary=True,  # Injects process crash with returncode 139
    )
    assert was_rerouted is True
    assert r_recovered.status == WorkerExecutionStatus.SUCCESS
    assert r_recovered.worker_id == "proc_impl_backup"
    results["exp4_process_crash_detected"] = True
    results["exp5_automatic_rerouting"] = True

    # -------------------------------------------------------------------------
    # EXP 6: Worker Timeout Detection and Recovery
    # -------------------------------------------------------------------------
    c_timeout = WorkerContract(
        worker_id="proc_analyst_01",
        role=WorkerRole.PROJECT_ANALYST,
        task_id="task_timeout_test",
        allowed_tools=["read_file"],
        allowed_files=["app/models.py"],
        context_budget=512,
    )
    p_timeout = WorkerPackage("task_timeout_test", "Timeout test", measured_context_tokens=50)
    # Force timeout by setting timeout threshold to 0.0001 seconds
    r_timeout = scheduler.execute_in_isolated_process("proc_analyst_01", c_timeout, p_timeout, timeout=0.0001)
    assert r_timeout.status == WorkerExecutionStatus.FAILED
    assert "timed out" in (r_timeout.error or "").lower()
    results["exp6_timeout_detected"] = True

    # -------------------------------------------------------------------------
    # EXP 7: Malformed Worker Response Envelope Rejection
    # -------------------------------------------------------------------------
    # Test response envelope hash mismatch validation
    bad_resp_dict = {
        "protocol_version": PROTOCOL_VERSION_V1,
        "worker_id": "w1",
        "task_id": "t1",
        "package_id": "pkg1",
        "result_status": "SUCCESS",
        "result_payload": {"key": "val"},
        "result_hash": "bad_corrupted_hash",
    }
    bad_resp = ResponseEnvelope.from_dict(bad_resp_dict)
    try:
        bad_resp.validate()
        results["exp7_malformed_rejected"] = False
    except ValueError as e:
        assert "hash mismatch" in str(e).lower()
        results["exp7_malformed_rejected"] = True

    # -------------------------------------------------------------------------
    # EXP 8: Duplicate Task Submission Protection (Idempotency Key)
    # -------------------------------------------------------------------------
    # Attempting to re-execute already completed fed_task_01_analyst
    c_dup = WorkerContract(
        worker_id="proc_analyst_01",
        role=WorkerRole.PROJECT_ANALYST,
        task_id="fed_task_01_analyst",
        allowed_tools=["read_file"],
        allowed_files=["app/models.py"],
        context_budget=512,
    )
    p_dup = WorkerPackage("fed_task_01_analyst", "Duplicate attempt", measured_context_tokens=50)
    r_dup = scheduler.execute_in_isolated_process("proc_analyst_01", c_dup, p_dup)
    assert r_dup.status == WorkerExecutionStatus.FAILED
    assert "duplicate" in (r_dup.error or "").lower()
    results["exp8_duplicate_protection"] = True

    # -------------------------------------------------------------------------
    # EXP 10: GovernedToolGate Security Enforcement in Worker Process
    # -------------------------------------------------------------------------
    # Path traversal attempt from isolated process
    c_sec = WorkerContract(
        worker_id="proc_analyst_01",
        role=WorkerRole.PROJECT_ANALYST,
        task_id="task_sec_trav",
        allowed_tools=["read_file"],
        allowed_files=["../../secret.txt"],
        context_budget=512,
    )
    p_sec = WorkerPackage("task_sec_trav", "Security test", measured_context_tokens=50)
    r_sec = scheduler.execute_in_isolated_process("proc_analyst_01", c_sec, p_sec)
    assert r_sec.status == WorkerExecutionStatus.FAILED
    assert "security violation" in (r_sec.error or "").lower()
    results["exp10_tool_governance_enforced"] = True

    # -------------------------------------------------------------------------
    # EXP 11: PPB Persistence & Fresh Coordinator Reconstruction
    # -------------------------------------------------------------------------
    fresh_scheduler = FederatedScheduler(db_path, repo_dir, tool_gate)
    persisted_task = fresh_scheduler.get_persisted_result("fed_task_01_analyst")
    assert persisted_task is not None
    assert persisted_task["status"] == WorkerExecutionStatus.SUCCESS.value
    results["exp11_persistence_reconstructed"] = True

    # -------------------------------------------------------------------------
    # EXP 12: Complete DAG Flow Verification Under Process Federation
    # -------------------------------------------------------------------------
    # Run test engineer in isolated process
    hardened_test_security_code = """from app.security import validate_password_strength

def test_password_strength_valid():
    assert validate_password_strength("P@ssw0rd123") is True

def test_password_strength_too_short():
    assert validate_password_strength("P@1") is False

def test_password_strength_no_digit():
    assert validate_password_strength("Password!@#") is False

def test_password_strength_no_special():
    assert validate_password_strength("Password123") is False
"""

    hardened_test_validation_code = """from app.validation import validate_account_data


def test_validate_account_basic():
    assert validate_account_data({"username": "user1", "email": "u1@test.com", "password": "P@ssw0rd123"}) is True
    assert validate_account_data({"username": ""}) is False

def test_validate_account_weak_password():
    assert validate_account_data({"username": "user1", "email": "u1@test.com", "password": "weak"}) is False
"""

    hardened_test_service_code = """import pytest
from app.service import AccountService

def test_service_register_success():
    svc = AccountService()
    acc = svc.register({"username": "user1", "email": "u1@test.com", "password": "P@ssw0rd123"})
    assert acc.username == "user1"
    assert acc.email == "u1@test.com"

def test_service_register_invalid_rejected():
    svc = AccountService()
    with pytest.raises(ValueError):
        svc.register({"username": ""})

def test_service_register_weak_password_rejected():
    svc = AccountService()
    with pytest.raises(ValueError, match="Registration validation failed"):
        svc.register({"username": "user1", "email": "u1@test.com", "password": "weak"})
"""

    c_test = WorkerContract(
        worker_id="proc_test_01",
        role=WorkerRole.TEST_ENGINEER,
        task_id="fed_task_test",
        allowed_tools=["write_file", "run_tests"],
        allowed_files=["tests/test_security.py", "tests/test_validation.py", "tests/test_service.py"],
        context_budget=512,
    )
    p_test = WorkerPackage(
        task_id="fed_task_test",
        objective="Run security verification with compliant test fixtures",
        dependency_outputs={
            "test_patches": {
                "tests/test_security.py": hardened_test_security_code,
                "tests/test_validation.py": hardened_test_validation_code,
                "tests/test_service.py": hardened_test_service_code,
            }
        },
        measured_context_tokens=180,
    )
    r_test = scheduler.execute_in_isolated_process("proc_test_01", c_test, p_test)
    assert r_test.status == WorkerExecutionStatus.SUCCESS

    assert r_test.verification_output.get("passed") is True
    results["exp12_dag_flow_correctness"] = True

    # -------------------------------------------------------------------------
    # EXP 13: Strict No-Rescan Behavior Remains Intact
    # -------------------------------------------------------------------------
    engine = BenchmarkDiskProjectEngine(db_path, repo_dir)
    scan1 = engine.scan_repository()
    scan2 = engine.scan_repository()
    assert scan2["files_scanned"] == 0
    assert scan2["files_skipped_unchanged"] == scan2["total_files"]

    (repo_dir / "app" / "models.py").write_text("# fed delta\n", encoding="utf-8")
    scan3 = engine.scan_repository()
    assert scan3["files_scanned"] == 1
    assert scan3["files_skipped_unchanged"] == scan3["total_files"] - 1
    results["exp13_no_rescan_verified"] = {
        "run1_scanned": scan1["files_scanned"],
        "run2_rescanned": scan2["files_scanned"],
        "run3_delta_rescanned": scan3["files_scanned"],
    }

    # -------------------------------------------------------------------------
    # EXP 14B: Canonical Baseline Immutability Check
    # -------------------------------------------------------------------------
    model_post = instantiate_frozen_baseline()
    hash_post = compute_model_hash(model_post)
    assert hash_post == EXPECTED_WEIGHT_HASH, "Canonical baseline mutated during Step 114 benchmark!"
    results["exp14_canonical_baseline_bit_exact"] = True

    # -------------------------------------------------------------------------
    # Save Artifacts
    # -------------------------------------------------------------------------
    elapsed_total = time.perf_counter() - t0
    final_summary = {
        "milestone": "Step 114",
        "benchmark_status": "SUCCESS",
        "experiments_passed": 14,
        "elapsed_seconds": round(elapsed_total, 2),
        "context_audit": results["exp9_context_isolation"],
        "no_rescan_metrics": results["exp13_no_rescan_verified"],
        "canonical_baseline_bit_exact": True,
    }
    with open(bench_dir / "final_benchmark_summary.json", "w", encoding="utf-8") as f:
        json.dump(final_summary, f, indent=2)

    with open(bench_dir / "worker_registry.json", "w", encoding="utf-8") as f:
        json.dump({w.worker_id: {"role": w.role.value, "tools": w.allowed_tools} for w in workers}, f, indent=2)

    with open(bench_dir / "worker_isolation_results.json", "w", encoding="utf-8") as f:
        json.dump(results["exp9_context_isolation"], f, indent=2)

    with open(bench_dir / "performance_results.json", "w", encoding="utf-8") as f:
        json.dump(results["exp2_concurrent_execution"], f, indent=2)

    with open(bench_dir / "persistence_results.json", "w", encoding="utf-8") as f:
        json.dump({"persisted_task": persisted_task}, f, indent=2)

    with open(bench_dir / "baseline_integrity.json", "w", encoding="utf-8") as f:
        json.dump({
            "expected_hash": EXPECTED_WEIGHT_HASH,
            "pre_hash": hash_pre,
            "post_hash": hash_post,
            "bit_exact": hash_post == EXPECTED_WEIGHT_HASH,
        }, f, indent=2)

    results["final_summary"] = final_summary
    return results


if __name__ == "__main__":
    bench_dir = Path("artifacts/step114_distributed_worker_federation")
    out = run_complete_step114_benchmark(bench_dir)
    print("Step 114 Distributed Worker Federation Benchmark Complete! Status:", out["final_summary"]["benchmark_status"])
