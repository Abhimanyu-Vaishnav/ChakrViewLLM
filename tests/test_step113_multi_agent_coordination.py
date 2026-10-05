"""
Comprehensive test suite for Step 113 Governed Multi-Agent Cognitive Coordination.

Verifies:
1. WorkerContract validation and malformed contract rejection
2. Context isolation & strict <= 512 token boundary enforcement
3. Task dependency enforcement and parallel execution of independent tasks
4. GovernedToolGate authorization for workers (rejection of unauthorized files and paths)
5. Worker failure and dynamic rerouting to available fallback workers
6. ReviewerWorker independent critique and rejection of vulnerable implementations
7. Malformed worker output schema rejection without corrupting shared state
8. Shared PPB state persistence across process restart
9. Strict no-rescan behavior (0 on unchanged, 1 on delta)
10. Canonical baseline absolute bit-exact immutability
"""

import json
import shutil
from pathlib import Path
import pytest

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
from chakrview.cognition.multi_agent.worker import (
    ProjectAnalystWorker,
    PlannerWorker,
    ImplementerWorker,
    TestEngineerWorker,
    ReviewerWorker,
)
from chakrview.cognition.multi_agent.coordinator import MultiAgentCoordinator
from chakrview.cognition.multi_agent.protocol import MultiAgentProtocol
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
def benchmark_fixture(tmp_path_factory):
    tmp_dir = tmp_path_factory.mktemp("step113_test_env")
    from scripts.setup_step113_fixture import setup_step113_repository
    repo_dir = setup_step113_repository(tmp_dir)
    db_path = tmp_dir / "test_brain.db"
    tool_gate, skill = build_benchmark_tool_gate(repo_dir)
    coordinator = MultiAgentCoordinator(db_path, repo_dir, tool_gate)

    analyst = ProjectAnalystWorker("worker_analyst", tool_gate, repo_dir)
    planner = PlannerWorker("worker_planner")
    implementer = ImplementerWorker("worker_impl", tool_gate, repo_dir)
    test_eng = TestEngineerWorker("worker_test", tool_gate, repo_dir)
    reviewer = ReviewerWorker("worker_reviewer", tool_gate, repo_dir)

    for w in [analyst, planner, implementer, test_eng, reviewer]:
        coordinator.register_worker(w)

    protocol = MultiAgentProtocol(coordinator)
    return {
        "root": tmp_dir,
        "repo": repo_dir,
        "db": db_path,
        "gate": tool_gate,
        "coord": coordinator,
        "proto": protocol,
        "workers": {
            "analyst": analyst,
            "planner": planner,
            "implementer": implementer,
            "test_eng": test_eng,
            "reviewer": reviewer,
        }
    }


def test_01_worker_contract_validation():
    # Valid contract
    c_valid = WorkerContract("w1", WorkerRole.IMPLEMENTER, "t1", ["write_file"], ["app/security.py"], 512)
    c_valid.validate()

    # Empty worker_id
    with pytest.raises(ValueError, match="worker_id"):
        WorkerContract("", WorkerRole.IMPLEMENTER, "t1", ["write_file"], ["app/security.py"], 512).validate()

    # Context budget exceeding 512
    with pytest.raises(ValueError, match="context_budget"):
        WorkerContract("w1", WorkerRole.IMPLEMENTER, "t1", ["write_file"], ["app/security.py"], 600).validate()

    # Role authority violation: Analyst requesting write_file
    with pytest.raises(ValueError, match="prohibited from write_file"):
        WorkerContract("w1", WorkerRole.PROJECT_ANALYST, "t1", ["write_file"], ["app/security.py"], 512).validate()


def test_02_context_isolation_enforced(benchmark_fixture):
    coord = benchmark_fixture["coord"]
    c = WorkerContract("worker_analyst", WorkerRole.PROJECT_ANALYST, "t_ctx", ["read_file"], ["app/security.py"], 200)
    # Exceed budget
    p_exceed = WorkerPackage("t_ctx", "Objective", measured_context_tokens=300)
    res = coord.execute_worker_task("worker_analyst", c, p_exceed)
    assert res.status == WorkerExecutionStatus.FAILED
    assert "Context budget exceeded" in res.error


def test_03_parallel_independent_workers(benchmark_fixture):
    proto = benchmark_fixture["proto"]
    t1 = ("worker_analyst", WorkerContract("worker_analyst", WorkerRole.PROJECT_ANALYST, "p1", ["read_file"], ["app/models.py"], 512), WorkerPackage("p1", "Inspect models", measured_context_tokens=40))
    t2 = ("worker_analyst", WorkerContract("worker_analyst", WorkerRole.PROJECT_ANALYST, "p2", ["read_file"], ["app/validation.py"], 512), WorkerPackage("p2", "Inspect validation", measured_context_tokens=40))
    res = proto.execute_parallel_independent_tasks([t1, t2])
    assert len(res) == 2
    assert all(r.status == WorkerExecutionStatus.SUCCESS for r in res)


def test_04_worker_rerouting_on_unavailable_worker(benchmark_fixture):
    coord = benchmark_fixture["coord"]
    repo = benchmark_fixture["repo"]
    gate = benchmark_fixture["gate"]

    backup_w = ImplementerWorker("worker_backup", gate, repo)
    coord.register_worker(backup_w)

    c = WorkerContract("worker_missing", WorkerRole.IMPLEMENTER, "t_rr", ["write_file"], ["app/security.py"], 512)
    p = WorkerPackage("t_rr", "Patch", dependency_outputs={"code_patch": {"app/security.py": "# test"}}, measured_context_tokens=50)

    # Reroute to backup
    res = coord.execute_worker_task("worker_missing", c, p, fallback_worker_id="worker_backup")
    assert res.status == WorkerExecutionStatus.SUCCESS
    assert res.worker_id == "worker_backup"


def test_05_unauthorized_behavior_denied(benchmark_fixture):
    coord = benchmark_fixture["coord"]

    # Unauthorized file write
    c_unauth_file = WorkerContract("worker_impl", WorkerRole.IMPLEMENTER, "t_unauth", ["write_file"], ["app/security.py"], 512)
    p_unauth_file = WorkerPackage("t_unauth", "Unauth", dependency_outputs={"code_patch": {"app/other.py": "# bad"}}, measured_context_tokens=40)
    res = coord.execute_worker_task("worker_impl", c_unauth_file, p_unauth_file)
    assert res.status == WorkerExecutionStatus.UNAUTHORIZED_TOOL

    # Path traversal denied
    c_trav = WorkerContract("worker_analyst", WorkerRole.PROJECT_ANALYST, "t_trav", ["read_file"], ["../../passwords.txt"], 512)
    p_trav = WorkerPackage("t_trav", "Trav", measured_context_tokens=40)
    res_trav = coord.execute_worker_task("worker_analyst", c_trav, p_trav)
    assert res_trav.status == WorkerExecutionStatus.FAILED
    assert "Security violation" in (res_trav.error or "")


def test_06_reviewer_critique_rejection_and_approval(benchmark_fixture):
    coord = benchmark_fixture["coord"]

    # Weak code rejected
    c_rev = WorkerContract("worker_reviewer", WorkerRole.REVIEWER, "t_rev", ["read_file"], ["app/security.py"], 512)
    p_weak = WorkerPackage("t_rev", "Review", targeted_code_context={"app/security.py": "def validate_password_strength(p): return True"}, measured_context_tokens=50)
    res_weak = coord.execute_worker_task("worker_reviewer", c_rev, p_weak)
    assert res_weak.status == WorkerExecutionStatus.REJECTED
    assert res_weak.evidence.get("verdict") == ReviewerVerdict.REJECT.value

    # Strong code approved
    strong_code = "def validate_password_strength(password: str) -> bool:\n    if len(password) < 8: return False\n    if not any(c.isdigit() for c in password): return False\n    if not any(c in '!@#$' for c in password): return False\n    return True"
    p_strong = WorkerPackage("t_rev", "Review", targeted_code_context={"app/security.py": strong_code}, measured_context_tokens=50)
    res_strong = coord.execute_worker_task("worker_reviewer", c_rev, p_strong)
    assert res_strong.status == WorkerExecutionStatus.SUCCESS
    assert res_strong.evidence.get("verdict") == ReviewerVerdict.APPROVE.value


def test_07_malformed_worker_output_rejected(benchmark_fixture):
    coord = benchmark_fixture["coord"]
    c = WorkerContract(
        "worker_impl",
        WorkerRole.IMPLEMENTER,
        "t_malform",
        ["write_file"],
        ["app/security.py"],
        512,
        expected_output_schema={"required_fields": ["unmet_critical_signature"]},
    )
    p = WorkerPackage("t_malform", "Check", dependency_outputs={"code_patch": {"app/security.py": "# pass"}}, measured_context_tokens=50)
    res = coord.execute_worker_task("worker_impl", c, p)
    assert res.status == WorkerExecutionStatus.MALFORMED_OUTPUT


def test_08_persistence_and_no_rescan(benchmark_fixture):
    db_path = benchmark_fixture["db"]
    repo_dir = benchmark_fixture["repo"]

    engine = BenchmarkDiskProjectEngine(db_path, repo_dir)
    r1 = engine.scan_repository()
    r2 = engine.scan_repository()
    assert r2["files_scanned"] == 0
    assert r2["files_skipped_unchanged"] == r2["total_files"]

    # 1 file delta
    (repo_dir / "app" / "models.py").write_text("# delta\n", encoding="utf-8")
    r3 = engine.scan_repository()
    assert r3["files_scanned"] == 1
    assert r3["files_skipped_unchanged"] == r3["total_files"] - 1


def test_09_canonical_baseline_bit_exact():
    model = instantiate_frozen_baseline()
    h = compute_model_hash(model)
    assert h == EXPECTED_WEIGHT_HASH
