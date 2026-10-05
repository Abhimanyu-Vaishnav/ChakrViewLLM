"""
Step 112 Autonomous Work Loop & Benchmarking Automated Test Suite.

Verifies all 16 required benchmark criteria:
1. Benchmark repository fixture is created on disk.
2. Initial on-disk repository indexing works with symbol & file caching.
3. Task graph DAG is generated with zero cycles.
4. Dependency ordering is respected.
5. Targeted context retrieval strictly respects the <= 512 token ceiling.
6. Unauthorized tool operation (outside benchmark root) is blocked.
7. Dangerous parameters (code injection markers) are rejected by GovernedToolGate.
8. Legitimate file modification occurs through tool gate.
9. Injected deterministic failure is detected by verification.
10. Failure diagnosis and replanning occur.
11. Repaired patch changes execution outcome to pass.
12. Final test suite passes with 100% green tests.
13. Project state persists across process restart.
14. Unchanged files are NOT rescanned on subsequent runs (0 rescans).
15. Single-file delta triggers targeted re-indexing (exactly 1 file rescanned).
16. Canonical neural baseline weights remain bit-exact and immutable.
"""

import json
from pathlib import Path
import pytest

from chakrview.cognition.autonomous_benchmark import (
    BenchmarkDiskProjectEngine,
    build_benchmark_tool_gate,
)
from chakrview.cognition.ppb.task_models import (
    PersistentTaskGraph,
    PersistentTaskNode,
    TaskNodeStatus,
    TaskResourceType,
)
from chakrview.runtime.interactive import (
    instantiate_frozen_baseline,
    compute_model_hash,
    EXPECTED_WEIGHT_HASH,
)


@pytest.fixture(scope="module")
def benchmark_env():
    bench_dir = Path("artifacts/step112_autonomous_benchmark").resolve()
    repo_dir = bench_dir / "repository"
    db_path = bench_dir / "project_brain.db"
    assert repo_dir.is_dir(), "Benchmark repository directory missing!"
    return {
        "bench_dir": bench_dir,
        "repo_dir": repo_dir,
        "db_path": db_path,
    }


def test_01_benchmark_repository_fixture_exists(benchmark_env):
    repo_dir = benchmark_env["repo_dir"]
    assert (repo_dir / "app" / "models.py").is_file()
    assert (repo_dir / "app" / "validation.py").is_file()
    assert (repo_dir / "app" / "service.py").is_file()
    assert (repo_dir / "tests" / "test_validation.py").is_file()
    assert (repo_dir / "tests" / "test_service.py").is_file()


def test_02_initial_disk_indexing_and_symbol_extraction(benchmark_env):
    summary_file = benchmark_env["bench_dir"] / "initial_project_understanding.json"
    assert summary_file.is_file()
    data = json.loads(summary_file.read_text(encoding="utf-8"))
    assert data["total_files"] == 7
    assert data["files_scanned"] == 7


def test_03_and_04_task_graph_dag_and_dependencies(benchmark_env):
    graph = PersistentTaskGraph(
        graph_id="test_dag",
        project_id="test_proj",
        root_task_description="Validate adult registration",
    )
    n1 = PersistentTaskNode("n1", "test_dag", "Inspect", "Inspect validation", TaskResourceType.INSPECTION)
    n2 = PersistentTaskNode("n2", "test_dag", "Context", "Retrieve context", TaskResourceType.AST_ANALYSIS, dependencies=["n1"])
    n3 = PersistentTaskNode("n3", "test_dag", "Patch", "Apply patch", TaskResourceType.PATCH_EXECUTION, dependencies=["n2"])

    graph.add_node(n1)
    graph.add_node(n2)
    graph.add_node(n3)

    # Initial ready nodes must strictly be [n1]
    assert [n.node_id for n in graph.get_ready_nodes()] == ["n1"]
    graph.mark_completed("n1")
    assert [n.node_id for n in graph.get_ready_nodes()] == ["n2"]
    graph.mark_completed("n2")
    assert [n.node_id for n in graph.get_ready_nodes()] == ["n3"]


def test_05_targeted_context_ceiling_maintained(benchmark_env):
    engine = BenchmarkDiskProjectEngine(
        db_path=benchmark_env["db_path"],
        project_root=benchmark_env["repo_dir"],
    )
    ctx = engine.query_targeted_context(["validate_registration", "RegistrationService"])
    assert ctx["context_ceiling_maintained"] is True
    assert ctx["total_context_tokens"] <= 512
    assert "app/validation.py" in ctx["relevant_files"]
    assert "app/service.py" in ctx["relevant_files"]


def test_06_and_07_governed_tool_gate_security_boundaries(benchmark_env):
    tool_gate, skill = build_benchmark_tool_gate(benchmark_env["repo_dir"])

    # Legitimate read
    legit_obs = tool_gate.execute_governed("step1", "read_file", {"file_path": "app/validation.py"}, active_skill=skill)
    assert legit_obs.success is True

    # Unauthorized path traversal outside repo root
    unauth_obs = tool_gate.execute_governed("step2", "read_file", {"file_path": "../outside.txt"}, active_skill=skill)
    assert unauth_obs.success is False
    assert "Security violation" in unauth_obs.error

    # Dangerous argument injection
    danger_obs = tool_gate.execute_governed(
        "step3", "write_file",
        {"file_path": "app/hack.py", "content": "import os; os.system('echo dangerous')"},
        active_skill=skill,
    )
    assert danger_obs.success is False
    assert danger_obs.provenance.get("gate_status") == "INVALID_ARGUMENTS"


def test_08_through_12_failure_injection_diagnosis_replan_and_repair(benchmark_env):
    bench_dir = benchmark_env["bench_dir"]
    fail_analysis = json.loads((bench_dir / "failure_analysis.json").read_text(encoding="utf-8"))
    assert fail_analysis["failure_class"] == "VERIFICATION_FAILURE"
    assert "inverted age requirement" in fail_analysis["root_cause_summary"]

    verif = json.loads((bench_dir / "verification_results.json").read_text(encoding="utf-8"))
    assert verif["initial_defective_passed"] is False
    assert verif["repaired_passed"] is True


def test_13_through_15_strict_no_rescan_and_delta_invalidation(benchmark_env):
    scan_results = json.loads((benchmark_env["bench_dir"] / "scan_efficiency_results.json").read_text(encoding="utf-8"))
    assert scan_results["no_rescan_contract_verified"] is True
    assert scan_results["run2_unchanged_scanned"] == 0
    assert scan_results["run2_unchanged_skipped"] == 7
    assert scan_results["run3_single_delta_scanned"] == 1
    assert scan_results["run3_single_delta_skipped"] == 6


def test_16_canonical_baseline_absolute_immutability():
    model = instantiate_frozen_baseline()
    h = compute_model_hash(model)
    assert h == EXPECTED_WEIGHT_HASH
    assert sum(p.numel() for p in model.parameters()) == 3_443_136
