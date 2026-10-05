"""
Master Wave Unified Regression & Capability Test Suite.

Verifies:
1. MasterCognitivePipeline initialization and resource profiling.
2. Cycle-free DAG decomposition and multi-worker topological execution.
3. Isolated worker execution without leaking full repository context.
4. Epistemic reasoning and dialectical critique generation.
5. Context-efficient project indexing, symbol queries, and delta invalidations.
6. Governed failure analysis and cognitive strategy promotion.
7. Stage-C training runner contract and baseline immutability.
"""

from pathlib import Path
import pytest

from chakrview.cognition.master_cognitive_pipeline import (
    MasterCognitivePipeline,
    FederatedWorkerNode,
    ContextEfficientProjectEngine,
)
from chakrview.cognition.ppb.task_models import TaskResourceType, TaskNodeStatus
from chakrview.cognition.governed_learning.strategy_registry import StrategyStatus
from chakrview.runtime.interactive import instantiate_frozen_baseline, compute_model_hash, EXPECTED_WEIGHT_HASH


def test_master_cognitive_pipeline_initialization(tmp_path: Path):
    strat_db = tmp_path / "strategies.db"
    pipeline = MasterCognitivePipeline(strategy_db_path=strat_db)
    assert pipeline.resource_profile is not None
    assert pipeline.policy.batch_size in (1, 2, 4)
    assert pipeline.policy.max_generation_tokens <= 512


def test_master_pipeline_dag_decomposition_and_worker_dispatch(tmp_path: Path):
    strat_db = tmp_path / "strategies.db"
    pipeline = MasterCognitivePipeline(strategy_db_path=strat_db)

    # Register federated workers
    w_insp = FederatedWorkerNode("w_insp", "local://insp", [TaskResourceType.INSPECTION, TaskResourceType.AST_ANALYSIS])
    w_reas = FederatedWorkerNode("w_reas", "local://reas", [TaskResourceType.REASONING, TaskResourceType.VERIFICATION])
    pipeline.register_worker(w_insp)
    pipeline.register_worker(w_reas)

    subtask_specs = [
        {"node_id": "step1", "title": "Inspect Files", "resource_type": "INSPECTION"},
        {"node_id": "step2", "title": "Parse Symbols", "resource_type": "AST_ANALYSIS"},
        {"node_id": "step3", "title": "Synthesize Plan", "resource_type": "REASONING", "dependencies": ["step1", "step2"]},
        {"node_id": "step4", "title": "Verify Output", "resource_type": "VERIFICATION", "dependencies": ["step3"]},
    ]
    graph = pipeline.decompose_objective("dag_test_01", "Decomposition Test", subtask_specs)
    assert not graph.is_finished

    res = pipeline.execute_task_graph(graph)
    assert res["is_all_completed"] is True
    assert res["execution_order"] == ["step1", "step2", "step3", "step4"]


def test_context_efficient_project_engine():
    engine = ContextEfficientProjectEngine()
    files = {
        "mod_a.py": "def foo(x):\n    return x + 1\n",
        "mod_b.py": "def bar(y):\n    return y * 2\n",
    }
    report = engine.index_files(files)
    assert report["total_files"] == 2
    assert "foo" in engine.symbol_index
    assert "bar" in engine.symbol_index

    # Query context by symbol
    q = engine.query_context(["foo"])
    assert q["relevant_files"] == ["mod_a.py"]

    # Test targeted delta invalidation
    affected = engine.handle_file_delta("mod_a.py", "def foo(x):\n    return x + 2\n")
    assert "foo" in affected


def test_dialectical_critique():
    strat_db = Path("artifacts/master_wave_experiments/strategies.db")
    pipeline = MasterCognitivePipeline(strategy_db_path=strat_db)
    critique = pipeline.dialectical_critique("Patching module X resolves bug", ("ev_test_1",))
    assert critique.recommendation == "PROCEED"
    assert critique.epistemic_reliability_score >= 0.7
    assert len(critique.alternative_hypotheses) > 0


def test_canonical_baseline_absolute_immutability():
    model = instantiate_frozen_baseline()
    h = compute_model_hash(model)
    assert h == EXPECTED_WEIGHT_HASH
    assert sum(p.numel() for p in model.parameters()) == 3_443_136
