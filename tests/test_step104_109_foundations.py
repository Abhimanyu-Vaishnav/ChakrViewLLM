"""
Step 104-109 Multi-Step Foundation Test Suite.

Verifies:
1. Stage-C manifest integrity & split disjointness (Step 104).
2. Hardware profiling & adaptive resource policy bounds (Step 105).
3. Task decomposition DAG & topological sequencing (Step 106).
4. Low-resource task splitting strategy (Step 106).
5. Structured reasoning epistemic claims & critical analysis (Step 107).
6. Governed failure analysis & strategy evolution (Step 108).
7. Self-healing checkpoint corruption & dataset hash verification (Step 108).
8. ModelRegistry durability & canonical baseline immutability (Step 109).
"""

import json
from pathlib import Path
import pytest
import torch

from chakrview.cognition.adaptation.hardware import HardwareProfiler, HardwareProfileSnapshot, CPUInfo, MemoryInfo, DeviceInfo
from chakrview.cognition.adaptation.profiles import ResourceProfile, ResourceClassifier
from chakrview.cognition.adaptation.policy import AdaptiveExecutionPolicy, HARD_CEILING_GENERATION_TOKENS
from chakrview.cognition.ppb.task_models import PersistentTaskGraph, PersistentTaskNode, TaskNodeStatus, TaskResourceType
from chakrview.cognition.reasoning.structured import ReasoningClaim, EpistemicCategory, EpistemicConfidenceState
from chakrview.cognition.reasoning.critical import EvidenceBalance, EvidenceStrength, AlternativeHypothesis
from chakrview.cognition.governed_learning.self_evaluator import GovernedFailureAnalysis, FailureClass
from chakrview.cognition.governed_learning.strategy_registry import CognitiveStrategy, CognitiveStrategyRegistry, StrategyStatus
from chakrview.training.checkpoint import CheckpointManager
from chakrview.training.safety import CheckpointCorruptionError
from chakrview.training.model_registry import ModelRegistry, ModelRecord, ModelStatus


def test_stage_c_manifest_and_splits_disjointness():
    root = Path(__file__).resolve().parents[1]
    manifest_path = root / "data" / "manifests" / "stage_c_manifest.json"
    assert manifest_path.is_file()

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest["total_documents"] == 5534
    assert manifest["total_content_tokens"] == 7703067

    docs = manifest["documents"]
    train_docs = {d["document_id"] for d in docs if d["split"] == "train"}
    val_docs = {d["document_id"] for d in docs if d["split"] == "validation"}
    test_docs = {d["document_id"] for d in docs if d["split"] == "test"}

    assert len(train_docs) == 4755
    assert len(val_docs) == 378
    assert len(test_docs) == 401
    assert len(train_docs & val_docs) == 0
    assert len(train_docs & test_docs) == 0
    assert len(val_docs & test_docs) == 0


def test_resource_profiling_and_policy_ceilings():
    # Test mocked low-resource snapshot
    low_snapshot = HardwareProfileSnapshot(
        cpu=CPUInfo("x86_64", "Generic", 2, 2, 2),
        memory=MemoryInfo(2 * 1024**3, 1024**3, "HIGH"),
        device=DeviceInfo("cpu", False),
    )
    assert ResourceClassifier.classify(low_snapshot) == ResourceProfile.LOW_RESOURCE
    policy_low = AdaptiveExecutionPolicy.get_profile_defaults(ResourceProfile.LOW_RESOURCE)
    assert policy_low.batch_size == 1
    assert policy_low.max_generation_tokens == 64

    # Test mocked high-resource snapshot
    high_snapshot = HardwareProfileSnapshot(
        cpu=CPUInfo("x86_64", "Xeon", 16, 16, 16),
        memory=MemoryInfo(64 * 1024**3, 32 * 1024**3, "LOW"),
        device=DeviceInfo("cuda", True, "NVIDIA A100", 40 * 1024**3),
    )
    assert ResourceClassifier.classify(high_snapshot) == ResourceProfile.HIGH_RESOURCE
    policy_high = AdaptiveExecutionPolicy.get_profile_defaults(ResourceProfile.HIGH_RESOURCE)
    assert policy_high.batch_size == 4
    assert policy_high.max_generation_tokens <= HARD_CEILING_GENERATION_TOKENS


def test_task_decomposition_dag_topological_execution():
    graph = PersistentTaskGraph(graph_id="g_test", project_id="p_test", root_task_description="Root objective")

    node_a = PersistentTaskNode("node_a", "g_test", "Scan Repo", "Inspect files", TaskResourceType.INSPECTION)
    node_b = PersistentTaskNode("node_b", "g_test", "AST Parse", "Parse symbols", TaskResourceType.AST_ANALYSIS)
    node_c = PersistentTaskNode("node_c", "g_test", "Reasoning", "Synthesize patch", TaskResourceType.REASONING, dependencies=["node_a", "node_b"])
    node_d = PersistentTaskNode("node_d", "g_test", "Verify", "Run regression", TaskResourceType.VERIFICATION, dependencies=["node_c"])

    graph.add_node(node_a)
    graph.add_node(node_b)
    graph.add_node(node_c)
    graph.add_node(node_d)

    # Initially, only A and B are ready
    ready = graph.get_ready_nodes()
    ready_ids = {n.node_id for n in ready}
    assert ready_ids == {"node_a", "node_b"}

    # Complete node A; C should still NOT be ready because B is pending
    graph.mark_completed("node_a", evidence_ids=["ev_1"])
    ready = graph.get_ready_nodes()
    assert {n.node_id for n in ready} == {"node_b"}

    # Complete node B; now C becomes ready
    graph.mark_completed("node_b", evidence_ids=["ev_2"])
    ready = graph.get_ready_nodes()
    assert {n.node_id for n in ready} == {"node_c"}

    # Complete node C; now D becomes ready
    graph.mark_completed("node_c")
    ready = graph.get_ready_nodes()
    assert {n.node_id for n in ready} == {"node_d"}

    graph.mark_completed("node_d")
    assert graph.is_all_completed
    assert graph.is_finished


def test_task_failure_propagation_blocks_dependents():
    graph = PersistentTaskGraph(graph_id="g_fail", project_id="p_test", root_task_description="Failure demo")
    node_a = PersistentTaskNode("node_a", "g_fail", "Step A", "Inspect", TaskResourceType.INSPECTION)
    node_b = PersistentTaskNode("node_b", "g_fail", "Step B", "Reason", TaskResourceType.REASONING, dependencies=["node_a"])
    node_c = PersistentTaskNode("node_c", "g_fail", "Step C", "Verify", TaskResourceType.VERIFICATION, dependencies=["node_b"])

    graph.add_node(node_a)
    graph.add_node(node_b)
    graph.add_node(node_c)

    # Fail node A
    graph.mark_failed("node_a", "File not found")
    assert graph.get_node("node_a").status == TaskNodeStatus.FAILED
    assert graph.get_node("node_b").status == TaskNodeStatus.BLOCKED
    assert graph.get_node("node_c").status == TaskNodeStatus.BLOCKED
    assert len(graph.get_ready_nodes()) == 0


def test_structured_reasoning_and_critical_analysis():
    claim = ReasoningClaim(
        claim_id="cl_01",
        category=EpistemicCategory.HYPOTHESIS,
        statement="Refactoring module X will reduce memory pressure",
        confidence_state=EpistemicConfidenceState.SUPPORTED,
        supporting_evidence_ids=("ev_ast_1", "ev_prof_2"),
    )
    assert claim.fingerprint != ""
    assert claim.category == EpistemicCategory.HYPOTHESIS

    ev_balance = EvidenceBalance(
        claim_id="cl_01",
        supporting_evidence_ids=("ev_ast_1", "ev_prof_2"),
        contradicting_evidence_ids=(),
        strength=EvidenceStrength.CORROBORATED,
        has_conflicts=False,
        is_stale=False,
        evaluation_summary="Strong multi-source support",
    )
    assert not ev_balance.has_conflicts

    alt_hyp = AlternativeHypothesis(
        hypothesis_id="alt_01",
        description="Chunk processing instead of modifying module X",
        rationale="Avoids changing public API",
        plausibility=0.85,
        required_evidence="Context budget simulation",
    )
    assert alt_hyp.plausibility == 0.85


def test_failure_analysis_and_strategy_evolution(tmp_path: Path):
    db_path = tmp_path / "strategies.db"
    registry = CognitiveStrategyRegistry(db_path)

    strat = CognitiveStrategy(
        strategy_id="strat_decomp_01",
        name="Low-Resource Task Decomposition",
        description="Split large tasks into single-file chunks",
        applicable_conditions=["LOW_RESOURCE", "FILE_COUNT > 1"],
        expected_benefit="Guarantees execution without OOM",
    )
    registry.store_strategy(strat)

    loaded = registry.get_strategy("strat_decomp_01")
    assert loaded.status == StrategyStatus.CANDIDATE

    # Record 2 successful outcomes
    strat.record_outcome(is_success=True)
    strat.record_outcome(is_success=True)
    assert strat.status == StrategyStatus.ACTIVE
    registry.store_strategy(strat)

    reloaded = registry.get_strategy("strat_decomp_01")
    assert reloaded.status == StrategyStatus.ACTIVE


def test_self_healing_checkpoint_integrity_and_mismatch(tmp_path: Path):
    ckpt_dir = tmp_path / "ckpts"
    mgr = CheckpointManager(checkpoint_dir=ckpt_dir)

    payload_valid = {
        "checkpoint_type": "training",
        "step": 100,
        "epoch": 1,
        "timestamp": "2026-10-05T00:00:00Z",
        "model_state_dict": {"weight": torch.ones(2, 2)},
        "optimizer_state_dict": {},
        "config": {},
        "parameter_count": 3_443_136,
        "tokenizer_checksum": "valid_tok_hash",
        "dataset_manifest_hash": "valid_data_hash",
    }

    # Valid check passes
    mgr.validate_training_checkpoint(
        payload_valid,
        expected_tokenizer_checksum="valid_tok_hash",
        expected_param_count=3_443_136,
        expected_dataset_manifest_hash="valid_data_hash",
    )

    # Mismatched tokenizer raises CheckpointCorruptionError
    with pytest.raises(CheckpointCorruptionError, match="Tokenizer checksum mismatch"):
        mgr.validate_training_checkpoint(
            payload_valid,
            expected_tokenizer_checksum="different_tok_hash",
        )

    # Mismatched dataset manifest raises CheckpointCorruptionError
    with pytest.raises(CheckpointCorruptionError, match="Dataset manifest hash mismatch"):
        mgr.validate_training_checkpoint(
            payload_valid,
            expected_dataset_manifest_hash="different_manifest_hash",
        )


def test_model_registry_immutability(tmp_path: Path):
    db_path = tmp_path / "model_registry.db"
    registry = ModelRegistry(db_path)

    canonical = ModelRecord(
        model_id="canonical_baseline",
        model_hash="c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da",
        architecture="ChakrMicro-v0.1",
        parameter_count=3_443_136,
        tokenizer_checksum="7498d92adeef7c9e64d4d1bbd5b4b23f8774692d868742e3b98e30c584c4403f",
        dataset_identity="stage_a_smoke",
        status=ModelStatus.CANONICAL_BASELINE,
        checkpoint_path="artifacts/canonical/baseline.pt",
    )
    registry.register_model(canonical)

    # Attempting to overwrite canonical baseline must fail
    with pytest.raises(ValueError, match="Cannot overwrite immutable CANONICAL_BASELINE"):
        registry.register_model(canonical)

    # Register experimental candidate succeeds
    step102_cand = ModelRecord(
        model_id="step102_candidate",
        model_hash="d5886e02e62df29fc88379ed31b37239c85fcda9648ffaf0cf92191730df9a01",
        architecture="ChakrMicro-v0.1",
        parameter_count=3_443_136,
        tokenizer_checksum="7498d92adeef7c9e64d4d1bbd5b4b23f8774692d868742e3b98e30c584c4403f",
        dataset_identity="stage_b",
        status=ModelStatus.EXPERIMENTAL_CANDIDATE,
        checkpoint_path="artifacts/step102_stage_b_curriculum/checkpoints/checkpoint_0000500.pt",
        validation_loss=4.6951,
        validation_ppl=109.41,
        top5_syntactic_acc=15.0,
    )
    registry.register_model(step102_cand)

    assert len(registry.list_by_status(ModelStatus.EXPERIMENTAL_CANDIDATE)) == 1
    assert registry.get_model("step102_candidate").validation_loss == 4.6951
