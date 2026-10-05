"""
Dedicated Tests for Steps 94–100: Governed Cognitive Learning & Release Readiness.

Covers:
- Step 94: GovernedExperienceRecord schema, fingerprinting, and SQLite persistence.
- Step 94: Epistemic authority ranking (FACT > OBSERVATION > LESSON > INFERENCE > HYPOTHESIS > UNKNOWN).
- Step 95: GovernedSelfEvaluator diagnostic questions, failure classification, and TaskAuditReport.
- Step 96: CognitiveLessonExtractor, lesson categorization, and conflict resolution rules.
- Step 97: GovernedCognitiveImprovementLoop closed-cycle execution and ΔW = 0 invariant preservation.
- Step 98: CognitiveStrategyRegistry persistence, confidence tracking, and promotion rules.
- Step 99: EvaluationRegressionMemory capability proof recording and summary reporting.
- Step 100: FirstReleaseReadinessGate 15-point audit, fail-closed behavior, and limitations disclosure.
- Neural baseline invariant: Parameters = 3,443,136, Hash = c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da.
"""

import os
import shutil
import tempfile
import time
from pathlib import Path
import pytest
import torch
import hashlib

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.cognition.ppb.task_models import PersistentTaskNode, TaskNodeStatus, TaskResourceType
from chakrview.cognition.ppb.expansion_models import GoalVerificationVerdict
from chakrview.cognition.governed_learning import (
    ExperienceEpistemicCategory,
    ExperienceProvenance,
    GovernedExperienceRecord,
    GovernedExperienceStore,
    EPISTEMIC_AUTHORITY_RANK,
    FailureClass,
    GovernedFailureAnalysis,
    TaskAuditReport,
    GovernedSelfEvaluator,
    LessonCategory,
    CognitiveLesson,
    ConflictResolutionOutcome,
    CognitiveLessonExtractor,
    StrategyStatus,
    CognitiveStrategy,
    CognitiveStrategyRegistry,
    ImprovementCycleResult,
    GovernedCognitiveImprovementLoop,
    CapabilityProofStatus,
    EvaluationMemoryRecord,
    EvaluationRegressionMemory,
    ReadinessVerdict,
    ReleaseCriterionCheck,
    ReleaseAuditReport,
    FirstReleaseReadinessGate,
)

EXPECTED_WEIGHT_HASH = "c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da"
EXPECTED_PARAM_COUNT = 3_443_136


@pytest.fixture
def temp_dir():
    d = tempfile.mkdtemp(prefix="step94_100_test_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


# =============================================================================
# Step 94 Tests: Experience Records & Epistemic Authority
# =============================================================================
def test_step94_experience_persistence_and_epistemic_ranks(temp_dir):
    db_path = os.path.join(temp_dir, "exp_store.db")
    store = GovernedExperienceStore(db_path)

    # 1. Epistemic authority hierarchy validation
    assert EPISTEMIC_AUTHORITY_RANK[ExperienceEpistemicCategory.FACT] > EPISTEMIC_AUTHORITY_RANK[ExperienceEpistemicCategory.LESSON]
    assert EPISTEMIC_AUTHORITY_RANK[ExperienceEpistemicCategory.LESSON] > EPISTEMIC_AUTHORITY_RANK[ExperienceEpistemicCategory.INFERENCE]
    assert EPISTEMIC_AUTHORITY_RANK[ExperienceEpistemicCategory.INFERENCE] > EPISTEMIC_AUTHORITY_RANK[ExperienceEpistemicCategory.HYPOTHESIS]
    assert EPISTEMIC_AUTHORITY_RANK[ExperienceEpistemicCategory.HYPOTHESIS] > EPISTEMIC_AUTHORITY_RANK[ExperienceEpistemicCategory.UNKNOWN]

    # 2. Record creation & Fingerprint
    rec = GovernedExperienceRecord(
        record_id="exp_001",
        project_id="proj_alpha",
        task_goal="Inspect API schema",
        action_taken="Parsed AST of schema.py",
        observation="Schema defines 5 endpoints",
        result_summary="Completed successfully",
        epistemic_category=ExperienceEpistemicCategory.FACT,
        is_success=True,
        affected_modules=["schema.py"],
        evidence_ids=["ev_101"],
    )
    fp = rec.compute_fingerprint()
    assert len(fp) == 64

    # 3. Persistence to SQLite and retrieval
    store.store_experience(rec)
    loaded = store.get_experience("exp_001")
    assert loaded is not None
    assert loaded.project_id == "proj_alpha"
    assert loaded.epistemic_category == ExperienceEpistemicCategory.FACT
    assert loaded.is_success is True
    assert loaded.affected_modules == ["schema.py"]

    # 4. Process restart simulation
    del store
    store_reloaded = GovernedExperienceStore(db_path)
    res = store_reloaded.query_experiences("proj_alpha", category=ExperienceEpistemicCategory.FACT)
    assert len(res) == 1
    assert res[0].record_id == "exp_001"


# =============================================================================
# Step 95 Tests: Self-Evaluation & Failure Diagnosis
# =============================================================================
def test_step95_self_evaluation_and_failure_analysis():
    evaluator = GovernedSelfEvaluator()

    # Case A: Success Node
    node_succ = PersistentTaskNode(
        node_id="node_succ",
        graph_id="g1",
        title="Verify imports",
        description="Verify clean imports",
        resource_type=TaskResourceType.VERIFICATION,
        status=TaskNodeStatus.COMPLETED,
        affected_files=["core/base.py"],
    )
    audit_succ = evaluator.audit_node_execution(node_succ, {"passed": True})
    assert audit_succ.is_success is True
    assert audit_succ.verdict == GoalVerificationVerdict.ACCEPT
    assert audit_succ.failure_analysis is None
    assert "Success Lesson" in audit_succ.extracted_lesson

    # Case B: Failure Node with Diagnostic
    node_fail = PersistentTaskNode(
        node_id="node_fail",
        graph_id="g1",
        title="Execute uninspected patch",
        description="Patch core database connection",
        resource_type=TaskResourceType.PATCH_EXECUTION,
        status=TaskNodeStatus.FAILED,
        failure_reason="ModuleNotFoundError: No module named 'vault_crypto'",
        affected_files=["db/conn.py"],
    )
    audit_fail = evaluator.audit_node_execution(node_fail, {}, error_message=node_fail.failure_reason)
    assert audit_fail.is_success is False
    assert audit_fail.verdict == GoalVerificationVerdict.REJECT
    assert audit_fail.failure_analysis is not None
    assert audit_fail.failure_analysis.failure_class == FailureClass.DEPENDENCY_MISMATCH
    assert "vault_crypto" in audit_fail.failure_analysis.root_cause_summary


# =============================================================================
# Step 96 Tests: Cognitive Lesson Extraction & Conflict Resolution
# =============================================================================
def test_step96_lesson_extraction_and_conflict_resolution():
    extractor = CognitiveLessonExtractor()

    # Conflict resolution checks
    # FACT vs LESSON -> Existing FACT must be preserved
    res1 = extractor.resolve_epistemic_conflict(
        existing_category=ExperienceEpistemicCategory.FACT,
        new_category=ExperienceEpistemicCategory.LESSON,
    )
    assert res1 == ConflictResolutionOutcome.PRESERVE_EXISTING

    # INFERENCE vs FACT -> FACT supersedes INFERENCE
    res2 = extractor.resolve_epistemic_conflict(
        existing_category=ExperienceEpistemicCategory.INFERENCE,
        new_category=ExperienceEpistemicCategory.FACT,
    )
    assert res2 == ConflictResolutionOutcome.SUPERSEDE

    # Extraction from failed audit
    evaluator = GovernedSelfEvaluator()
    node_fail = PersistentTaskNode(
        node_id="node_mem_overflow",
        graph_id="g1",
        title="Load large file into context",
        description="Load 50MB file",
        resource_type=TaskResourceType.REASONING,
        status=TaskNodeStatus.FAILED,
        failure_reason="ContextBudgetExceededError: Sequence length exceeds memory budget",
        affected_files=["big_data.csv"],
    )
    audit_fail = evaluator.audit_node_execution(node_fail, {}, error_message=node_fail.failure_reason)
    lesson = extractor.extract_from_audit(audit_fail, project_id="proj_alpha")
    assert lesson is not None
    assert lesson.category == LessonCategory.RESOURCE_CONSTRAINT
    assert lesson.confidence == 0.90


# =============================================================================
# Step 97 & 98 Tests: Strategy Registry & Improvement Loop
# =============================================================================
def test_step97_98_strategy_registry_and_improvement_loop(temp_dir):
    exp_db = os.path.join(temp_dir, "exp.db")
    strat_db = os.path.join(temp_dir, "strat.db")
    exp_store = GovernedExperienceStore(exp_db)
    strat_reg = CognitiveStrategyRegistry(strat_db)

    # Verify default seeded active strategies
    active_strats = strat_reg.list_strategies(status=StrategyStatus.ACTIVE)
    assert len(active_strats) >= 3

    loop = GovernedCognitiveImprovementLoop(
        experience_store=exp_store,
        strategy_registry=strat_reg,
    )

    node_t = PersistentTaskNode(
        node_id="node_scan",
        graph_id="g_loop",
        title="Scan repository",
        description="Scan codebase",
        resource_type=TaskResourceType.INSPECTION,
        status=TaskNodeStatus.COMPLETED,
    )

    # Run loop on successful completion with associated strategy
    res = loop.run_cycle_on_task_completion(
        project_id="proj_loop",
        node=node_t,
        execution_result={"files": 5},
        associated_strategy_id="strat_targeted_ppb_retrieval",
    )
    assert res.is_success is True
    assert res.updated_strategy is not None
    assert res.updated_strategy.success_count >= 6
    assert res.neural_invariant_preserved is True


# =============================================================================
# Step 99 Tests: Evaluation & Regression Memory
# =============================================================================
def test_step99_evaluation_regression_memory(temp_dir):
    eval_db = os.path.join(temp_dir, "eval_mem.db")
    mem = EvaluationRegressionMemory(eval_db)

    rec = EvaluationMemoryRecord(
        eval_id="eval_001",
        capability_name="TaskGraphCyclePrevention",
        test_identity="test_cycle_prevention",
        input_condition="Cyclic DAG A->B->A",
        expected_result="ValueError raised",
        actual_result="ValueError raised",
        passed=True,
        proof_status=CapabilityProofStatus.PROVEN,
        resource_profile="STANDARD",
        model_weight_hash=EXPECTED_WEIGHT_HASH,
    )
    mem.record_evaluation(rec)

    st = mem.get_capability_status("TaskGraphCyclePrevention")
    assert st == CapabilityProofStatus.PROVEN

    report = mem.get_summary_report()
    assert report["proven_count"] == 1
    assert "TaskGraphCyclePrevention" in report["proven_capabilities"]


# =============================================================================
# Step 100 Tests: First Release Readiness Gate
# =============================================================================
def test_step100_release_readiness_gate():
    gate = FirstReleaseReadinessGate()

    # Case 1: All criteria passed and neural hash intact
    report = gate.evaluate_release_readiness(
        current_param_count=EXPECTED_PARAM_COUNT,
        current_weight_hash=EXPECTED_WEIGHT_HASH,
        regression_all_passed=True,
    )
    assert report.verdict == ReadinessVerdict.READY_FOR_RELEASE
    assert report.passed_criteria_count == 15
    assert report.readiness_percentage == 100.0
    assert len(report.blocked_items) == 0
    assert len(report.disclosed_limitations) >= 4

    summary_text = report.generate_human_readable_summary()
    assert "CHAKRVIEW RELEASE READINESS AUDIT: READY_FOR_RELEASE" in summary_text
    assert "Neural Core Integrity" in summary_text

    # Case 2: Neural invariant broken -> Fails closed with NOT_READY
    bad_report = gate.evaluate_release_readiness(
        current_param_count=EXPECTED_PARAM_COUNT,
        current_weight_hash="corrupted_hash",
        regression_all_passed=True,
    )
    assert bad_report.verdict == ReadinessVerdict.NOT_READY
    assert "Neural Core Integrity" in bad_report.blocked_items


# =============================================================================
# Neural Baseline Invariant Verification
# =============================================================================
def test_neural_baseline_invariant():
    torch.manual_seed(42)
    model = ChakrMicro(ModelConfig())
    model.eval()

    param_count = sum(p.numel() for p in model.parameters())
    assert param_count == EXPECTED_PARAM_COUNT

    hasher = hashlib.sha256()
    with torch.no_grad():
        for name, param in sorted(model.named_parameters()):
            hasher.update(name.encode("utf-8"))
            hasher.update(param.detach().cpu().numpy().tobytes())
    h = hasher.hexdigest()
    assert h == EXPECTED_WEIGHT_HASH
