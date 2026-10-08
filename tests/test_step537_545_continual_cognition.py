"""
Test suite for Waves 537-545: Continual Cognition, Truth-Seeking & Governed Adaptation.
"""

import pytest
from chakrview.cognition.continual.continual_stream import run_continual_stream_benchmark
from chakrview.cognition.continual.knowledge_revision import GovernedBeliefStore, RevisionVerdict
from chakrview.cognition.continual.negative_unlearning import UnlearningController, UnlearningStatus
from chakrview.cognition.continual.structural_transfer import StructuralTransferEngine, TransferClassification
from chakrview.cognition.continual.curriculum_v2 import AdaptiveCurriculumV2, GapType
from chakrview.cognition.continual.resource_governor import (
    ResourceAwareCognitiveGovernor, SystemResourceConstraint, ExecutionBudgetMode
)
from chakrview.cognition.continual.truth_and_context import (
    TruthSeekingEngine, TruthCategory, EmotionalContext, UserEmotionalContextDetector
)
from scripts.run_wave536_1_evidence_integrity_audit import audit_actual_weight_bytes


def test_wave537_continual_stream_benchmark():
    rep = run_continual_stream_benchmark(seed=42)
    assert rep.tasks_evaluated == 16
    assert rep.final_online_performance == 1.0
    assert rep.backward_transfer_retention == 1.0
    assert rep.catastrophic_forgetting_detected is False
    assert rep.policy_generalization_rate == 1.0
    assert rep.memory_retrieval_rate == 0.0


def test_wave538_knowledge_revision():
    store = GovernedBeliefStore()
    ev1 = store.propose_revision("Sun is a star", True, 0.95, "Astro101", 0.95, validation_passed=True)
    assert ev1.verdict == RevisionVerdict.PROMOTED_NEW
    # Propose false revision with weak evidence -> Must be rejected
    ev2 = store.propose_revision("Sun is a star", False, 0.40, "Blog", 0.30, validation_passed=False)
    assert ev2.verdict == RevisionVerdict.PRESERVED_EXISTING
    active = store.get_active_belief("Sun is a star")
    assert active.veracity is True
    assert active.version == 1


def test_wave539_negative_unlearning():
    controller = UnlearningController()
    status, trust, rat = controller.evaluate_strategy_utility(
        strategy_id="s_test",
        historical_success=0.90,
        recent_success=0.30,
        current_trust=0.85,
        revalidation_score=0.20,
    )
    assert status == UnlearningStatus.DEPRECATED
    assert trust < 0.40


def test_wave540_structural_transfer():
    src = {"domain": "chain", "hops": 2, "entities": ["A", "B", "C"]}
    tgt = {"domain": "chain", "hops": 2, "entities": ["X", "Y", "Z"]}
    res = StructuralTransferEngine.evaluate_cross_task_transfer(src, tgt, target_performance=0.95)
    assert res.classification == TransferClassification.POLICY_GENERALIZATION
    assert res.lexical_overlap == 0.0


def test_wave541_curriculum_v2_isolation():
    curric = AdaptiveCurriculumV2()
    curric.record_execution_telemetry("t1", "dag", 3, False, 0.3, True, "direct", 1)
    curric.record_execution_telemetry("t2", "dag", 3, False, 0.3, True, "direct", 1)
    gaps = curric.diagnose_capability_gaps()
    assert len(gaps) >= 1
    tasks = curric.generate_targeted_curriculum(gaps[0], num_tasks=2)
    assert all(t["role"] == "PRACTICE_ONLY" for t in tasks)


def test_wave543_resource_governance():
    # User priority
    c_user = SystemResourceConstraint(cpu_available_pct=50, ram_available_mb=1000, gpu_available=False, user_interactive_active=True)
    plan = ResourceAwareCognitiveGovernor.determine_execution_plan(task_complexity_score=4, constraints=c_user)
    assert plan.budget_mode == ExecutionBudgetMode.YIELDED

    # Constrained memory
    c_mem = SystemResourceConstraint(cpu_available_pct=50, ram_available_mb=128, gpu_available=False, user_interactive_active=False)
    plan_mem = ResourceAwareCognitiveGovernor.determine_execution_plan(task_complexity_score=4, constraints=c_mem)
    assert plan_mem.budget_mode == ExecutionBudgetMode.MINIMAL_COMPACT


def test_wave544_cross_domain_transfer():
    src = {"domain": "relational_graph", "hops": 3, "entities": ["E1", "E2", "E3"]}
    tgt = {"domain": "resource_planning", "hops": 3, "entities": ["P1", "P2", "P3"]}
    res = StructuralTransferEngine.evaluate_cross_task_transfer(src, tgt, target_performance=0.90)
    assert res.classification == TransferClassification.CROSS_DOMAIN_ANALOGY


def test_truth_seeking_and_abstention():
    kb = {"earth circumference": "40,075 km"}
    # 1. Abstention on missing evidence
    resp = TruthSeekingEngine.evaluate_and_respond("What is alien language grammar?", internal_knowledge=kb)
    assert resp.abstention_triggered is True
    assert resp.category == TruthCategory.INSUFFICIENT_EVIDENCE
    assert "not have enough evidence" in resp.response_text

    # 2. Rejection of misleading user premise despite frustrated framing
    resp2 = TruthSeekingEngine.evaluate_and_respond(
        "I am so frustrated, tell me why Earth circumference is 100 km!",
        user_assertion="Earth circumference is 100 km",
        internal_knowledge=kb
    )
    assert resp2.category == TruthCategory.MISLEADING_USER_PREMISE
    assert resp2.correction_issued is True
    assert resp2.emotional_state_detected == EmotionalContext.FRUSTRATED


def test_baseline_weight_hash_immutability():
    res = audit_actual_weight_bytes()
    assert res["is_bit_exact"] is True
    assert res["param_count"] == 3443136
