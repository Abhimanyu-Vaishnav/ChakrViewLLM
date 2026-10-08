"""
Test Suite for Waves 528-535: ChakrView v0.3 Governed Self-Improvement & Adaptive Cognitive Loop.
Validates:
1. Learning worthiness decisions across novelty profiles
2. Experience trust evaluator adversarial discrimination
3. Strategy lifecycle transitions, aging, and revalidation
4. Conflict resolution epistemic safety (UNKNOWN state)
5. Self-generated failure-driven practice curriculum
6. Transactional rollback under candidate regression
7. Multi-cycle long-horizon self-improvement loop
8. Master self-improvement gate verdict
"""

import pytest
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

def test_wave528_learning_worthiness():
    from chakrview.cognition.learning_worthiness import (
        LearningWorthinessEngine, LearningWorthinessProfile, LearningDecision
    )
    p_dup = LearningWorthinessProfile(0.05, 10, 0.9, 0.9, 1.0, 0.0, 5, 0, True, 1.0, 100, existing_strategy_coverage=True)
    res = LearningWorthinessEngine.evaluate(p_dup)
    assert res["decision"] == LearningDecision.IGNORE

def test_wave529_experience_trust():
    from chakrview.cognition.experience_trust import (
        ExperienceTrustEvaluator, EvidenceObservation, TrustLevel
    )
    # Overconfident false prediction
    obs = [EvidenceObservation("test", 0.99, False, False, True, False, False)]
    res = ExperienceTrustEvaluator.evaluate(obs)
    assert res["trust_level"] == TrustLevel.REJECTED

def test_wave530_strategy_lifecycle():
    from chakrview.cognition.strategy_lifecycle import TrackedStrategy, StrategyState
    strat = TrackedStrategy("s_test", "fam")
    for _ in range(4):
        strat.record_outcome(False)
    assert strat.state == StrategyState.REVALIDATION_REQUIRED

def test_wave531_conflict_resolution():
    from chakrview.cognition.conflict_resolver import (
        ConflictResolver, ConflictingClaim, ConflictResolutionDecision
    )
    c1 = ConflictingClaim("c1", "Claim A", 0.5, 1, True, "global")
    c2 = ConflictingClaim("c2", "Claim B", 0.5, 1, True, "global")
    res = ConflictResolver.resolve(c1, c2)
    assert res["decision"] == ConflictResolutionDecision.UNKNOWN

def test_wave532_self_generated_curriculum():
    from chakrview.cognition.self_generated_curriculum import SelfGeneratedCurriculumEngine
    eng = SelfGeneratedCurriculumEngine()
    eng.record_failure({"id": "f1", "task_family": "chain_4", "hops": 4}, "hop_cutoff")
    eng.record_failure({"id": "f2", "task_family": "chain_4", "hops": 4}, "hop_cutoff")
    gaps = eng.analyze_gaps()
    assert len(gaps) == 1
    tasks = eng.generate_practice_curriculum(gaps[0], 3)
    assert len(tasks) == 3
    assert tasks[0]["role"] == "practice"

def test_wave533_self_improvement_rollback():
    from chakrview.cognition.self_improvement_transaction import SelfImprovementTransaction
    state = {"strategies": {"strat_1": "active"}}
    tx = SelfImprovementTransaction(state)
    tx.apply_candidate({"strategy_updates": {"strat_1": "degraded"}})
    res = tx.evaluate_and_finalize(held_out_gain=0.0, regression_detected=True)
    assert res["status"] == "ROLLED_BACK"
    assert res["final_state"]["strategies"]["strat_1"] == "active"

def test_wave534_long_horizon_loop():
    w534_path = REPO_ROOT / "artifacts" / "waves_520_527_execution" / "wave534_long_horizon_loop_results.json"
    assert w534_path.exists()
    with open(w534_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["status"] == "PASS"
    assert data["cycles_completed"] == 6

def test_wave535_master_gate():
    from chakrview.cognition.v030_self_improvement_master_gate import evaluate_v030_self_improvement_gate
    rep = evaluate_v030_self_improvement_gate()
    assert rep["all_passed"] is True
    assert rep["final_status"] == "V030_SELF_IMPROVEMENT_VERIFIED_UNDER_DEFINED_BENCHMARK"
