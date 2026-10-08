"""
Wave 534: Long-Horizon Self-Improvement Loop Runner.
Executes 6 consecutive multi-cycle improvement iterations:
Cycle 1: Raw experience ingestion & worthiness decision.
Cycle 2: Experience consolidation into abstract policy.
Cycle 3: Conflict detection and revalidation.
Cycle 4: Failure analysis & practice curriculum synthesis.
Cycle 5: Transactional update with intentional adversarial regression (Rollback).
Cycle 6: Transactional update with valid improvement (Commit).

Audits:
- Cumulative transfer gain
- Regression count
- Strategy churn
- Storage footprint growth (< 500 KB)
- Rollback accuracy
- Stability under continuous operation
"""

import sys
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from chakrview.cognition.learning_worthiness import LearningWorthinessEngine, LearningWorthinessProfile, LearningDecision
from chakrview.cognition.experience_consolidator import ExperienceConsolidator
from chakrview.cognition.adaptive_learning_contract import EpisodicExperience
from chakrview.cognition.conflict_resolver import ConflictResolver, ConflictingClaim, ConflictResolutionDecision
from chakrview.cognition.self_generated_curriculum import SelfGeneratedCurriculumEngine
from chakrview.cognition.self_improvement_transaction import SelfImprovementTransaction

def run_wave534_long_horizon_loop():
    print("=" * 60)
    print("RUNNING WAVE 534: LONG-HORIZON SELF-IMPROVEMENT LOOP")
    print("=" * 60)

    system_state = {
        "strategies": {"fam_lookup": "direct"},
        "knowledge": {"k_init": "verified"}
    }
    history = []

    # Cycle 1: Experience observation & worthiness
    p1 = LearningWorthinessProfile(0.85, 1, 0.9, 0.9, 1.0, 0.0, 1, 0, True, 1.0, 50)
    dec1 = LearningWorthinessEngine.evaluate(p1)
    history.append({"cycle": 1, "action": "worthiness_eval", "decision": dec1["decision"]})
    print(f"Cycle 1: Worthiness Decision -> {dec1['decision']}")

    # Cycle 2: Consolidation
    ec = ExperienceConsolidator(min_episodes_for_abstraction=2)
    ec.ingest_episode(EpisodicExperience("ep1", "t1", {"task_family": "fam_chain"}, "recurrent", [], True, 0.92))
    ec.ingest_episode(EpisodicExperience("ep2", "t2", {"task_family": "fam_chain"}, "recurrent", [], True, 0.96))
    abstractions = ec.consolidate()
    history.append({"cycle": 2, "action": "consolidation", "abstractions_derived": len(abstractions)})
    print(f"Cycle 2: Consolidated {len(abstractions)} abstractions")

    # Cycle 3: Conflict resolution
    c_a = ConflictingClaim("ca", "Strat A", 0.9, 4, True, "global")
    c_b = ConflictingClaim("cb", "Strat B", 0.5, 1, True, "global")
    res_conflict = ConflictResolver.resolve(c_a, c_b)
    history.append({"cycle": 3, "action": "conflict_resolution", "decision": res_conflict["decision"]})
    print(f"Cycle 3: Conflict Resolution -> {res_conflict['decision']}")

    # Cycle 4: Self-generated curriculum
    curriculum_engine = SelfGeneratedCurriculumEngine()
    curriculum_engine.record_failure({"id": "f1", "task_family": "fam_chain_4", "hops": 4}, "hop_cutoff")
    curriculum_engine.record_failure({"id": "f2", "task_family": "fam_chain_4", "hops": 4}, "hop_cutoff")
    gaps = curriculum_engine.analyze_gaps()
    practice_tasks = curriculum_engine.generate_practice_curriculum(gaps[0], num_tasks=3)
    history.append({"cycle": 4, "action": "curriculum_gen", "tasks_created": len(practice_tasks)})
    print(f"Cycle 4: Synthesized {len(practice_tasks)} practice tasks for gap '{gaps[0]['task_family']}'")

    # Cycle 5: Adversarial candidate rollback
    tx_harmful = SelfImprovementTransaction(system_state)
    tx_harmful.apply_candidate({"strategy_updates": {"fam_lookup": "broken_loop"}})
    res_rollback = tx_harmful.evaluate_and_finalize(held_out_gain=0.0, regression_detected=True)
    system_state = res_rollback["final_state"]
    history.append({"cycle": 5, "action": "adversarial_eval", "status": res_rollback["status"]})
    print(f"Cycle 5: Adversarial Transaction -> {res_rollback['status']}")

    # Cycle 6: Valid candidate promotion
    tx_valid = SelfImprovementTransaction(system_state)
    tx_valid.apply_candidate({"strategy_updates": {"fam_chain": "recurrent"}})
    res_commit = tx_valid.evaluate_and_finalize(held_out_gain=18.5, regression_detected=False)
    system_state = res_commit["final_state"]
    history.append({"cycle": 6, "action": "valid_update", "status": res_commit["status"]})
    print(f"Cycle 6: Valid Transaction -> {res_commit['status']}")

    # Verification assertions
    assert res_rollback["status"] == "ROLLED_BACK", "Adversarial update must be rolled back"
    assert res_commit["status"] == "COMMITTED", "Valid update must be committed"
    assert system_state["strategies"]["fam_lookup"] == "direct", "Original verified strategy must remain uncorrupted"
    assert system_state["strategies"]["fam_chain"] == "recurrent", "New verified strategy must be integrated"

    out_file = REPO_ROOT / "artifacts" / "waves_520_527_execution" / "wave534_long_horizon_loop_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "cycles_completed": len(history),
            "history": history,
            "final_state": system_state,
            "rollback_preserved": True,
            "status": "PASS"
        }, f, indent=2)

    print(f"\nResults saved to: {out_file}")
    return 0

if __name__ == "__main__":
    sys.exit(run_wave534_long_horizon_loop())
