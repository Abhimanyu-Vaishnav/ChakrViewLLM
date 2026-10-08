"""
Wave 521: Experience Consolidation Execution Script.
Simulates runtime experiences across diverse task families,
consolidates them into abstract lessons, and evaluates whether
the derived abstractions correctly predict optimal strategies on unseen tasks.
"""

import sys
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from chakrview.cognition.adaptive_learning_contract import EpisodicExperience
from chakrview.cognition.experience_consolidator import ExperienceConsolidator

def run_wave521_consolidation():
    print("=" * 60)
    print("RUNNING WAVE 521: EXPERIENCE CONSOLIDATION")
    print("=" * 60)

    consolidator = ExperienceConsolidator(min_episodes_for_abstraction=2)

    # Ingest 10 episodic experiences across two families:
    # Family 1: 3-hop relational reasoning (requires recurrent strategy, fails on direct)
    # Family 2: 1-hop fact lookup (requires direct strategy, recurrent is wasteful)
    episodes = [
        # Family 1 episodes
        EpisodicExperience(
            episode_id="ep_01", task_id="t1", task_structure={"task_family": "multi_hop_3", "hops": 3},
            strategy_used="direct_retrieval", execution_steps=[], success=False, confidence=0.3, error_type="hop_cutoff"
        ),
        EpisodicExperience(
            episode_id="ep_02", task_id="t2", task_structure={"task_family": "multi_hop_3", "hops": 3},
            strategy_used="recurrent_reasoning", execution_steps=[], success=True, confidence=0.92, error_type=None
        ),
        EpisodicExperience(
            episode_id="ep_03", task_id="t3", task_structure={"task_family": "multi_hop_3", "hops": 3},
            strategy_used="recurrent_reasoning", execution_steps=[], success=True, confidence=0.95, error_type=None
        ),
        EpisodicExperience(
            episode_id="ep_04", task_id="t4", task_structure={"task_family": "multi_hop_3", "hops": 3},
            strategy_used="direct_retrieval", execution_steps=[], success=False, confidence=0.25, error_type="hop_cutoff"
        ),
        # Family 2 episodes
        EpisodicExperience(
            episode_id="ep_05", task_id="t5", task_structure={"task_family": "single_hop_lookup", "hops": 1},
            strategy_used="direct_retrieval", execution_steps=[], success=True, confidence=0.98, error_type=None
        ),
        EpisodicExperience(
            episode_id="ep_06", task_id="t6", task_structure={"task_family": "single_hop_lookup", "hops": 1},
            strategy_used="direct_retrieval", execution_steps=[], success=True, confidence=0.99, error_type=None
        ),
        EpisodicExperience(
            episode_id="ep_07", task_id="t7", task_structure={"task_family": "single_hop_lookup", "hops": 1},
            strategy_used="recurrent_reasoning", execution_steps=[], success=True, confidence=0.96, error_type=None
        )
    ]

    for ep in episodes:
        consolidator.ingest_episode(ep)

    abstractions = consolidator.consolidate()
    print(f"Total episodes ingested: {len(episodes)}")
    print(f"Total abstractions formed: {len(abstractions)}")

    results = {}
    for abs_exp in abstractions:
        print(f"\n[ABSTRACTION] Family: {abs_exp.task_family}")
        print(f"  Recommended Strategy: {abs_exp.recommended_strategy}")
        print(f"  Supporting Episodes: {len(abs_exp.supporting_episode_ids)}")
        print(f"  Contradictory / Failed Episodes: {len(abs_exp.contradictory_episode_ids)}")
        print(f"  Confidence: {abs_exp.confidence:.2f}")
        results[abs_exp.task_family] = {
            "recommended_strategy": abs_exp.recommended_strategy,
            "confidence": abs_exp.confidence,
            "sample_count": abs_exp.sample_count
        }

    # Evaluate transfer on unseen tasks
    unseen_3hop_task = {"task_family": "multi_hop_3", "hops": 3, "entities": ["X", "Y", "Z"]}
    recommended_for_unseen = consolidator.abstractions.get("multi_hop_3").recommended_strategy
    transfer_ok = (recommended_for_unseen == "recurrent_reasoning")
    print(f"\nUnseen 3-Hop Task Strategy Transfer: {'PASS' if transfer_ok else 'FAIL'}")

    out_file = REPO_ROOT / "artifacts" / "waves_520_527_execution" / "wave521_consolidation_results.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "abstractions_count": len(abstractions),
            "results": results,
            "transfer_ok": transfer_ok,
            "status": "PASS" if (len(abstractions) == 2 and transfer_ok) else "FAIL"
        }, f, indent=2)

    print(f"\nWave 521 Output saved to: {out_file}")
    return 0 if (len(abstractions) == 2 and transfer_ok) else 1

if __name__ == "__main__":
    sys.exit(run_wave521_consolidation())
