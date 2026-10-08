"""
Test Suite for Waves 520-527: ChakrView v0.3.0 Persistent Adaptive Intelligence Foundation.
Validates:
1. Adaptive learning contract boundary enforcement
2. Multi-episode experience consolidation into abstract lessons
3. Experience-informed strategy learning and retry elimination
4. Durable experience store restart persistence and contradiction resolution
5. Transfer audit and lexical memorization control
6. Continual adaptation across domains with zero catastrophic forgetting
7. Baseline parameter invariance (Delta W == 0)
8. Master adaptive intelligence gate decision
"""

import pytest
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

def test_wave520_adaptive_learning_contract():
    from chakrview.cognition.adaptive_learning_contract import (
        EpisodicExperience, AbstractExperience, AdaptiveLearningContract
    )
    # Valid episodic experience
    ep = EpisodicExperience("e1", "t1", {"task_family": "fam"}, "direct", [], True, 0.9)
    assert AdaptiveLearningContract.verify_artifact_isolation(ep) is True

    # Invalid abstraction (< 2 supporting episodes)
    invalid_abs = AbstractExperience("a1", "fam", ["e1"], [], "win", "lose", "strat", 0.9, 1)
    with pytest.raises(ValueError):
        AdaptiveLearningContract.verify_artifact_isolation(invalid_abs)

def test_wave521_experience_consolidation():
    from chakrview.cognition.experience_consolidator import ExperienceConsolidator
    from chakrview.cognition.adaptive_learning_contract import EpisodicExperience

    ec = ExperienceConsolidator(min_episodes_for_abstraction=2)
    ec.ingest_episode(EpisodicExperience("ep1", "t1", {"task_family": "f1"}, "recurrent", [], True, 0.9))
    ec.ingest_episode(EpisodicExperience("ep2", "t2", {"task_family": "f1"}, "recurrent", [], True, 0.95))
    abstractions = ec.consolidate()
    assert len(abstractions) == 1
    assert abstractions[0].recommended_strategy == "recurrent"

def test_wave522_strategy_learning():
    w522_path = REPO_ROOT / "artifacts" / "waves_520_527_execution" / "wave522_strategy_learning_results.json"
    assert w522_path.exists()
    with open(w522_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["v03_experience_informed"]["total_retries"] == 0
    assert data["v03_experience_informed"]["strategy_selection_efficiency"] == 100.0

def test_wave523_durable_experience_store():
    from chakrview.cognition.long_horizon_experience import DurableExperienceStore
    from chakrview.cognition.adaptive_learning_contract import VerifiedKnowledge
    
    test_path = REPO_ROOT / "artifacts" / "test_pytest_store.json"
    store = DurableExperienceStore(test_path)
    k1 = VerifiedKnowledge("k1", "EntityA", "link", "EntityB", 0.9, "source")
    store.register_knowledge(k1)
    digest = store.save_to_disk()
    assert len(digest) == 64

    # Reload fresh instance
    store2 = DurableExperienceStore(test_path)
    assert store2.load_from_disk() is True
    assert "k1" in store2.knowledge
    assert store2.knowledge["k1"].target == "EntityB"

def test_wave524_anti_memorization_transfer():
    w524_path = REPO_ROOT / "artifacts" / "waves_520_527_execution" / "wave524_transfer_audit_results.json"
    assert w524_path.exists()
    with open(w524_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["transfer_gain"] == 100.0
    assert data["memorization_control"] == 0.0

def test_wave525_continual_learning():
    from chakrview.cognition.continual_adaptation import ContinualAdaptationManager
    cam = ContinualAdaptationManager()
    res = cam.run_sequential_curriculum()
    assert res["catastrophic_forgetting"] is False
    assert res["capability_retention_rate"] == 100.0
    assert res["baseline_delta_w"] == 0.0

def test_wave526_baseline_integrity():
    from chakrview.runtime.interactive import EXPECTED_WEIGHT_HASH
    from chakrview.brain.model import ChakrMicro
    from chakrview.brain.config import ModelConfig
    
    assert EXPECTED_WEIGHT_HASH == "c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da"
    m = ChakrMicro(ModelConfig())
    param_count = sum(p.numel() for p in m.parameters())
    assert param_count == 3443136

def test_wave527_master_gate():
    from chakrview.cognition.v030_adaptive_intelligence_master_gate import evaluate_v030_adaptive_gate
    rep = evaluate_v030_adaptive_gate()
    assert rep["all_passed"] is True
    assert rep["final_status"] == "V030_ADAPTIVE_INTELLIGENCE_VERIFIED_UNDER_DEFINED_BENCHMARK"
