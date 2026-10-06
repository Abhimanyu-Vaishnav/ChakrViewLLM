"""Tests for Steps 241-248: Generalized Neural Associative Learning & I3 Promotion Gate."""

from __future__ import annotations

import pytest
import torch

from chakrview.brain.model import ChakrMicro
from chakrview.runtime.interactive import (
    instantiate_frozen_baseline,
    compute_model_hash,
    EXPECTED_WEIGHT_HASH,
)
from chakrview.cognition.generalized_binding_episodes import GeneralizedEpisodeGenerator
from chakrview.cognition.generalized_association_training import run_generalized_associative_training
from chakrview.cognition.identity_diversity_curriculum import (
    IdentityDiversityManager,
    run_identity_diversity_curriculum,
)
from chakrview.cognition.association_layout_training import run_layout_invariance_training
from chakrview.cognition.generalized_disjoint_training import (
    DisjointSplitGenerator,
    run_disjoint_identity_training,
)
from chakrview.cognition.compositional_binding_training import run_compositional_binding_training
from chakrview.cognition.generalized_association_stress import run_generalized_association_stress
from chakrview.tokenizer import BPETokenizer
from scripts.run_step248_benchmark import run_master_generalization_benchmark


@pytest.fixture(scope="module")
def frozen_baseline():
    return instantiate_frozen_baseline()


def test_step241_generalized_episode_generator():
    gen = GeneralizedEpisodeGenerator(seed=42)
    # Test 1, 2, 3, 5, 8 associations
    for n in [1, 2, 3, 5, 8]:
        ep = gen.generate_episode("train", num_associations=n, episode_idx=n)
        assert ep.num_associations <= n
        assert ep.expected_value in [p.val for p in ep.pairs]
        assert ep.query_key in [p.key for p in ep.pairs]
        assert ep.episode_hash is not None

    # Test layouts
    for fmt in ["format_a", "format_b", "format_c", "format_d"]:
        ep = gen.generate_episode("train", num_associations=2, layout_format=fmt)
        assert ep.layout_format == fmt

    # Test contamination detection
    train_eps = gen.generate_batch(10, split="train", num_associations=2)
    eval_eps = gen.generate_batch(10, split="disjoint_test", num_associations=2)
    is_clean, overlap = gen.verify_no_contamination(train_eps, eval_eps)
    assert is_clean is True
    assert overlap == 0


def test_step242_generalized_associative_training(frozen_baseline):
    pre_hash = compute_model_hash(frozen_baseline)
    rep = run_generalized_associative_training(frozen_baseline, seed=42, steps_per_phase=5, eval_episodes=5)
    post_hash = compute_model_hash(frozen_baseline)

    assert pre_hash == post_hash == EXPECTED_WEIGHT_HASH
    assert rep.is_base_frozen is True
    assert len(rep.phases) == 5
    assert rep.language_retention_ratio <= 1.05


def test_step243_identity_diversity_curriculum(frozen_baseline):
    tok = BPETokenizer()
    mgr = IdentityDiversityManager(tokenizer=tok, seed=42)
    # Token integrity check passed without assertions raising
    assert len(mgr.POOL_A_KEYS) > 0
    assert len(mgr.DISJOINT_EVAL_KEYS) > 0

    rep = run_identity_diversity_curriculum(frozen_baseline, seed=42, steps_per_stage=5, eval_episodes=5)
    assert len(rep.stages) == 3
    # Baseline unchanged
    assert compute_model_hash(frozen_baseline) == EXPECTED_WEIGHT_HASH


def test_step244_layout_invariance_training(frozen_baseline):
    rep = run_layout_invariance_training(frozen_baseline, seed=42, steps_per_layout=5, eval_episodes_per_layout=5)
    assert len(rep.layout_results) == 4
    assert rep.all_layouts_trained is True
    assert compute_model_hash(frozen_baseline) == EXPECTED_WEIGHT_HASH


def test_step245_disjoint_identity_training(frozen_baseline):
    rep = run_disjoint_identity_training(
        frozen_baseline,
        seeds=[42, 101, 2026],
        train_steps=5,
        eval_episodes_per_split=5,
    )
    assert len(rep.seeds_tested) == 3
    assert rep.is_base_immutable is True
    assert rep.i3_status in ("I3_CANDIDATE_ACHIEVED", "I3_EMERGING", "I3_NOT_ACHIEVED")
    assert compute_model_hash(frozen_baseline) == EXPECTED_WEIGHT_HASH


def test_step246_compositional_binding_prerequisite_guard(frozen_baseline):
    # Step 245 disjoint transfer is 0.0, so compositional progression must be blocked cleanly
    rep_disjoint = run_disjoint_identity_training(frozen_baseline, seeds=[42], train_steps=2, eval_episodes_per_split=2)
    comp_rep = run_compositional_binding_training(frozen_baseline, disjoint_report=rep_disjoint, seed=42)

    assert comp_rep.prerequisite_satisfied is False
    assert "PREREQUISITE_FAILED" in comp_rep.status_summary
    assert len(comp_rep.level_metrics) == 7


def test_step247_generalized_association_stress_prerequisite_guard(frozen_baseline):
    rep_disjoint = run_disjoint_identity_training(frozen_baseline, seeds=[42], train_steps=2, eval_episodes_per_split=2)
    stress_rep = run_generalized_association_stress(frozen_baseline, disjoint_report=rep_disjoint, seed=42)

    assert stress_rep.prerequisite_satisfied is False
    assert "PREREQUISITE_FAILED" in stress_rep.status_summary
    assert len(stress_rep.dimension_metrics) == 10


def test_step248_master_generalization_benchmark():
    res = run_master_generalization_benchmark(seed=42)
    assert res.all_integrity_passed is True
    assert res.baseline_param_count == 3443136
    assert res.baseline_sha256 == EXPECTED_WEIGHT_HASH
    assert res.i3_status == "I3_NOT_ACHIEVED"
    assert res.i3_promoted is False
