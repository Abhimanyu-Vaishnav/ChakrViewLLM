"""Tests for Steps 289-296: Stable Intermediate State & Iterative Composition Wave."""

from __future__ import annotations

import pytest
import torch

from chakrview.brain.model import ChakrMicro
from chakrview.runtime.interactive import (
    instantiate_frozen_baseline,
    compute_model_hash,
    EXPECTED_WEIGHT_HASH,
)
from chakrview.cognition.forensic_intermediate_baseline import (
    run_forensic_intermediate_baseline,
)
from chakrview.cognition.minimal_state_normalization import (
    run_minimal_state_normalization_experiment,
)
from chakrview.cognition.tiny_gated_refinement import (
    run_gated_state_refinement_experiment,
)
from chakrview.cognition.contrastive_intermediate_stabilization import (
    run_contrastive_intermediate_stabilization,
)
from chakrview.cognition.controlled_iterative_refinement import (
    run_controlled_iterative_refinement,
)
from chakrview.cognition.strict_i4_reevaluation import (
    run_strict_i4_reevaluation,
)
from chakrview.cognition.three_hop_eligibility_analysis import (
    evaluate_three_hop_eligibility_or_failure_analysis,
)
from chakrview.cognition.two_hop_composition_architecture import (
    ChakrMicroCompositionalReasoningModel,
)
from scripts.run_step296_benchmark import run_master_wave296_benchmark


@pytest.fixture(scope="module")
def frozen_baseline():
    return instantiate_frozen_baseline()


def test_step289_forensic_intermediate_baseline(frozen_baseline):
    """Step 289: Verify forensic measurement of intermediate state norm drift & cosine similarity."""
    rep = run_forensic_intermediate_baseline(frozen_baseline, seeds=[42], num_episodes_per_split=4, train_steps=5)
    assert len(rep.per_split_seed_metrics) == 4
    assert hasattr(rep, "overall_mean_norm_drift")
    assert hasattr(rep, "overall_mean_cosine_sim")
    assert hasattr(rep, "overall_mean_key_margin")
    assert compute_model_hash(frozen_baseline) == EXPECTED_WEIGHT_HASH


def test_step290_minimal_state_normalization(frozen_baseline):
    """Step 290: Verify minimal normalization comparison (raw, layernorm, rmsnorm)."""
    rep = run_minimal_state_normalization_experiment(frozen_baseline, seeds=[42], train_steps=5, eval_episodes=4)
    assert "raw" in rep.variants
    assert "layernorm" in rep.variants
    assert "rmsnorm" in rep.variants
    assert rep.best_variant in rep.variants
    assert compute_model_hash(frozen_baseline) == EXPECTED_WEIGHT_HASH


def test_step291_tiny_gated_refinement(frozen_baseline):
    """Step 291: Verify tiny gated refinement module, learned gamma, and ablation."""
    rep = run_gated_state_refinement_experiment(frozen_baseline, seeds=[42], train_steps=5, eval_episodes=4)
    assert hasattr(rep, "active_g4_tok_acc")
    assert hasattr(rep, "ablated_g4_tok_acc")
    assert rep.refiner_params > 0
    assert rep.refiner_params < 30000  # Strict parameter budget check
    assert compute_model_hash(frozen_baseline) == EXPECTED_WEIGHT_HASH


def test_step292_contrastive_intermediate_stabilization(frozen_baseline):
    """Step 292: Verify auxiliary contrastive objective on intermediate query vector."""
    rep = run_contrastive_intermediate_stabilization(frozen_baseline, seeds=[42], train_steps=5, eval_episodes=4)
    assert hasattr(rep, "mean_g4_tok_acc")
    assert hasattr(rep, "mean_key_margin")
    assert compute_model_hash(frozen_baseline) == EXPECTED_WEIGHT_HASH


def test_step293_controlled_iterative_refinement(frozen_baseline):
    """Step 293: Verify 0 vs 1 vs 2 refinement steps."""
    rep = run_controlled_iterative_refinement(frozen_baseline, seeds=[42], train_steps=5, eval_episodes=4)
    assert hasattr(rep, "step0_g4_tok_acc")
    assert hasattr(rep, "step1_g4_tok_acc")
    assert hasattr(rep, "step2_g4_tok_acc")
    assert rep.best_step_count in [0, 1, 2]
    assert compute_model_hash(frozen_baseline) == EXPECTED_WEIGHT_HASH


def test_step294_strict_i4_reevaluation(frozen_baseline):
    """Step 294: Verify strict I4 evaluation across seeds 42, 101, 2026."""
    rep = run_strict_i4_reevaluation(frozen_baseline, seeds=[42, 101, 2026], train_steps=4, eval_episodes_per_split=4)
    assert set(rep.per_seed_results.keys()) == {42, 101, 2026}
    assert hasattr(rep, "mean_g4_tok_acc")
    assert hasattr(rep, "i4_gate_passed")
    assert rep.is_base_bit_exact is True
    assert compute_model_hash(frozen_baseline) == EXPECTED_WEIGHT_HASH


def test_step295_three_hop_eligibility_analysis(frozen_baseline):
    """Step 295: Verify conditional 3-hop execution or focused failure boundary analysis."""
    cand = ChakrMicroCompositionalReasoningModel(frozen_baseline, rank=16)
    rep = evaluate_three_hop_eligibility_or_failure_analysis(cand, i4_gate_passed=False, seed=42, num_episodes=4)
    assert rep.three_hop_executed is False
    assert len(rep.remaining_architectural_limitation) > 10
    assert compute_model_hash(frozen_baseline) == EXPECTED_WEIGHT_HASH


def test_step296_master_wave296_benchmark_summary(frozen_baseline):
    """Step 296: Verify complete master consolidated report."""
    rep = run_master_wave296_benchmark(seed=42, train_steps_quick=4, eval_episodes_quick=4)
    assert rep.baseline_integrity is True
    assert rep.baseline_params == 3443136
    assert rep.baseline_sha == EXPECTED_WEIGHT_HASH
    assert rep.contamination_collisions == 0
    assert rep.final_classification in ["I4_ACHIEVED", "I4_EMERGING", "I4_NOT_ACHIEVED"]
    assert compute_model_hash(frozen_baseline) == EXPECTED_WEIGHT_HASH
