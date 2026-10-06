"""Tests for Steps 249-256: Associative Pointer Architecture & I3 Capability Gate Wave."""

from __future__ import annotations

import pytest
import torch

from chakrview.brain.model import ChakrMicro
from chakrview.runtime.interactive import (
    instantiate_frozen_baseline,
    compute_model_hash,
    EXPECTED_WEIGHT_HASH,
)
from chakrview.cognition.associative_pointer_circuit import (
    AssociativePointerHead,
    ChakrMicroWithAssociativePointer,
    compute_module_parameter_hash,
)
from chakrview.cognition.randomized_associative_episodes import (
    RandomizedAssociativeEnvironment,
)
from chakrview.cognition.associative_pointer_training import (
    run_pointer_circuit_curriculum,
)
from chakrview.cognition.value_position_routing_test import (
    evaluate_value_position_routing,
)
from chakrview.cognition.positional_heuristic_destruction import (
    evaluate_positional_heuristic_destruction,
)
from chakrview.cognition.disjoint_identity_evaluation import (
    run_disjoint_identity_generalization,
)
from chakrview.cognition.associative_pointer_integration import (
    run_architecture_integration_comparison,
)
from scripts.run_step256_benchmark import run_master_decision_benchmark


@pytest.fixture(scope="module")
def frozen_baseline():
    return instantiate_frozen_baseline()


def test_step249_associative_pointer_architecture(frozen_baseline):
    """Step 249: Verify modular associative pointer architecture and parameter contracts."""
    pre_hash = compute_model_hash(frozen_baseline)
    cand = ChakrMicroWithAssociativePointer(frozen_baseline)

    base_params = sum(p.numel() for p in frozen_baseline.parameters())
    cand_params = sum(p.numel() for p in cand.parameters())
    head_params = sum(p.numel() for p in cand.pointer_head.parameters())

    assert base_params == 3443136
    assert head_params == 569409
    assert cand_params == 3443136 + 569409 == 4012545

    head_hash = compute_module_parameter_hash(cand.pointer_head)
    assert len(head_hash) == 64

    # Forward pass tensor shape contracts
    dummy_input = torch.tensor([[68, 69, 70, 71]], dtype=torch.long)
    logits, out = cand(dummy_input)

    assert logits.shape == (1, 4096)
    assert out.key_distribution.shape == (1, 3)
    assert out.value_distribution.shape == (1, 3)
    assert out.copy_prob.shape == (1, 1)
    assert out.gen_prob.shape == (1, 1)

    # Immutability
    post_hash = compute_model_hash(frozen_baseline)
    assert pre_hash == post_hash == EXPECTED_WEIGHT_HASH


def test_step250_randomized_associative_episodes():
    """Step 250: Verify randomized multi-pair episode generator and contamination audit."""
    env = RandomizedAssociativeEnvironment(seed=42)

    # 1. Episodes across variable counts and layouts
    for n in [1, 2, 3]:
        ep = env.generate_episode("train", num_associations=n, episode_idx=n)
        assert ep.num_associations <= n
        assert ep.expected_value in [p.val for p in ep.pairs]
        assert ep.query_key in [p.key for p in ep.pairs]
        assert len(ep.prompt_tokens) > 0
        assert 0 <= ep.query_key_pos < len(ep.prompt_tokens)
        assert 0 <= ep.matching_key_pos < len(ep.prompt_tokens)
        assert 0 <= ep.associated_val_pos < len(ep.prompt_tokens)

    # 2. Layout diversity
    for lay in env.LAYOUTS:
        ep = env.generate_episode("train", num_associations=2, layout_name=lay)
        assert ep.layout_name == lay

    # 3. Contamination audit
    train_batch = env.generate_batch(10, split="train", num_associations=2)
    eval_batch = env.generate_batch(10, split="disjoint_test", num_associations=2)
    clean, coll = env.verify_no_contamination(train_batch, eval_batch)
    assert clean is True
    assert coll == 0


def test_step251_pointer_circuit_training(frozen_baseline):
    """Step 251: Verify controlled curriculum training and base model retention."""
    pre_hash = compute_model_hash(frozen_baseline)
    cand, rep = run_pointer_circuit_curriculum(
        frozen_baseline,
        seed=42,
        steps_per_phase=3,
        eval_episodes=3,
    )
    post_hash = compute_model_hash(frozen_baseline)

    assert pre_hash == post_hash == EXPECTED_WEIGHT_HASH
    assert rep.is_base_frozen is True
    assert len(rep.phases) == 6
    assert rep.language_retention_ratio <= 1.05


def test_step252_value_position_routing(frozen_baseline):
    """Step 252: Verify value-position routing test and 2x2 matrix calculation."""
    cand = ChakrMicroWithAssociativePointer(frozen_baseline)
    res = evaluate_value_position_routing(cand, seed=42, num_episodes=5, num_associations=3)

    assert res.num_episodes == 5
    assert res.chance_key_acc == pytest.approx(1.0 / 3.0, rel=1e-3)
    assert 0.0 <= res.key_selection_accuracy <= 1.0
    assert 0.0 <= res.value_selection_accuracy <= 1.0
    assert 0.0 <= res.final_token_accuracy <= 1.0
    # 2x2 matrix rates sum to 1.0
    total_rate = (
        res.key_corr_val_corr_rate
        + res.key_corr_val_incorr_rate
        + res.key_incorr_val_corr_rate
        + res.key_incorr_val_incorr_rate
    )
    assert pytest.approx(total_rate, rel=1e-3) == 1.0


def test_step253_positional_heuristic_destruction(frozen_baseline):
    """Step 253: Verify layout permutation destruction test and shortcut detection."""
    cand = ChakrMicroWithAssociativePointer(frozen_baseline)
    rep = evaluate_positional_heuristic_destruction(cand, seed=42, samples_per_layout=3)

    assert len(rep.layout_results) == 5
    assert rep.routing_nature in ("RELATIONSHIP_DRIVEN", "POSITIONAL_SHORTCUT")
    assert isinstance(rep.is_shortcut_dependent, bool)


def test_step254_disjoint_identity_generalization(frozen_baseline):
    """Step 254: Verify multi-seed disjoint evaluation across orthogonal identity splits."""
    rep = run_disjoint_identity_generalization(
        frozen_baseline,
        seeds=[42, 101, 2026],
        train_steps_per_seed=3,
        eval_episodes_per_split=3,
    )

    assert len(rep.seeds_tested) == 3
    assert rep.is_base_immutable is True
    assert 0.0 <= rep.mean_known_known_acc <= 1.0
    assert 0.0 <= rep.mean_unseen_unseen_acc <= 1.0
    assert rep.i3_status in ("I3_CANDIDATE_ACHIEVED", "I3_EMERGING", "I3_NOT_ACHIEVED")
    assert rep.i3_promoted == (rep.mean_unseen_unseen_acc >= 0.50)


def test_step255_architecture_integration_comparison(frozen_baseline):
    """Step 255: Verify comparative integration across candidate architectures."""
    rep = run_architecture_integration_comparison(
        frozen_baseline,
        seed=42,
        train_steps=3,
        eval_episodes=3,
    )

    assert rep.baseline_param_count == 3443136
    assert rep.baseline_sha256 == EXPECTED_WEIGHT_HASH
    assert set(rep.candidates.keys()) == {
        "1_pointer_only",
        "2_chakrmicro_plus_pointer",
        "3_associative_matching_pointer",
    }
    for c_met in rep.candidates.values():
        assert c_met.baseline_exact is True
        assert c_met.language_retention_ratio <= 1.05


def test_step256_master_decision_gate():
    """Step 256: Verify master decision gate benchmark across all categories A through U."""
    res = run_master_decision_benchmark(train_steps_quick=2, eval_episodes_quick=2)

    assert res.all_integrity_passed is True
    assert res.baseline_param_count == 3443136
    assert res.baseline_sha256 == EXPECTED_WEIGHT_HASH
    assert len(res.categories) == 21
    assert "strongest_candidate_name" in res.__dict__
    assert "remaining_blocker" in res.__dict__
    assert "next_wave_decision" in res.__dict__
