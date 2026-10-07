"""Tests for Steps 273-280: Dynamic Contextual Token Binding Wave."""

from __future__ import annotations

import pytest
import torch

from chakrview.brain.model import ChakrMicro
from chakrview.runtime.interactive import (
    instantiate_frozen_baseline,
    compute_model_hash,
    EXPECTED_WEIGHT_HASH,
)
from chakrview.cognition.dynamic_contextual_token_binding import (
    DynamicContextualTokenBinding,
    ChakrMicroWithDynamicBinding,
    compute_module_sha256,
)
from chakrview.cognition.contextual_token_candidates import (
    extract_contextual_candidates_from_episode,
)
from chakrview.cognition.dynamic_emission_training import (
    run_dynamic_emission_training,
)
from chakrview.cognition.true_disjoint_generalization import (
    evaluate_true_disjoint_binding,
)
from chakrview.cognition.dynamic_binding_anti_memorization import (
    evaluate_dynamic_binding_anti_memorization,
)
from chakrview.cognition.dynamic_binding_interventions import (
    run_dynamic_binding_interventions,
)
from chakrview.cognition.dynamic_multiseed_comparison import (
    run_dynamic_multiseed_comparison,
)
from scripts.run_step280_benchmark import run_master_dynamic_binding_benchmark


@pytest.fixture(scope="module")
def frozen_baseline():
    return instantiate_frozen_baseline()


def test_step273_dynamic_binding_module(frozen_baseline):
    """Step 273: Verify dynamic binding module architecture, parameters, and baseline immutability."""
    pre_hash = compute_model_hash(frozen_baseline)
    model = ChakrMicroWithDynamicBinding(frozen_baseline, rank=16, d_bind=64)

    base_params = sum(p.numel() for p in frozen_baseline.parameters())
    binding_params = sum(p.numel() for p in model.binding.parameters())
    adapter_params = sum(p.numel() for p in model.adapter.parameters())

    assert base_params == 3443136
    assert adapter_params == 6529
    # query_norm (192*2) + query_proj (192*64) + key_norm (192*2) + key_proj (192*64) + scale (1)
    # 384 + 12288 + 384 + 12288 + 1 = 25345
    assert binding_params == 25345
    assert sum(p.numel() for p in model.parameters()) == 3443136 + 6529 + 25345

    b_hash = compute_module_sha256(model.binding)
    assert len(b_hash) == 64

    # Baseline untouched
    post_hash = compute_model_hash(frozen_baseline)
    assert pre_hash == EXPECTED_WEIGHT_HASH
    assert post_hash == EXPECTED_WEIGHT_HASH


def test_step274_contextual_candidate_extraction(frozen_baseline):
    """Step 274: Verify candidate extraction directly from prompt hidden states."""
    from chakrview.cognition.randomized_associative_episodes import RandomizedAssociativeEnvironment
    env = RandomizedAssociativeEnvironment(seed=42)
    ep = env.generate_episode("train", 2, episode_idx=1)
    model = ChakrMicroWithDynamicBinding(frozen_baseline, rank=16)

    inp = torch.tensor([ep.prompt_tokens], dtype=torch.long)
    h_ad, _ = model.forward_backbone(inp)
    bundle = extract_contextual_candidates_from_episode(ep, h_ad, env.tok)

    assert len(bundle.candidate_positions) >= 2
    assert bundle.candidate_states.shape == (1, len(bundle.candidate_positions), 192)
    assert bundle.target_candidate_idx is not None
    assert bundle.candidate_mask.shape == (1, len(bundle.candidate_positions))


def test_step275_dynamic_emission_training(frozen_baseline):
    """Step 275: Verify training curriculum runs and preserves baseline."""
    model, rep = run_dynamic_emission_training(
        frozen_baseline, steps_per_phase=4, eval_episodes_per_phase=4, seed=42
    )
    assert len(rep.phases) == 6
    assert rep.is_base_frozen is True
    assert compute_model_hash(frozen_baseline) == EXPECTED_WEIGHT_HASH


def test_step276_true_disjoint_generalization(frozen_baseline):
    """Step 276: Verify 4-condition disjoint evaluation."""
    model = ChakrMicroWithDynamicBinding(frozen_baseline, rank=16)
    rep = evaluate_true_disjoint_binding(model, num_episodes_per_condition=4, seed=42)

    for cond_key in ["A_known_known", "B_known_unseen", "C_unseen_known", "D_unseen_unseen"]:
        assert cond_key in rep.conditions
        cond = rep.conditions[cond_key]
        assert hasattr(cond, "key_routing_accuracy")
        assert hasattr(cond, "value_routing_accuracy")
        assert hasattr(cond, "candidate_token_accuracy")
        assert hasattr(cond, "final_token_accuracy")

    assert compute_model_hash(frozen_baseline) == EXPECTED_WEIGHT_HASH


def test_step277_anti_memorization_audit(frozen_baseline):
    """Step 277: Verify shortcut destruction, candidate order permutation, & contamination auditing."""
    model = ChakrMicroWithDynamicBinding(frozen_baseline, rank=16)
    results = evaluate_dynamic_binding_anti_memorization(model, episodes_per_condition=4, seed=42)
    assert results.contamination_count == 0
    assert results.contamination_free is True
    assert len(results.conditions) == 13
    assert compute_model_hash(frozen_baseline) == EXPECTED_WEIGHT_HASH


def test_step278_dynamic_binding_interventions(frozen_baseline):
    """Step 278: Verify causal interventions trace correctly."""
    model = ChakrMicroWithDynamicBinding(frozen_baseline, rank=16)
    res = run_dynamic_binding_interventions(model, num_samples=4, seed=42)
    assert res.num_samples == 4
    assert hasattr(res, "wrong_cand_prob_drop")
    assert hasattr(res, "binding_ablation_drop")
    assert compute_model_hash(frozen_baseline) == EXPECTED_WEIGHT_HASH


def test_step279_multiseed_candidate_comparison(frozen_baseline):
    """Step 279: Verify comparison across Candidates A, B, C, D and seeds 42, 101, 2026."""
    results = run_dynamic_multiseed_comparison(
        frozen_baseline, seeds=[42, 101, 2026], train_steps_per_phase=3, eval_episodes=4
    )
    for c_id in ["CANDIDATE_A", "CANDIDATE_B", "CANDIDATE_C", "CANDIDATE_D"]:
        assert c_id in results.candidate_comparison
    assert results.candidate_comparison["CANDIDATE_C"].baseline_exact is True
    assert set(results.seeds_tested) == {42, 101, 2026}
    assert compute_model_hash(frozen_baseline) == EXPECTED_WEIGHT_HASH


def test_step280_master_benchmark_summary(frozen_baseline):
    """Step 280: Verify full benchmark runs and outputs structured categories A-AG."""
    summary = run_master_dynamic_binding_benchmark(seed=42, train_steps_quick=3, eval_episodes_quick=4)
    assert summary.all_integrity_passed is True
    assert summary.baseline_param_count == 3443136
    assert summary.baseline_sha256 == EXPECTED_WEIGHT_HASH
    assert summary.categories["X_contamination_audit"]["collisions"] == 0
    assert summary.categories["AG_baseline_immutability"]["passed"] is True
    assert compute_model_hash(frozen_baseline) == EXPECTED_WEIGHT_HASH
