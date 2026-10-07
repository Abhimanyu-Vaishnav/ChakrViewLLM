"""Tests for Steps 281-288: Neural Compositional Reasoning & Iterative Binding Wave."""

from __future__ import annotations

import pytest
import torch

from chakrview.brain.model import ChakrMicro
from chakrview.runtime.interactive import (
    instantiate_frozen_baseline,
    compute_model_hash,
    EXPECTED_WEIGHT_HASH,
)
from chakrview.cognition.compositional_benchmark_environment import (
    CompositionalAssociativeEnvironment,
)
from chakrview.cognition.intermediate_state_extraction import (
    trace_compositional_execution,
)
from chakrview.cognition.two_hop_composition_architecture import (
    CompositionalBridgeProjector,
    ChakrMicroCompositionalReasoningModel,
    compute_module_sha256,
)
from chakrview.cognition.iterative_binding_interventions import (
    evaluate_iterative_binding_interventions,
)
from chakrview.cognition.composition_generalization_training import (
    run_compositional_training_and_evaluation,
)
from chakrview.cognition.composition_robustness_suite import (
    evaluate_compositional_robustness,
)
from chakrview.cognition.three_hop_stress_test import (
    run_three_hop_stress_test,
)
from scripts.run_step288_benchmark import run_master_compositional_benchmark


@pytest.fixture(scope="module")
def frozen_baseline():
    return instantiate_frozen_baseline()


def test_step281_compositional_environment(frozen_baseline):
    """Step 281: Verify 2-hop compositional episode generation & position metadata."""
    env = CompositionalAssociativeEnvironment(seed=42)
    ep = env.generate_episode("train", num_distractors=1, episode_idx=1)

    assert ep.chain.start_key != ep.chain.intermediate_val
    assert ep.chain.intermediate_val != ep.chain.final_val
    assert len(ep.prompt_tokens) > 5
    assert ep.intermediate_token > 0
    assert ep.target_token > 0
    assert ep.hop1_key_pos >= 0
    assert ep.hop1_val_pos >= 0
    assert ep.hop2_key_pos >= 0
    assert ep.hop2_val_pos >= 0


def test_step282_intermediate_state_extraction(frozen_baseline):
    """Step 282: Verify intermediate representation tracing through hops 1 and 2."""
    env = CompositionalAssociativeEnvironment(seed=42)
    ep = env.generate_episode("train", num_distractors=0, episode_idx=2)
    model = ChakrMicroCompositionalReasoningModel(frozen_baseline, rank=16)

    trace = trace_compositional_execution(model, ep, env.tok)
    assert trace.episode_id == ep.episode_id
    assert hasattr(trace, "hop1_key_matched_correctly")
    assert hasattr(trace, "intermediate_state_similarity")
    assert hasattr(trace, "hop2_key_matched_correctly")
    assert hasattr(trace, "final_token_correct")


def test_step283_two_hop_composition_architecture(frozen_baseline):
    """Step 283: Verify candidate architecture parameters, bridge module, and baseline freeze."""
    pre_hash = compute_model_hash(frozen_baseline)
    model = ChakrMicroCompositionalReasoningModel(frozen_baseline, rank=16, d_bind=64)

    base_params = sum(p.numel() for p in frozen_baseline.parameters())
    adapter_params = sum(p.numel() for p in model.adapter.parameters())
    bridge_params = sum(p.numel() for p in model.bridge.parameters())
    binding_params = sum(p.numel() for p in model.binding.parameters())

    assert base_params == 3443136
    assert adapter_params == 6529
    # LayerNorm(192*2) + Linear(192*192) + gate(1) = 384 + 36864 + 1 = 37249
    assert bridge_params == 37249
    assert binding_params == 25345

    tot_trainable = adapter_params + bridge_params + binding_params
    assert tot_trainable == 69123

    b_sha = compute_module_sha256(model.bridge)
    assert len(b_sha) == 64

    post_hash = compute_model_hash(frozen_baseline)
    assert pre_hash == EXPECTED_WEIGHT_HASH
    assert post_hash == EXPECTED_WEIGHT_HASH


def test_step284_iterative_binding_interventions(frozen_baseline):
    """Step 284: Verify causal interventions on intermediate representations."""
    model = ChakrMicroCompositionalReasoningModel(frozen_baseline, rank=16)
    res = evaluate_iterative_binding_interventions(model, seed=42, num_samples=4)
    assert res.num_samples == 4
    assert hasattr(res, "corrupted_intermediate_prob_drop")
    assert hasattr(res, "swapped_intermediate_tracking_rate")
    assert compute_model_hash(frozen_baseline) == EXPECTED_WEIGHT_HASH


def test_step285_composition_generalization(frozen_baseline):
    """Step 285: Verify training loop and G1-G4 split evaluations."""
    candidate, rep = run_compositional_training_and_evaluation(
        frozen_baseline, train_steps=5, eval_episodes_per_split=4, seed=42
    )
    for sp_key in ["G1_known_known", "G2_unseen_identities", "G3_unseen_composition", "G4_unseen_unseen"]:
        assert sp_key in rep.splits
        assert hasattr(rep.splits[sp_key], "final_token_acc")
        assert hasattr(rep.splits[sp_key], "intermediate_state_acc")

    assert rep.is_base_frozen is True
    assert compute_model_hash(frozen_baseline) == EXPECTED_WEIGHT_HASH


def test_step286_distractor_and_permutation_robustness(frozen_baseline):
    """Step 286: Verify 10 perturbation stress conditions and contamination audit."""
    model = ChakrMicroCompositionalReasoningModel(frozen_baseline, rank=16)
    rep = evaluate_compositional_robustness(model, seed=42, episodes_per_condition=4)
    assert rep.contamination_count == 0
    assert rep.contamination_free is True
    assert len(rep.conditions) >= 8
    assert compute_model_hash(frozen_baseline) == EXPECTED_WEIGHT_HASH


def test_step287_three_hop_stress_test(frozen_baseline):
    """Step 287: Verify 3-hop composition chain execution and boundary detection."""
    model = ChakrMicroCompositionalReasoningModel(frozen_baseline, rank=16)
    res = run_three_hop_stress_test(model, seed=42, num_episodes=4)
    assert res.num_episodes == 4
    assert hasattr(res, "hop1_routing_acc")
    assert hasattr(res, "hop2_routing_acc")
    assert hasattr(res, "hop3_routing_acc")
    assert hasattr(res, "first_failure_boundary")
    assert compute_model_hash(frozen_baseline) == EXPECTED_WEIGHT_HASH


def test_step288_master_compositional_benchmark_summary(frozen_baseline):
    """Step 288: Verify complete master decision benchmark runs and checks categories A-AA."""
    summary = run_master_compositional_benchmark(seed=42, train_steps_quick=5, eval_episodes_quick=4)
    assert summary.all_integrity_passed is True
    assert summary.baseline_param_count == 3443136
    assert summary.baseline_sha256 == EXPECTED_WEIGHT_HASH
    assert summary.categories["V_contamination_audit"]["collisions"] == 0
    assert summary.categories["AA_baseline_immutability"]["passed"] is True
    assert compute_model_hash(frozen_baseline) == EXPECTED_WEIGHT_HASH
