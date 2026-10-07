"""Unit and Integration Tests for Wave 321-328: Integrated Recurrent State-Transition Neural Core.

Tests:
1. Canonical baseline immutability invariant (3,443,136 params, hash c5571c..., Delta W = 0)
2. Step 321: Integrated state-transition forensics layer-wise diagnostic
3. Step 322: Minimal recurrent state-transition mechanism forward pass and zero identity initialization
4. Step 323: State integration with attention comparison (Configurations A, B, C)
5. Step 324: Two-step recurrent composition & 5 causal interventions
6. Step 325: Distractor-robust state transition across [0, 1, 2, 3, 5] distractors
7. Step 326: Generalization & 14-condition anti-shortcut audit
8. Step 327/328: Strict multi-seed I3 preservation, I4 evaluation, and master benchmark execution
"""

from __future__ import annotations

import pytest
import torch

from chakrview.brain.model import ChakrMicro
from chakrview.runtime.interactive import (
    instantiate_frozen_baseline,
    compute_model_hash,
    EXPECTED_WEIGHT_HASH,
)
from chakrview.cognition.recurrent_state_transition import (
    GatedRecurrentStateTransition,
    compute_module_sha256,
)
from chakrview.cognition.integrated_state_transition_forensics import (
    run_integrated_state_transition_forensics,
)
from chakrview.cognition.state_attention_integration import (
    StateAttentionIntegrationModel,
    run_state_integration_comparison,
)
from chakrview.cognition.recurrent_causal_interventions import (
    run_recurrent_causal_interventions,
)
from chakrview.cognition.distractor_robust_state import (
    evaluate_distractor_robust_state,
)
from chakrview.cognition.recurrent_generalization_suite import (
    run_recurrent_anti_shortcut_suite,
)
from chakrview.cognition.strict_recurrent_i4_evaluation import (
    train_recurrent_core_candidate,
    run_strict_recurrent_i3_i4_evaluation,
)
from chakrview.cognition.compositional_benchmark_environment import (
    CompositionalAssociativeEnvironment,
)


def test_baseline_immutability_wave328():
    """Verify canonical baseline remains strictly frozen and bit-exact."""
    base = instantiate_frozen_baseline()
    params = sum(p.numel() for p in base.parameters())
    assert params == 3_443_136
    h = compute_model_hash(base)
    assert h == EXPECTED_WEIGHT_HASH


def test_step321_state_transition_forensics():
    """Verify Step 321 forensics diagnostic."""
    base = instantiate_frozen_baseline()
    rep = run_integrated_state_transition_forensics(base, seeds=[42], num_episodes=2)
    assert len(rep.per_layer_metrics) == 6
    assert rep.best_insertion_layer in range(6)


def test_step322_recurrent_state_transition_module():
    """Verify Step 322 GRU-like gated transition forward pass and no-op init."""
    trans = GatedRecurrentStateTransition(d_model=192, d_state=64)
    params = sum(p.numel() for p in trans.parameters() if p.requires_grad)
    assert params < 80_000
    h = torch.randn(2, 192)
    s0 = trans.get_initial_state(2)
    s_next, h_out = trans.transition(h, s0)
    assert s_next.shape == (2, 64)
    assert h_out.shape == (2, 192)
    diff = float(torch.norm(h_out - h).item())
    assert diff < 1e-6


def test_step323_state_integration_comparison():
    """Verify Step 323 integration configurations A, B, C."""
    base = instantiate_frozen_baseline()
    for m in ["A", "B", "C"]:
        model = StateAttentionIntegrationModel(base, integration_mode=m)
        v = torch.randn(1, 192)
        s0 = model.transition_core.get_initial_state(1)
        q2, s1 = model.compute_second_hop_query(v, s0)
        assert q2.shape == (1, 192)


def test_step324_recurrent_causal_interventions():
    """Verify Step 324 causal state interventions."""
    base = instantiate_frozen_baseline()
    model = StateAttentionIntegrationModel(base, integration_mode="A")
    env = CompositionalAssociativeEnvironment(seed=42)
    rep = run_recurrent_causal_interventions(model, env, num_episodes=2, seed=42)
    assert 0.0 <= rep.normal_tok_acc <= 1.0
    assert 0.0 <= rep.zeroed_tok_acc <= 1.0


def test_step325_distractor_robust_state():
    """Verify Step 325 distractor robustness suite."""
    base = instantiate_frozen_baseline()
    model = StateAttentionIntegrationModel(base, integration_mode="A")
    rep = evaluate_distractor_robust_state(model, seed=42, episodes_per_scale=2)
    assert len(rep.scale_metrics) == 5


def test_step326_recurrent_anti_shortcut():
    """Verify Step 326 14-condition anti-shortcut testing."""
    base = instantiate_frozen_baseline()
    model = StateAttentionIntegrationModel(base, integration_mode="A")
    rep = run_recurrent_anti_shortcut_suite(model, seed=42, episodes_per_cond=2)
    assert len(rep.conditions) == 14
    for c in rep.conditions:
        assert 0.0 <= c.final_tok_acc <= 1.0


def test_step327_328_strict_evaluation_and_comparisons():
    """Verify Step 327 & 328 multi-seed evaluation, 3-way comparisons, and baseline immutability."""
    base = instantiate_frozen_baseline()
    rep = run_strict_recurrent_i3_i4_evaluation(base, seeds=[42, 101], train_steps=2, eval_episodes=2)
    assert rep.final_classification in ("I4_ACHIEVED", "I4_EMERGING", "I4_NOT_ACHIEVED")
    assert len(rep.comparisons) == 3
    assert rep.is_base_bit_exact is True
