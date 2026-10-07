"""Unit and Integration Tests for Wave 313-320: Differentiable Relational Memory Architecture.

Tests:
1. Canonical baseline immutability invariant (3,443,136 params, hash c5571c..., Delta W = 0)
2. Step 313: Relational memory forensics baseline tracing
3. Step 314 & Step 315: Differentiable relational memory forward pass (S=1, S=2, S=4 slots)
4. Step 316: Sequential two-hop state update & 5 causal interventions
5. Step 317: Distractor resistance suite across [0, 1, 2, 3, 5] distractors
6. Step 318: Generalization & 14-condition anti-shortcut testing
7. Step 319: Strict multi-seed I3 preservation & I4 evaluation with ablations
8. Step 320: Master decision benchmark execution
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
from chakrview.cognition.relational_memory import (
    DifferentiableRelationalMemory,
    compute_module_sha256,
)
from chakrview.cognition.relational_memory_forensics import (
    run_relational_memory_forensics,
)
from chakrview.cognition.sequential_memory_update import (
    ChakrMicroWithRelationalMemory,
    run_sequential_state_interventions,
)
from chakrview.cognition.distractor_resistance_suite import (
    evaluate_distractor_resistance,
)
from chakrview.cognition.relational_memory_generalization import (
    run_relational_memory_anti_shortcut_suite,
)
from chakrview.cognition.strict_relational_memory_evaluation import (
    train_relational_memory_candidate,
    run_strict_relational_memory_i3_i4_evaluation,
)
from chakrview.cognition.compositional_benchmark_environment import (
    CompositionalAssociativeEnvironment,
)


def test_baseline_immutability_wave320():
    """Verify canonical baseline remains strictly frozen and bit-exact."""
    base = instantiate_frozen_baseline()
    params = sum(p.numel() for p in base.parameters())
    assert params == 3_443_136
    h = compute_model_hash(base)
    assert h == EXPECTED_WEIGHT_HASH


def test_step313_relational_memory_forensics():
    """Verify Step 313 forensics baseline calculation."""
    base = instantiate_frozen_baseline()
    rep = run_relational_memory_forensics(base, seeds=[42], num_episodes=2)
    assert rep.mean_distractor_cross_talk >= 0.0
    assert 42 in rep.per_seed_metrics


def test_step314_315_relational_memory_shapes():
    """Verify Step 314 & 315 relational memory shapes and parameter bounds."""
    for s in [1, 2, 4]:
        mem = DifferentiableRelationalMemory(d_model=192, num_slots=s, d_mem=64)
        params = sum(p.numel() for p in mem.parameters() if p.requires_grad)
        assert params < 60_000
        state = mem.get_initial_state(1)
        v = torch.randn(1, 192)
        state_up, w_w = mem.write(state, v)
        q2, r_w = mem.read(state_up, v)
        assert state_up.shape == (1, s, 64)
        assert q2.shape == (1, 192)


def test_step316_sequential_state_interventions():
    """Verify Step 316 causal state interventions."""
    base = instantiate_frozen_baseline()
    cand = ChakrMicroWithRelationalMemory(base, num_slots=2)
    env = CompositionalAssociativeEnvironment(seed=42)
    rep = run_sequential_state_interventions(cand, env, num_episodes=2, seed=42)
    assert 0.0 <= rep.normal_tok_acc <= 1.0
    assert 0.0 <= rep.zeroed_tok_acc <= 1.0


def test_step317_distractor_resistance():
    """Verify Step 317 distractor resistance suite."""
    base = instantiate_frozen_baseline()
    cand = ChakrMicroWithRelationalMemory(base, num_slots=2)
    rep = evaluate_distractor_resistance(cand, seed=42, episodes_per_scale=2)
    assert len(rep.scale_metrics) == 5


def test_step318_relational_memory_anti_shortcut():
    """Verify Step 318 14-condition anti-shortcut testing."""
    base = instantiate_frozen_baseline()
    cand = ChakrMicroWithRelationalMemory(base, num_slots=2)
    rep = run_relational_memory_anti_shortcut_suite(cand, seed=42, episodes_per_cond=2)
    assert len(rep.conditions) == 14
    for c in rep.conditions:
        assert 0.0 <= c.final_tok_acc <= 1.0


def test_step319_320_strict_evaluation_and_ablations():
    """Verify Step 319 & 320 multi-seed evaluation, ablations, and baseline immutability."""
    base = instantiate_frozen_baseline()
    rep = run_strict_relational_memory_i3_i4_evaluation(base, seeds=[42, 101], train_steps=2, eval_episodes=2, num_slots=2)
    assert rep.final_classification in ("I4_ACHIEVED", "I4_EMERGING", "I4_NOT_ACHIEVED")
    assert len(rep.ablations) >= 3
    assert rep.is_base_bit_exact is True
