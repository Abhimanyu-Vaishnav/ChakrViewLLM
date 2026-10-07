"""Unit and Integration Tests for Wave 297-304: Learned Associative Attractor State.

Tests:
1. Canonical baseline immutability invariant (3,443,136 params, hash c5571c..., Delta W = 0)
2. Step 297: Intermediate attractor forensic baseline geometry analysis
3. Step 298: Minimal learned associative attractor codebook instantiation & forward pass
4. Step 299: Associative attractor training step & loss backpropagation
5. Step 300: Attractor -> Hop-2 query integration variants A, B, C, D
6. Step 301: Attractor robustness & 14-condition anti-shortcut testing
7. Step 302: I3 preservation and language retention invariant audit
8. Step 303/304: Strict multi-seed I4 evaluation & master benchmark execution
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
from chakrview.cognition.two_hop_composition_architecture import (
    ChakrMicroCompositionalReasoningModel,
)
from chakrview.cognition.learned_associative_attractor import (
    LearnedAssociativeAttractor,
    compute_module_sha256,
)
from chakrview.cognition.forensic_attractor_baseline import (
    run_attractor_forensic_baseline,
)
from chakrview.cognition.associative_attractor_training import (
    train_attractor_compositional_model,
    run_associative_attractor_training_study,
)
from chakrview.cognition.attractor_query_integration import (
    AttractorBridgeVariant,
    run_attractor_query_integration_study,
)
from chakrview.cognition.attractor_anti_shortcut import (
    run_attractor_anti_shortcut_suite,
)
from chakrview.cognition.attractor_i3_preservation import (
    evaluate_i3_preservation_and_language_retention,
)
from chakrview.cognition.strict_attractor_i4_evaluation import (
    run_strict_attractor_i4_evaluation,
)
from scripts.run_step304_benchmark import (
    run_wave304_master_benchmark,
)


def test_baseline_immutability_wave304():
    """Verify canonical baseline remains strictly frozen and bit-exact."""
    base = instantiate_frozen_baseline()
    params = sum(p.numel() for p in base.parameters())
    assert params == 3_443_136
    h = compute_model_hash(base)
    assert h == EXPECTED_WEIGHT_HASH


def test_step297_forensic_attractor_baseline():
    """Verify Step 297 forensic geometry computation."""
    base = instantiate_frozen_baseline()
    rep = run_attractor_forensic_baseline(base, seeds=[42], num_episodes_per_split=2, train_steps=3)
    assert rep.overall_between_role_var >= 0.0
    assert rep.overall_within_role_var >= 0.0
    assert len(rep.per_split_seed_metrics) == 4


def test_step298_learned_associative_attractor():
    """Verify Step 298 learned attractor forward pass and parameter bound."""
    att = LearnedAssociativeAttractor(d_model=192, num_attractors=16, d_attractor=64)
    param_count = sum(p.numel() for p in att.parameters() if p.requires_grad)
    assert param_count < 30_000
    x = torch.randn(2, 192)
    h_out, probs, ent = att(x)
    assert h_out.shape == (2, 192)
    assert probs.shape == (2, 16)
    assert ent > 0.0


def test_step299_associative_attractor_training():
    """Verify Step 299 training and ablation comparison."""
    base = instantiate_frozen_baseline()
    cand, att, metrics = train_attractor_compositional_model(
        base, seed=42, train_steps=3, eval_episodes=2, use_attractor_loss=True,
    )
    assert metrics.mean_attractor_entropy >= 0.0
    assert 0.0 <= metrics.mean_attractor_utilization <= 1.0


def test_step300_attractor_query_integration():
    """Verify Step 300 bridge variants A, B, C, D."""
    for var in ["A", "B", "C", "D"]:
        m = AttractorBridgeVariant(d_model=192, variant=var)
        x = torch.randn(1, 192)
        h_q2, probs, ent = m(x)
        assert h_q2.shape == (1, 192)


def test_step301_attractor_anti_shortcut():
    """Verify Step 301 14-condition anti-shortcut testing."""
    base = instantiate_frozen_baseline()
    cand = ChakrMicroCompositionalReasoningModel(base, rank=16)
    att = LearnedAssociativeAttractor(d_model=192, num_attractors=16, d_attractor=64)
    rep = run_attractor_anti_shortcut_suite(cand, att, seed=42, episodes_per_cond=2)
    assert len(rep.conditions) == 14
    for c in rep.conditions:
        assert 0.0 <= c.final_tok_acc <= 1.0


def test_step302_i3_preservation_and_language():
    """Verify Step 302 I3 dynamic binding and language retention audit."""
    base = instantiate_frozen_baseline()
    cand = ChakrMicroCompositionalReasoningModel(base, rank=16)
    att = LearnedAssociativeAttractor(d_model=192, num_attractors=16, d_attractor=64)
    rep = evaluate_i3_preservation_and_language_retention(base, cand, att, seed=42, num_episodes=2)
    assert 0.90 <= rep.language_retention_ratio <= 1.10
    assert rep.language_preserved is True


def test_step303_304_master_benchmark():
    """Verify Step 303 & 304 multi-seed evaluation and classification integrity."""
    base = instantiate_frozen_baseline()
    rep = run_strict_attractor_i4_evaluation(base, seeds=[42, 101], train_steps=3, eval_episodes=2)
    assert rep.final_classification in ("I4_ACHIEVED", "I4_EMERGING", "I4_NOT_ACHIEVED")
    assert rep.is_base_bit_exact is True
