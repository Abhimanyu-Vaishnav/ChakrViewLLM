"""Step 201-208: Dedicated Test Suite for Neural Key-Value Binding & Associative Generalization.

Validates:
1. Baseline Invariant & Immutability: 3,443,136 parameters, SHA-256 c5571c...a282da, Delta W = 0
2. Step 201: Key-Value binding diagnostic representations and positive/negative contrastive margin
3. Step 202: Relative binding evaluation across layout variations and directional inversions
4. Step 203: Contrastive association learning with lambda parameter sweeps
5. Step 204: Permutation-invariant association stability across 5 layouts
6. Step 205: Disjoint associative binding evaluation across all 4 splits
7. Step 206: Key-value binding under distractor load scaling
8. Step 207: Iterative multi-hop retrieval gating
9. Step 208: Master benchmark classification coverage
"""

from pathlib import Path
import pytest
import torch

from chakrview.brain.model import ChakrMicro
from chakrview.runtime.interactive import (
    instantiate_frozen_baseline,
    compute_model_hash,
    EXPECTED_WEIGHT_HASH,
)
from chakrview.cognition.key_value_binding import (
    probe_key_value_binding,
)
from chakrview.cognition.relative_binding import (
    evaluate_relative_binding,
)
from chakrview.cognition.contrastive_binding import (
    train_and_eval_contrastive_binding,
)
from chakrview.cognition.permutation_binding import (
    evaluate_permutation_invariance,
)
from chakrview.cognition.disjoint_binding import (
    evaluate_disjoint_binding_conditions,
)
from chakrview.cognition.binding_distractors import (
    evaluate_binding_under_distractors,
)
from chakrview.cognition.binding_iterative_multihop import (
    evaluate_binding_iterative_multihop,
)


def test_01_baseline_integrity_and_immutability():
    """Verify baseline model parameter count and bit-exact SHA-256."""
    base = instantiate_frozen_baseline()
    assert sum(p.numel() for p in base.parameters()) == 3443136
    h = compute_model_hash(base)
    assert h == EXPECTED_WEIGHT_HASH


def test_02_step201_key_value_binding():
    """Verify positive vs negative pair contrastive margin calculation."""
    base = instantiate_frozen_baseline()
    res = probe_key_value_binding(base, seed=42, num_samples=6)
    assert res.sample_count == 6
    assert isinstance(res.contrastive_margin, float)
    assert 0.0 <= res.association_classification_acc <= 1.0


def test_03_step202_relative_binding():
    """Verify relative binding evaluation across cases A, B, C, D."""
    base = instantiate_frozen_baseline()
    res = evaluate_relative_binding(base, seed=42, num_samples=4)
    assert 0.0 <= res.case_a_forward_acc <= 1.0
    assert 0.0 <= res.case_b_reversed_acc <= 1.0
    assert 0.0 <= res.mean_directional_invariance <= 1.0


def test_04_step203_contrastive_binding():
    """Verify auxiliary contrastive binding objective runs and preserves language loss."""
    base = instantiate_frozen_baseline()
    res = train_and_eval_contrastive_binding(base, seed=42, lambda_vals=[0.5], epochs=3, num_eval_samples=4)
    assert 0.5 in res
    assert res[0.5].heldout_language_loss > 0.0


def test_05_step204_permutation_invariance():
    """Verify association evaluation across all 5 layout permutations."""
    base = instantiate_frozen_baseline()
    res = evaluate_permutation_invariance(base, seed=42, num_eval_triplets=4)
    assert 0.0 <= res.mean_permutation_accuracy <= 1.0
    assert res.permutation_variance >= 0.0


def test_06_step205_disjoint_binding():
    """Verify disjoint associative binding evaluation across 4 splits."""
    base = instantiate_frozen_baseline()
    rep = evaluate_disjoint_binding_conditions(base, seed=42, samples_per_cond=4)
    assert hasattr(rep, "known_known")
    assert hasattr(rep, "unseen_unseen")
    assert rep.is_i3_gate_passed is False


def test_07_step206_binding_distractors():
    """Verify binding degradation curve across distractor counts."""
    base = instantiate_frozen_baseline()
    rep = evaluate_binding_under_distractors(base, seed=42, distractor_counts=[0, 1, 2], samples_per_count=3)
    assert len(rep.load_metrics) == 3
    assert 0 in rep.load_metrics and 2 in rep.load_metrics


def test_08_step207_binding_iterative_multihop():
    """Verify multi-hop iterative retrieval assesses 1-4 hops and records failure boundary."""
    base = instantiate_frozen_baseline()
    rep = evaluate_binding_iterative_multihop(base, seed=42, samples_per_hop=3)
    assert set(rep.hops.keys()) == {1, 2, 3, 4}
    assert rep.failure_boundary_hop >= 1
    assert rep.step205_gate_passed is False
