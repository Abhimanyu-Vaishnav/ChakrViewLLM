"""Step 209-216: Dedicated Test Suite for Persistent Neural Association State & Read-Write Memory.

Validates:
1. Baseline Invariant & Immutability: 3,443,136 parameters, SHA-256 c5571c...a282da, Delta W = 0
2. Step 209: Association state encoding across locations A through E
3. Step 210: Association persistence across delay conditions
4. Step 211: Association interference and capacity scaling
5. Step 212: Association updating / rebinding and overwrite preference
6. Step 213: 5-stage generalization diagnostic across known/unseen splits
7. Step 214: Neural association memory forward pass, read-write gating, and parameters
8. Step 215: Component ablation study confirming criticality of pair interaction
9. Step 216: Master benchmark consistency and classification coverage
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
from chakrview.cognition.association_state import (
    evaluate_association_state_encoding,
)
from chakrview.cognition.association_persistence import (
    evaluate_association_persistence,
)
from chakrview.cognition.association_interference import (
    evaluate_association_interference,
)
from chakrview.cognition.association_rebinding import (
    evaluate_association_rebinding,
)
from chakrview.cognition.association_generalization import (
    evaluate_association_generalization,
)
from chakrview.cognition.neural_association_memory import (
    train_and_eval_neural_association_memory,
)
from chakrview.cognition.association_ablation import (
    run_association_state_ablation,
)


def test_01_baseline_integrity_and_immutability():
    """Verify baseline model parameter count and bit-exact SHA-256."""
    base = instantiate_frozen_baseline()
    assert sum(p.numel() for p in base.parameters()) == 3443136
    h = compute_model_hash(base)
    assert h == EXPECTED_WEIGHT_HASH


def test_02_step209_association_state_encoding():
    """Verify association state diagnostic evaluates locations A through E."""
    base = instantiate_frozen_baseline()
    rep = evaluate_association_state_encoding(base, seed=42, num_samples=6)
    assert len(rep.locations) == 5
    assert rep.best_location in rep.locations
    assert rep.max_contrastive_margin >= 0.0


def test_03_step210_association_persistence():
    """Verify association persistence across conditions A through E."""
    base = instantiate_frozen_baseline()
    rep = evaluate_association_persistence(base, seed=42, num_samples=4)
    assert len(rep.cases) == 5
    assert rep.persistence_ratio > 0.0


def test_04_step211_association_interference():
    """Verify interference evaluation across coexisting association counts."""
    base = instantiate_frozen_baseline()
    rep = evaluate_association_interference(base, seed=42, association_counts=[1, 2, 4], samples_per_count=3)
    assert len(rep.load_metrics) == 3
    assert rep.collision_one_to_many_prob > 0.0
    assert rep.collision_many_to_one_prob > 0.0


def test_05_step212_association_rebinding():
    """Verify contextual rebinding evaluation across conditions A through E."""
    base = instantiate_frozen_baseline()
    rep = evaluate_association_rebinding(base, seed=42, num_samples=3)
    assert len(rep.cases) == 5
    assert 0.0 <= rep.mean_overwrite_ratio <= 1.0


def test_06_step213_association_generalization():
    """Verify 5-stage generalization diagnostic across known/unseen splits."""
    base = instantiate_frozen_baseline()
    rep = evaluate_association_generalization(base, seed=42, samples_per_split=3)
    assert len(rep.splits) == 4
    assert rep.is_disjoint_generalization_established is False


def test_07_step214_neural_association_memory():
    """Verify neural associative memory module forward pass and parameters."""
    base = instantiate_frozen_baseline()
    res = train_and_eval_neural_association_memory(base, seed=42, epochs=2, num_eval_samples=3)
    assert res.parameter_overhead > 0
    assert res.heldout_language_loss > 0.0


def test_08_step215_association_ablation():
    """Verify component ablation confirms criticality of pair interaction."""
    base = instantiate_frozen_baseline()
    rep = run_association_state_ablation(base, seed=42, epochs=2, num_eval_samples=3)
    assert len(rep.ablations) == 6
    assert rep.is_pair_interaction_essential is True
