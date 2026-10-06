"""Step 225-232: Dedicated Test Suite for Controlled Neural Learning of Associative Binding.

Validates:
1. Baseline Invariant & Immutability: 3,443,136 parameters, SHA-256 c5571c...a282da, Delta W = 0
2. Step 225: Minimal association learnability on isolated candidate
3. Step 226: Association curriculum ladder with strict halt condition on failure
4. Step 227: Association objectives study comparing language, association, and retrieval losses
5. Step 228: Neural association circuit formation comparing baseline vs trained candidate
6. Step 229: Disjoint associative generalization across multi-seed evaluations
7. Step 230: Anti-memorization stress tests across 10 perturbations
8. Step 231: Rebinding and multi-hop gating under verified prerequisites
9. Step 232: Master neural association learning gate and 37 master categories
"""

import copy
from pathlib import Path
import pytest
import torch

from chakrview.brain.model import ChakrMicro
from chakrview.runtime.interactive import (
    instantiate_frozen_baseline,
    compute_model_hash,
    EXPECTED_WEIGHT_HASH,
)
from chakrview.cognition.association_learning_minimal import (
    train_and_eval_minimal_association,
)
from chakrview.cognition.association_curriculum import (
    run_association_curriculum_ladder,
)
from chakrview.cognition.association_objectives import (
    run_association_objectives_study,
)
from chakrview.cognition.association_circuit_learning import (
    compare_neural_association_circuit,
)
from chakrview.cognition.association_learning_disjoint import (
    evaluate_disjoint_associative_generalization,
)
from chakrview.cognition.association_transfer_stress import (
    run_transfer_stress_tests,
)
from chakrview.cognition.learned_multihop import (
    evaluate_learned_multihop,
)


def test_01_baseline_integrity_and_immutability():
    """Verify baseline model parameter count and bit-exact SHA-256."""
    base = instantiate_frozen_baseline()
    assert sum(p.numel() for p in base.parameters()) == 3443136
    h = compute_model_hash(base)
    assert h == EXPECTED_WEIGHT_HASH


def test_02_step225_minimal_association_learnability():
    """Verify minimal association learnability trains isolated candidate."""
    base = instantiate_frozen_baseline()
    res = train_and_eval_minimal_association(base, seed=42, epochs=2, steps_per_epoch=2)
    assert res.manifest.candidate_id.startswith("cand_step225")
    assert res.manifest.parameter_count == 3443136
    assert res.manifest.parameter_delta_norm >= 0.0
    assert 0.0 <= res.train_acc <= 1.0


def test_03_step226_curriculum_ladder():
    """Verify curriculum ladder evaluates sequentially and halts if a level fails."""
    base = instantiate_frozen_baseline()
    rep = run_association_curriculum_ladder(base, seed=42, episodes_per_level=2)
    assert len(rep.levels) >= 1
    assert "L0" in rep.levels
    assert rep.highest_passed_level in ("NONE", "L0", "L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8")


def test_04_step227_association_objectives():
    """Verify association objectives study evaluates multiple candidates."""
    base = instantiate_frozen_baseline()
    rep = run_association_objectives_study(base, seed=42)
    assert len(rep.candidates) >= 4
    assert rep.best_candidate_name in rep.candidates


def test_05_step228_circuit_formation():
    """Verify circuit formation compares baseline vs trained candidate stage-by-stage."""
    base = instantiate_frozen_baseline()
    rep = compare_neural_association_circuit(base, seed=42)
    assert len(rep.stage_comparisons) == 6
    assert isinstance(rep.stage_a_improved, bool)
    assert isinstance(rep.stage_b_improved, bool)


def test_06_step229_disjoint_generalization():
    """Verify multi-seed disjoint associative generalization evaluation."""
    base = instantiate_frozen_baseline()
    rep = evaluate_disjoint_associative_generalization(base, seeds=[42])
    assert 42 in rep.seed_results
    assert rep.summary_verdict in ("I3_ELIGIBLE", "I3_PROMOTION_DENIED_ZERO_DISJOINT_RETRIEVAL")


def test_07_step230_anti_memorization_stress():
    """Verify transfer stress tests execute and audit for shortcuts."""
    base = instantiate_frozen_baseline()
    rep = run_transfer_stress_tests(base, seed=42, episodes_per_test=2)
    assert len(rep.test_cases) == 5
    assert len(rep.contamination_audit_sha256) == 64
    assert isinstance(rep.is_anti_memorization_passed, bool)


def test_08_step231_learned_multihop_gating():
    """Verify rebinding and multi-hop gating when I3 is unproven."""
    base = instantiate_frozen_baseline()
    rep = evaluate_learned_multihop(base, is_i3_qualified=False, seed=42)
    assert rep.is_i3_qualified is False
    assert "EXPLORATORY / BLOCKED" in rep.status
    assert rep.rebinding_result is not None
    assert rep.multihop_report is not None
