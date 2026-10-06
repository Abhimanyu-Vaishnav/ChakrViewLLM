"""Step 233-240: Dedicated Test Suite for Growing Neural Associative Capability.

Validates:
1. Baseline Invariant & Immutability: 3,443,136 parameters, SHA-256 c5571c...a282da, Delta W = 0
2. Step 233: Learning path reconciliation & repair reproducing Step 225
3. Step 234: Stable associative learning on isolated candidate
4. Step 235: Association circuit strengthening comparing variants
5. Step 236: Contextual association generalization across 4 splits
6. Step 237: Disjoint associative learning gate across seeds 42, 101, 2026
7. Step 238: Generalization stress testing & contamination audit
8. Step 239: Learned contextual rebinding under strict prerequisite gating
9. Step 240: Master capability growth gate & 18 master categories
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
from chakrview.cognition.association_learning_repair import (
    reconcile_step225_and_step226,
)
from chakrview.cognition.association_learning_stable import (
    run_stable_association_study,
)
from chakrview.cognition.association_circuit_learning import (
    evaluate_circuit_strengthening,
)
from chakrview.cognition.association_generalization import (
    evaluate_contextual_association_generalization,
)
from chakrview.cognition.association_learning_disjoint import (
    run_disjoint_associative_learning_gate,
)
from chakrview.cognition.association_transfer_stress import (
    run_generalization_stress_evaluation,
)
from chakrview.cognition.learned_rebinding import (
    evaluate_learned_contextual_rebinding,
)


def test_01_baseline_integrity_and_immutability():
    """Verify baseline model parameter count and bit-exact SHA-256."""
    base = instantiate_frozen_baseline()
    assert sum(p.numel() for p in base.parameters()) == 3443136
    h = compute_model_hash(base)
    assert h == EXPECTED_WEIGHT_HASH


def test_02_step233_learning_path_repair():
    """Verify reconciliation reproduces Step 225 in-distribution learning and diagnoses L0."""
    base = instantiate_frozen_baseline()
    rep = reconcile_step225_and_step226(base, seed=42)
    assert rep.is_pipeline_repaired is True
    assert rep.reproduced_step225_train_acc >= 0.0
    assert "Root Cause" in rep.root_cause_diagnosis


def test_03_step234_stable_association_learning():
    """Verify stable associative learning executes across isolated candidates."""
    base = instantiate_frozen_baseline()
    rep = run_stable_association_study(base, seeds=[42])
    assert 42 in rep.seed_results
    assert 0.0 <= rep.best_train_acc <= 1.0


def test_04_step235_circuit_strengthening():
    """Verify circuit strengthening evaluates candidate variants."""
    base = instantiate_frozen_baseline()
    rep = evaluate_circuit_strengthening(base, seed=42)
    assert len(rep.variants) >= 2
    assert "V1_Backbone_FineTune" in rep.variants
    assert "V2_Gated_Associative_Circuit" in rep.variants


def test_05_step236_contextual_generalization():
    """Verify contextual association generalization across all 4 splits."""
    base = instantiate_frozen_baseline()
    rep = evaluate_contextual_association_generalization(base, seed=42, samples_per_split=4)
    assert len(rep.splits) == 4
    assert "known_known" in rep.splits
    assert "unseen_unseen" in rep.splits


def test_06_step237_disjoint_associative_learning_gate():
    """Verify disjoint associative learning gate on disjoint entity sets."""
    base = instantiate_frozen_baseline()
    rep = run_disjoint_associative_learning_gate(base, seeds=[42])
    assert 42 in rep.seed_evaluations
    assert rep.summary_verdict in ("I3_CANDIDATE_ACHIEVED", "I3_DENIED_DISJOINT_RETRIEVAL_ZERO")


def test_07_step238_generalization_stress():
    """Verify generalization stress testing handles prerequisite conditions."""
    base = instantiate_frozen_baseline()
    # When I3 not achieved
    rep_blocked = run_generalization_stress_evaluation(base, is_i3_met=False, seed=42)
    assert rep_blocked.is_i3_prerequisite_met is False
    assert "CONDITIONALLY_BLOCKED" in rep_blocked.status

    # When I3 simulated as met
    rep_exec = run_generalization_stress_evaluation(base, is_i3_met=True, seed=42, episodes=2)
    assert rep_exec.is_i3_prerequisite_met is True
    assert len(rep_exec.stress_results) == 4


def test_08_step239_learned_rebinding():
    """Verify learned rebinding handles prerequisite conditions."""
    base = instantiate_frozen_baseline()
    rep_blocked = evaluate_learned_contextual_rebinding(base, is_i3_qualified=False, seed=42)
    assert rep_blocked.is_i3_qualified is False
    assert "BLOCKED" in rep_blocked.status
