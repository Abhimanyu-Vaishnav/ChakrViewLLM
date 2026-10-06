"""Tests for Steps 257-264: Neural Representation Adaptation Wave."""

from __future__ import annotations

import pytest
import torch

from chakrview.brain.model import ChakrMicro
from chakrview.runtime.interactive import (
    instantiate_frozen_baseline,
    compute_model_hash,
    EXPECTED_WEIGHT_HASH,
)
from chakrview.cognition.neural_representation_adapter import (
    CompactResidualAdapter,
    GatedResidualAdapter,
    ChakrMicroWithAdaptedRepresentations,
    compute_module_sha256,
)
from chakrview.cognition.identity_invariant_objective import (
    IdentityInvariantRepresentationLoss,
)
from chakrview.cognition.adapter_curriculum_training import (
    run_adapter_training_curriculum,
)
from chakrview.cognition.representation_generalization_test import (
    evaluate_representation_generalization,
)
from chakrview.cognition.multiseed_learnability_experiment import (
    run_multiseed_learnability,
)
from chakrview.cognition.anti_memorization_evaluation import (
    evaluate_anti_memorization,
)
from chakrview.cognition.adapter_pointer_integration import (
    run_adapter_pointer_integration,
)
from scripts.run_step264_benchmark import run_master_adaptation_benchmark


@pytest.fixture(scope="module")
def frozen_baseline():
    return instantiate_frozen_baseline()


def test_step257_adapter_architecture(frozen_baseline):
    """Step 257: Verify adapter architectures, parameters, and baseline immutability."""
    pre_hash = compute_model_hash(frozen_baseline)
    cand_gated = ChakrMicroWithAdaptedRepresentations(frozen_baseline, adapter_type="gated", rank=32)

    base_params = sum(p.numel() for p in frozen_baseline.parameters())
    adapter_params = sum(p.numel() for p in cand_gated.adapter.parameters())

    assert base_params == 3443136
    assert adapter_params == 12673
    assert sum(p.numel() for p in cand_gated.parameters()) == 3443136 + 12673

    ad_hash = compute_module_sha256(cand_gated.adapter)
    assert len(ad_hash) == 64

    # Forward pass tensor contracts
    inp = torch.tensor([[68, 69, 70, 71]], dtype=torch.long)
    h_ad, h_raw, logits = cand_gated.forward_adapted_backbone(inp)
    assert h_ad.shape == (1, 4, 192)
    assert h_raw.shape == (1, 4, 192)
    assert logits.shape == (1, 4, 4096)

    # Baseline hash invariant
    assert compute_model_hash(frozen_baseline) == pre_hash == EXPECTED_WEIGHT_HASH


def test_step258_identity_invariant_objective():
    """Step 258: Verify identity-invariant role loss computation."""
    loss_fn = IdentityInvariantRepresentationLoss()
    dummy_hidden = torch.randn(1, 10, 192)
    dummy_logits = torch.randn(1, 10, 4096)

    res = loss_fn(
        adapted_hidden=dummy_hidden,
        logits=dummy_logits,
        query_key_pos=8,
        matching_key_pos=2,
        distractor_key_positions=[4, 6],
        associated_val_pos=3,
        target_token=52,
    )

    assert res.total_loss.ndim == 0
    assert -1.0 <= res.positive_role_alignment <= 1.0
    assert isinstance(res.negative_key_separation, float)


def test_step259_adapter_curriculum_training(frozen_baseline):
    """Step 259: Verify 7-phase curriculum execution and language retention."""
    pre_hash = compute_model_hash(frozen_baseline)
    cand, rep = run_adapter_training_curriculum(
        base_model=frozen_baseline,
        adapter_type="gated",
        rank=16,
        seed=42,
        steps_per_phase=2,
        eval_episodes_per_phase=2,
    )
    post_hash = compute_model_hash(frozen_baseline)

    assert pre_hash == post_hash == EXPECTED_WEIGHT_HASH
    assert rep.is_base_frozen is True
    assert len(rep.phases) == 7
    assert rep.language_retention_ratio <= 1.05


def test_step260_representation_generalization(frozen_baseline):
    """Step 260: Verify representation generalization across all 4 splits."""
    cand = ChakrMicroWithAdaptedRepresentations(frozen_baseline, adapter_type="gated", rank=16)
    rep = evaluate_representation_generalization(cand, seed=42, num_episodes_per_split=3)

    assert len(rep.splits) == 4
    for sp in ["known_known", "known_unseen", "unseen_known", "unseen_unseen"]:
        assert sp in rep.splits
        m = rep.splits[sp]
        assert 0.0 <= m.key_selection_accuracy <= 1.0
        assert 0.0 <= m.val_position_accuracy <= 1.0
        assert 0.0 <= m.final_token_accuracy <= 1.0


def test_step261_multiseed_learnability(frozen_baseline):
    """Step 261: Verify multi-seed learnability evaluation across seeds 42, 101, 2026."""
    rep = run_multiseed_learnability(
        base_model=frozen_baseline,
        seeds=[42, 101, 2026],
        adapter_type="gated",
        rank=16,
        steps_per_phase=2,
        eval_episodes=2,
    )

    assert len(rep.seeds_tested) == 3
    assert rep.is_base_immutable is True
    assert 0.0 <= rep.mean_unseen_unseen_key_acc <= 1.0
    assert 0.0 <= rep.mean_unseen_unseen_val_acc <= 1.0


def test_step262_anti_memorization_evaluation(frozen_baseline):
    """Step 262: Verify 8 anti-shortcut tests and contamination audit."""
    cand = ChakrMicroWithAdaptedRepresentations(frozen_baseline, adapter_type="gated", rank=16)
    rep = evaluate_anti_memorization(cand, seed=42, episodes_per_condition=2)

    assert len(rep.condition_results) == 8
    assert rep.contamination_free is True
    assert rep.contamination_count == 0
    assert rep.learned_property in ("ROLE_RELATIONSHIP", "TOKEN_POSITION_SHORTCUT")


def test_step263_adapter_pointer_integration(frozen_baseline):
    """Step 263: Verify comparative integration across Candidates A, B, and C."""
    rep = run_adapter_pointer_integration(
        base_model=frozen_baseline,
        seed=42,
        train_steps=2,
        eval_episodes=2,
        rank=16,
    )

    assert rep.baseline_param_count == 3443136
    assert rep.baseline_sha256 == EXPECTED_WEIGHT_HASH
    assert set(rep.candidates.keys()) == {
        "cand_A_pointer_only",
        "cand_B_adapter_only",
        "cand_C_adapter_plus_pointer",
    }


def test_step264_master_adaptation_benchmark():
    """Step 264: Verify master decision gate benchmark across all categories A through X."""
    res = run_master_adaptation_benchmark(train_steps_quick=2, eval_episodes_quick=2)

    assert res.all_integrity_passed is True
    assert res.baseline_param_count == 3443136
    assert res.baseline_sha256 == EXPECTED_WEIGHT_HASH
    assert len(res.categories) == 24
    assert res.adaptation_verdict in (
        "REPRESENTATION ADAPTATION WORKS",
        "REPRESENTATION ADAPTATION PARTIALLY WORKS",
        "REPRESENTATION ADAPTATION FAILS",
    )
    assert len(res.strongest_conclusion) > 0
