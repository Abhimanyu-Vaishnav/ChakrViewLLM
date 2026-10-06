"""Tests for Steps 265-272: Neural Contextual Vocabulary Emission Wave."""

from __future__ import annotations

import pytest
import torch

from chakrview.brain.model import ChakrMicro
from chakrview.runtime.interactive import (
    instantiate_frozen_baseline,
    compute_model_hash,
    EXPECTED_WEIGHT_HASH,
)
from chakrview.cognition.contextual_vocabulary_readout import (
    ContextualVocabularyReadout,
    ChakrMicroWithContextualReadout,
    compute_module_sha256,
)
from chakrview.cognition.learned_contextual_emission import (
    run_contextual_emission_training,
)
from chakrview.cognition.disjoint_vocabulary_emission import (
    evaluate_disjoint_vocabulary_emission,
)
from chakrview.cognition.pointer_vs_readout_comparison import (
    run_pointer_vs_readout_comparison,
)
from chakrview.cognition.emission_anti_memorization import (
    evaluate_emission_anti_memorization,
)
from chakrview.cognition.emission_multiseed_validation import (
    run_emission_multiseed_validation,
)
from chakrview.cognition.emission_mechanism_verification import (
    run_emission_mechanism_verification,
)
from scripts.run_step272_benchmark import run_master_emission_benchmark


@pytest.fixture(scope="module")
def frozen_baseline():
    return instantiate_frozen_baseline()


def test_step265_contextual_vocabulary_readout(frozen_baseline):
    """Step 265: Verify readout architecture, parameters, and baseline immutability."""
    pre_hash = compute_model_hash(frozen_baseline)
    model = ChakrMicroWithContextualReadout(frozen_baseline, rank=16)

    base_params = sum(p.numel() for p in frozen_baseline.parameters())
    readout_params = sum(p.numel() for p in model.readout.parameters())
    adapter_params = sum(p.numel() for p in model.adapter.parameters())

    assert base_params == 3443136
    assert readout_params == 823872
    assert adapter_params == 6529
    assert sum(p.numel() for p in model.parameters()) == 3443136 + 6529 + 823872

    r_hash = compute_module_sha256(model.readout)
    assert len(r_hash) == 64

    # Baseline untouched
    post_hash = compute_model_hash(frozen_baseline)
    assert pre_hash == EXPECTED_WEIGHT_HASH
    assert post_hash == EXPECTED_WEIGHT_HASH


def test_step266_learned_contextual_emission(frozen_baseline):
    """Step 266: Verify multi-phase emission curriculum runs and preserves baseline."""
    model, rep = run_contextual_emission_training(frozen_baseline, steps_per_phase=5, eval_episodes_per_phase=4, seed=42)
    assert len(rep.phases) == 6
    assert rep.is_base_frozen is True
    assert compute_model_hash(frozen_baseline) == EXPECTED_WEIGHT_HASH


def test_step267_disjoint_vocabulary_emission(frozen_baseline):
    """Step 267: Verify 4-condition disjoint emission evaluation."""
    model = ChakrMicroWithContextualReadout(frozen_baseline, rank=16)
    rep = evaluate_disjoint_vocabulary_emission(model, num_episodes_per_condition=4, seed=42)

    for cond_key in ["A_known_known", "B_known_unseen", "C_unseen_known", "D_unseen_unseen"]:
        assert cond_key in rep.conditions
        cond = rep.conditions[cond_key]
        assert hasattr(cond, "key_position_accuracy")
        assert hasattr(cond, "value_position_accuracy")
        assert hasattr(cond, "final_token_accuracy")

    assert compute_model_hash(frozen_baseline) == EXPECTED_WEIGHT_HASH


def test_step268_pointer_vs_readout_comparison(frozen_baseline):
    """Step 268: Verify comparison across Candidates A, B, C, D."""
    results = run_pointer_vs_readout_comparison(frozen_baseline, eval_episodes=4, train_steps=5, seed=42)
    for c_id in ["CANDIDATE_A", "CANDIDATE_B", "CANDIDATE_C", "CANDIDATE_D"]:
        assert c_id in results.candidates
    assert results.candidates["CANDIDATE_B"].baseline_exact is True
    assert compute_model_hash(frozen_baseline) == EXPECTED_WEIGHT_HASH


def test_step269_anti_memorization_audit(frozen_baseline):
    """Step 269: Verify shortcut destruction & contamination auditing."""
    model = ChakrMicroWithContextualReadout(frozen_baseline, rank=16)
    results = evaluate_emission_anti_memorization(model, episodes_per_condition=4, seed=42)
    assert results.contamination_count == 0
    assert results.contamination_free is True
    assert len(results.conditions) == 9
    assert compute_model_hash(frozen_baseline) == EXPECTED_WEIGHT_HASH


def test_step270_multiseed_validation(frozen_baseline):
    """Step 270: Verify 3-seed validation runs across seeds 42, 101, 2026."""
    res = run_emission_multiseed_validation(frozen_baseline, steps_per_phase=4, eval_episodes=4)
    assert set(res.seeds_tested) == {42, 101, 2026}
    assert 42 in res.per_seed
    assert 101 in res.per_seed
    assert 2026 in res.per_seed
    assert compute_model_hash(frozen_baseline) == EXPECTED_WEIGHT_HASH


def test_step271_emission_interventions(frozen_baseline):
    """Step 271: Verify causal interventions trace correctly."""
    model = ChakrMicroWithContextualReadout(frozen_baseline, rank=16)
    res = run_emission_mechanism_verification(model, num_samples=4, seed=42)
    assert res.num_samples == 4
    assert hasattr(res, "logit_drop_on_wrong_value")
    assert hasattr(res, "readout_ablation_drop")
    assert compute_model_hash(frozen_baseline) == EXPECTED_WEIGHT_HASH


def test_step272_master_benchmark_summary(frozen_baseline):
    """Step 272: Verify full benchmark runs and outputs structured categories A-AC."""
    summary = run_master_emission_benchmark(seed=42, train_steps_quick=5, eval_episodes_quick=4)
    assert summary.all_integrity_passed is True
    assert summary.baseline_param_count == 3443136
    assert summary.baseline_sha256 == EXPECTED_WEIGHT_HASH
    assert summary.categories["U_contamination_audit"]["collisions"] == 0
    assert summary.categories["AC_baseline_immutability"]["passed"] is True
    assert compute_model_hash(frozen_baseline) == EXPECTED_WEIGHT_HASH
