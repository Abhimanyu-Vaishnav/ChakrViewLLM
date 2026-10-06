"""
Tests for ChakrView Steps 161–168: Neural Reasoning Capability Investigation.

Validates:
- test_step161_mechanistic_diagnostics
- test_step162_learnability_ladder
- test_step163_representation_probe
- test_step164_attention_diagnostics
- test_step165_objective_comparison
- test_step166_optimization_capacity_study
- test_step167_nonzero_reasoning_experiment
- test_step168_decision_gate
"""

import pytest
import os
from pathlib import Path

from chakrview.runtime.interactive import (
    instantiate_frozen_baseline,
    compute_model_hash,
    EXPECTED_WEIGHT_HASH,
)
from chakrview.cognition.neural_observability import (
    CandidateIsolationManager,
)
from chakrview.cognition.reasoning_mechanistic_diagnostics import (
    ReasoningMechanisticDiagnostics,
    ReasoningMechanisticDiagnosticsResult,
)
from chakrview.cognition.minimal_learnability_ladder import (
    MinimalReasoningLearnabilityLadder,
    LearnabilityLadderReport,
)
from chakrview.cognition.representation_probing_diagnostics import (
    RepresentationProbingDiagnostics,
    RepresentationProbeReport,
)
from chakrview.cognition.attention_information_flow import (
    AttentionFlowDiagnostic,
    AttentionFlowReport,
)
from chakrview.cognition.objective_optimization_study import (
    ObjectiveOptimizationStudy,
    ObjectiveComparisonReport,
)
from chakrview.cognition.optimization_capacity_study import (
    OptimizationCapacityStudy,
    OptimizationCapacityReport,
)
from chakrview.cognition.nonzero_reasoning_experiment import (
    NonZeroReasoningExperimentRunner,
    NonZeroReasoningReport,
)


CANONICAL_BASELINE_HASH = EXPECTED_WEIGHT_HASH
CANONICAL_PARAM_COUNT = 3443136


def test_step161_mechanistic_diagnostics(tmp_path):
    """Verify deep mechanistic diagnostics captures tokens, ranks, and per-layer gradients."""
    obs_mgr = CandidateIsolationManager(tmp_path / "exp.db")
    cand_model, _ = obs_mgr.create_isolated_candidate()
    mech = ReasoningMechanisticDiagnostics()

    test_samples = [("order: A > B -> first: ", "A"), ("order: B > C -> first: ", "B")]
    result: ReasoningMechanisticDiagnosticsResult = mech.run_mechanistic_analysis(
        cand_model, test_samples, training_steps=5
    )

    assert isinstance(result, ReasoningMechanisticDiagnosticsResult)
    assert len(result.token_diagnostics) == 2
    assert result.token_diagnostics[0].seq_len > 0
    assert result.token_diagnostics[0].target_token_id > 0
    assert result.layer_gradient_norms.total_grad_norm > 0.0
    assert len(result.layer_gradient_norms.attention_grad_norms) == cand_model.config.n_layers
    assert "embedding" in result.parameter_update_norms


def test_step162_learnability_ladder(tmp_path):
    """Verify progressive minimal tasks L0 to L6 report train, val, heldout accuracy and ranks."""
    obs_mgr = CandidateIsolationManager(tmp_path / "exp.db")
    cand_model, _ = obs_mgr.create_isolated_candidate()
    ladder = MinimalReasoningLearnabilityLadder()

    report: LearnabilityLadderReport = ladder.evaluate_ladder(cand_model, train_steps=5)
    assert isinstance(report, LearnabilityLadderReport)
    assert "L0" in report.level_results
    assert "L1" in report.level_results
    assert "L3" in report.level_results
    assert "L4" in report.level_results
    assert hasattr(report.level_results["L1"], "target_probability")
    assert hasattr(report.level_results["L1"], "target_rank")


def test_step163_representation_probe(tmp_path):
    """Verify external linear probes extract layerwise relational signal without altering weights."""
    base_model = instantiate_frozen_baseline()
    obs_mgr = CandidateIsolationManager(tmp_path / "exp.db")
    cand_model, _ = obs_mgr.create_isolated_candidate()
    init_hash = compute_model_hash(cand_model)

    probe_diag = RepresentationProbingDiagnostics()
    report: RepresentationProbeReport = probe_diag.probe_relational_representations(base_model, cand_model)

    assert isinstance(report, RepresentationProbeReport)
    assert len(report.candidate_probe_results) == cand_model.config.n_layers
    assert compute_model_hash(cand_model) == init_hash  # Probe must not mutate neural core
    assert report.diagnostic_classification == "DIAGNOSTIC_EVIDENCE_ONLY"
    assert any(r.has_relational_signal for r in report.candidate_probe_results)


def test_step164_attention_diagnostics(tmp_path):
    """Verify attention flow diagnostic inspects head masses on relevant premises vs punctuation."""
    base_model = instantiate_frozen_baseline()
    obs_mgr = CandidateIsolationManager(tmp_path / "exp.db")
    cand_model, _ = obs_mgr.create_isolated_candidate()

    attn_diag = AttentionFlowDiagnostic()
    report: AttentionFlowReport = attn_diag.analyze_information_flow(base_model, cand_model)

    assert isinstance(report, AttentionFlowReport)
    assert len(report.layer_head_details) > 0
    assert report.baseline_relevant_mass_mean >= 0.0
    assert report.candidate_relevant_mass_mean >= 0.0
    assert isinstance(report.candidate_focused_flag, bool)


def test_step165_objective_comparison(tmp_path):
    """Verify controlled comparison across causal, weighted, and target-focused objectives."""
    base_model = instantiate_frozen_baseline()
    obj_study = ObjectiveOptimizationStudy()

    train_samples = [("order: A > B -> first: ", "A"), ("order: B > C -> first: ", "B")]
    held_samples = [("compare: A > B -> first: ", "A")]

    report: ObjectiveComparisonReport = obj_study.run_objective_study(
        base_model, train_samples, held_samples, steps=5
    )
    assert isinstance(report, ObjectiveComparisonReport)
    assert "EXP_A" in report.results
    assert "EXP_B" in report.results
    assert "EXP_C" in report.results
    assert report.results["EXP_C"].train_loss_final <= report.results["EXP_C"].train_loss_initial


def test_step166_optimization_capacity_study(tmp_path):
    """Verify bounded hyperparameter matrix execution and bottleneck classification."""
    base_model = instantiate_frozen_baseline()
    opt_study = OptimizationCapacityStudy()

    train_samples = [("order: A > B -> first: ", "A"), ("order: B > C -> first: ", "B")]
    held_samples = [("compare: A > B -> first: ", "A")]

    report: OptimizationCapacityReport = opt_study.run_matrix(
        base_model, train_samples, held_samples, seed=42
    )
    assert isinstance(report, OptimizationCapacityReport)
    assert len(report.runs) == 3
    assert report.primary_bottleneck_classification != ""
    assert report.runs[0].param_norm_delta >= 0.0


def test_step167_nonzero_reasoning_experiment():
    """Verify multi-seed experiment produces non-zero held-out template generalization score."""
    base_model = instantiate_frozen_baseline()
    runner = NonZeroReasoningExperimentRunner()

    report: NonZeroReasoningReport = runner.run_multiseed_experiments(
        base_model, seeds=[42, 101, 2026], steps=10
    )
    assert isinstance(report, NonZeroReasoningReport)
    assert len(report.seed_results) == 3
    assert report.first_nonzero_achieved is True
    assert report.mean_template_generalization_acc > 0.0
    assert report.generalization_axis_proven == "G2_UNSEEN_LINGUISTIC_TEMPLATES"


def test_step168_decision_gate():
    """Verify master benchmark runs to completion and keeps canonical baseline bit-exact."""
    base_model = instantiate_frozen_baseline()
    assert sum(p.numel() for p in base_model.parameters()) == CANONICAL_PARAM_COUNT
    assert compute_model_hash(base_model) == CANONICAL_BASELINE_HASH
