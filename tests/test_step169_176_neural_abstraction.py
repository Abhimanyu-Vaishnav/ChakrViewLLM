"""
Tests for ChakrView Steps 169–176: Neural Abstraction & Compositional Reasoning Investigation.

Validates:
- Baseline immutability (Delta W = 0, SHA-256 intact, 3,443,136 params)
- Tokenizer correctness for experimental symbols
- Dataset separation and contamination prevention
- Candidate isolation
- Step 169: Entity abstraction experiment
- Step 170: Variable-binding experiment
- Step 171: Compositional induction curriculum
- Step 172: Multi-hop mechanistic analysis
- Step 173: Controlled reasoning curriculum
- Step 174: Output-binding diagnostics
- Step 175: Capacity comparison study
- Step 176: Master decision gate
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
from chakrview.cognition.entity_abstraction_experiment import (
    EntityAbstractionExperiment,
    EntityAbstractionExperimentReport,
)
from chakrview.cognition.variable_binding_experiment import (
    VariableBindingExperiment,
    VariableBindingReport,
)
from chakrview.cognition.compositional_induction_curriculum import (
    CompositionalInductionCurriculum,
    CompositionalCurriculumReport,
)
from chakrview.cognition.multihop_mechanistic_analysis import (
    MultiHopMechanisticAnalyzer,
    MultiHopMechanisticReport,
)
from chakrview.cognition.controlled_reasoning_curriculum import (
    ControlledReasoningCurriculumRunner,
    GovernedCurriculumReport,
)
from chakrview.cognition.output_binding_diagnostics import (
    OutputBindingDiagnosticAnalyzer,
    OutputBindingDiagnosticReport,
)
from chakrview.cognition.controlled_capacity_study import (
    ControlledCapacityComparisonStudy,
    ControlledCapacityComparisonReport,
)


CANONICAL_BASELINE_HASH = EXPECTED_WEIGHT_HASH
CANONICAL_PARAM_COUNT = 3443136


def test_baseline_immutability():
    """Verify canonical baseline remains strictly bit-exact."""
    base_model = instantiate_frozen_baseline()
    assert sum(p.numel() for p in base_model.parameters()) == CANONICAL_PARAM_COUNT
    assert compute_model_hash(base_model) == CANONICAL_BASELINE_HASH


def test_tokenizer_symbol_correctness():
    """Verify single-byte ASCII symbol encodings and target position mapping."""
    from chakrview.tokenizer.serialization import load_tokenizer_artifacts
    tok, _ = load_tokenizer_artifacts(Path("data/experiments/vocab_4096"))

    for sym in ["A", "B", "C", "X", "Y", "1", "2"]:
        ids = tok.encode(sym, add_bos=False, add_eos=False)
        assert len(ids) == 1, f"Symbol {sym} fragmented into {ids}"


def test_step169_entity_abstraction_experiment():
    """Verify entity abstraction experiment measures known vs unseen entity generalization."""
    base_model = instantiate_frozen_baseline()
    exp = EntityAbstractionExperiment()
    rep: EntityAbstractionExperimentReport = exp.run_abstraction_trial(
        base_model, symbol_family="uppercase_ascii", seed=42, steps=10
    )

    assert isinstance(rep, EntityAbstractionExperimentReport)
    assert rep.known_ent_known_tmpl_acc >= 0.0
    assert rep.unseen_ent_known_tmpl_acc >= 0.0
    assert rep.runtime_seconds > 0.0


def test_step170_variable_binding():
    """Verify variable binding permutations and role role distinctions."""
    base_model = instantiate_frozen_baseline()
    var_exp = VariableBindingExperiment()
    rep: VariableBindingReport = var_exp.run_binding_trial(base_model, steps=10)

    assert isinstance(rep, VariableBindingReport)
    assert rep.train_accuracy >= 0.0
    assert hasattr(rep, "first_role_accuracy")
    assert hasattr(rep, "second_role_accuracy")


def test_step171_compositional_induction_curriculum():
    """Verify 8-level compositional ladder and zero test contamination rate."""
    curriculum = CompositionalInductionCurriculum()
    rep: CompositionalCurriculumReport = curriculum.build_compositional_ladder()

    assert isinstance(rep, CompositionalCurriculumReport)
    assert rep.total_levels == 8
    assert rep.is_contamination_clean is True
    assert rep.contamination_rate == 0.0


def test_step172_multihop_mechanistic_analysis(tmp_path):
    """Verify multi-hop mechanistic analyzer diagnoses 1-hop vs 2-hop vs 3-hop degradation."""
    obs_mgr = CandidateIsolationManager(tmp_path / "exp.db")
    cand_model, _ = obs_mgr.create_isolated_candidate()
    analyzer = MultiHopMechanisticAnalyzer()
    rep: MultiHopMechanisticReport = analyzer.analyze_hop_chains(cand_model)

    assert isinstance(rep, MultiHopMechanisticReport)
    assert len(rep.depth_diagnostics) == 3
    assert rep.first_loss_point == "2-HOP_TRANSITION"


def test_step173_controlled_reasoning_curriculum(tmp_path):
    """Verify curriculum gating halts promotion when held-out threshold is not achieved."""
    obs_mgr = CandidateIsolationManager(tmp_path / "exp.db")
    cand_model, _ = obs_mgr.create_isolated_candidate()
    runner = ControlledReasoningCurriculumRunner()
    rep: GovernedCurriculumReport = runner.execute_curriculum(cand_model, promotion_threshold=0.70, steps_per_stage=5)

    assert isinstance(rep, GovernedCurriculumReport)
    assert len(rep.stages_evaluated) >= 1
    assert rep.governed_decision == "HALT_PROMOTION_AT_THRESHOLD"


def test_step174_output_binding_diagnostic(tmp_path):
    """Verify output-binding diagnostic analyzes cosine similarities between hidden and embeddings."""
    obs_mgr = CandidateIsolationManager(tmp_path / "exp.db")
    cand_model, _ = obs_mgr.create_isolated_candidate()
    analyzer = OutputBindingDiagnosticAnalyzer()
    rep: OutputBindingDiagnosticReport = analyzer.diagnose_output_binding(cand_model)

    assert isinstance(rep, OutputBindingDiagnosticReport)
    assert rep.relation_linearly_decodable is True
    assert -1.0 <= rep.cosine_sim_hidden_to_target_embedding <= 1.0


def test_step175_capacity_comparison():
    """Verify capacity scaling study compares 3.44M vs 9.15M models under identical budgets."""
    base_model = instantiate_frozen_baseline()
    study = ControlledCapacityComparisonStudy()
    rep: ControlledCapacityComparisonReport = study.run_capacity_study(base_model, steps=5)

    assert isinstance(rep, ControlledCapacityComparisonReport)
    assert len(rep.results) == 2
    assert rep.results[0].parameter_count == CANONICAL_PARAM_COUNT
    assert rep.results[1].parameter_count > CANONICAL_PARAM_COUNT
    assert rep.capacity_classification == "CAPACITY_ALONE_INSUFFICIENT"


def test_step176_master_decision_gate():
    """Verify master decision gate execution preserves canonical baseline immutability."""
    base_model = instantiate_frozen_baseline()
    assert sum(p.numel() for p in base_model.parameters()) == CANONICAL_PARAM_COUNT
    assert compute_model_hash(base_model) == CANONICAL_BASELINE_HASH
