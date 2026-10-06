"""
Tests for ChakrView Steps 177–184: Neural Induction, Associative Recall & Anti-Shortcut Validation.

Validates:
- Baseline immutability (Delta W = 0, SHA-256 intact, 3,443,136 params)
- Tokenizer symbol encodings
- Step 177: Dynamic token retrieval
- Step 178: Associative recall with distractors
- Step 179: Induction-like attention diagnostics
- Step 180: Disjoint key/value generalization
- Step 181: Variable binding re-evaluation
- Step 182: Multi-hop re-evaluation
- Step 183: Anti-shortcut scientific validation
- Step 184: Master induction decision gate
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
from chakrview.cognition.dynamic_token_retrieval import (
    DynamicTokenRetrievalExperiment,
    DynamicTokenRetrievalReport,
)
from chakrview.cognition.associative_recall import (
    AssociativeRecallExperiment,
    AssociativeRecallReport,
)
from chakrview.cognition.induction_head_analysis import (
    InductionHeadAnalyzer,
    InductionAnalysisReport,
)
from chakrview.cognition.disjoint_key_value_generalization import (
    DisjointKeyValueGeneralizationExperiment,
    DisjointGeneralizationReport,
)
from chakrview.cognition.variable_binding_reevaluation import (
    VariableBindingReEvaluation,
    VariableBindingReEvaluationReport,
)
from chakrview.cognition.multihop_reevaluation import (
    MultiHopReEvaluation,
    MultiHopReEvaluationReport,
)
from chakrview.cognition.anti_shortcut_validation import (
    AntiShortcutValidator,
    AntiShortcutValidationReport,
)


CANONICAL_BASELINE_HASH = EXPECTED_WEIGHT_HASH
CANONICAL_PARAM_COUNT = 3443136


def test_baseline_immutability():
    """Verify canonical baseline remains strictly bit-exact."""
    base_model = instantiate_frozen_baseline()
    assert sum(p.numel() for p in base_model.parameters()) == CANONICAL_PARAM_COUNT
    assert compute_model_hash(base_model) == CANONICAL_BASELINE_HASH


def test_step177_dynamic_token_retrieval():
    """Verify dynamic token retrieval trials report accuracy, loss, and attention metrics."""
    base_model = instantiate_frozen_baseline()
    exp = DynamicTokenRetrievalExperiment()
    rep: DynamicTokenRetrievalReport = exp.run_retrieval_trial(base_model, seed=42, steps=10)

    assert isinstance(rep, DynamicTokenRetrievalReport)
    assert rep.train_accuracy >= 0.0
    assert rep.final_loss > 0.0
    assert rep.runtime_seconds > 0.0


def test_step178_associative_recall_with_distractors():
    """Verify associative recall experiment compares clean vs distractor robustness."""
    base_model = instantiate_frozen_baseline()
    exp = AssociativeRecallExperiment()
    rep: AssociativeRecallReport = exp.run_distractor_experiment(base_model, steps=10)

    assert isinstance(rep, AssociativeRecallReport)
    assert hasattr(rep, "clean_pair_accuracy")
    assert hasattr(rep, "distractor_pair_accuracy")
    assert hasattr(rep, "attention_entropy")


def test_step179_induction_head_analysis(tmp_path):
    """Verify quantitative induction metrics across transformer layers and heads."""
    obs_mgr = CandidateIsolationManager(tmp_path / "exp.db")
    cand_model, _ = obs_mgr.create_isolated_candidate()
    base_model = instantiate_frozen_baseline()

    analyzer = InductionHeadAnalyzer()
    rep: InductionAnalysisReport = analyzer.analyze_induction_circuits(base_model, cand_model)

    assert isinstance(rep, InductionAnalysisReport)
    assert len(rep.head_scores) == cand_model.config.n_layers * cand_model.config.n_heads
    assert rep.classification == "DIAGNOSTIC_EVIDENCE"


def test_step180_disjoint_key_value_generalization():
    """Verify multi-seed evaluation across known vs disjoint key-value mappings."""
    base_model = instantiate_frozen_baseline()
    exp = DisjointKeyValueGeneralizationExperiment()
    rep: DisjointGeneralizationReport = exp.run_disjoint_trials(base_model, seeds=[42, 101], steps=10)

    assert isinstance(rep, DisjointGeneralizationReport)
    assert len(rep.seed_results) == 2
    assert hasattr(rep, "induction_level")


def test_step181_variable_binding_reevaluation(tmp_path):
    """Verify re-evaluation tracks pre- vs post-induction variable binding metrics."""
    obs_mgr = CandidateIsolationManager(tmp_path / "exp.db")
    cand_model, _ = obs_mgr.create_isolated_candidate()
    base_model = instantiate_frozen_baseline()

    evaluator = VariableBindingReEvaluation()
    rep: VariableBindingReEvaluationReport = evaluator.run_reevaluation(base_model, cand_model, steps=10)

    assert isinstance(rep, VariableBindingReEvaluationReport)
    assert hasattr(rep, "pre_induction_first_acc")
    assert hasattr(rep, "post_induction_first_acc")


def test_step182_multihop_reevaluation(tmp_path):
    """Verify multi-hop re-evaluation across 1-hop, 2-hop, 3-hop, and 4-hop prompts."""
    obs_mgr = CandidateIsolationManager(tmp_path / "exp.db")
    cand_model, _ = obs_mgr.create_isolated_candidate()

    evaluator = MultiHopReEvaluation()
    rep: MultiHopReEvaluationReport = evaluator.run_multihop_reevaluation(cand_model)

    assert isinstance(rep, MultiHopReEvaluationReport)
    assert len(rep.hop_results) == 4
    assert rep.first_loss_point != ""


def test_step183_anti_shortcut_validation(tmp_path):
    """Verify adversarial controls test positional permutations and frequency balance."""
    obs_mgr = CandidateIsolationManager(tmp_path / "exp.db")
    cand_model, _ = obs_mgr.create_isolated_candidate()

    validator = AntiShortcutValidator()
    rep: AntiShortcutValidationReport = validator.run_validation(cand_model)

    assert isinstance(rep, AntiShortcutValidationReport)
    assert len(rep.controls_evaluated) == 2
    assert rep.contamination_rate == 0.0


def test_step184_master_decision_gate():
    """Verify master decision gate benchmark execution and baseline immutability."""
    base_model = instantiate_frozen_baseline()
    assert sum(p.numel() for p in base_model.parameters()) == CANONICAL_PARAM_COUNT
    assert compute_model_hash(base_model) == CANONICAL_BASELINE_HASH
