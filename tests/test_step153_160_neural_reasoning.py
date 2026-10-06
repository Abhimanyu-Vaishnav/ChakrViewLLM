"""Tests for ChakrView Steps 153–160: Neural Reasoning Breakthrough, Generalization & Learning Science.

Validates:
- Step 153: Reasoning failure autopsy and multi-hypothesis diagnosis
- Step 154: Curriculum ladder for neural reasoning (levels 0 to 8)
- Step 155: Deterministic procedural reasoning generator with contamination control
- Step 156: Neural reasoning training experiments and candidate parameter isolation
- Step 157: Systematic and compositional generalization across tiers G1 to G8
- Step 158: Multitask neural learning (language vs reasoning vs interleaved)
- Step 159: Evidence-based governed self-improvement loop with mandatory capability gating
- Step 160: Master neural reasoning benchmark categories and canonical baseline immutability
"""

import pytest
import os
import shutil
from pathlib import Path

from chakrview.runtime.interactive import (
    instantiate_frozen_baseline,
    compute_model_hash,
    EXPECTED_WEIGHT_HASH,
)
from chakrview.cognition.neural_observability import (
    CandidateIsolationManager,
)
from chakrview.cognition.reasoning_autopsy import (
    ReasoningFailureAutopsy,
    ReasoningFailureDiagnosis,
)
from chakrview.cognition.reasoning_curriculum_ladder import (
    ReasoningCurriculumLadder,
    ReasoningLadderLevel,
    ReasoningLadderItem,
)
from chakrview.cognition.procedural_reasoning_generator import (
    ProceduralReasoningGenerator,
)
from chakrview.cognition.neural_reasoning_experiment import (
    NeuralReasoningExperimentRunner,
    ReasoningTrainingRunMetrics,
)
from chakrview.cognition.compositional_generalization import (
    CompositionalGeneralizationEvaluator,
    GeneralizationTiersReport,
)
from chakrview.cognition.neural_language_learning import (
    ControlledNeuralLanguageTrainer,
)
from chakrview.cognition.multitask_neural_coordinator import (
    MultitaskNeuralCoordinator,
)
from chakrview.cognition.evidence_self_improvement import (
    EvidenceSelfImprovementOrchestrator,
    EvidenceSelfImprovementDecision,
)


CANONICAL_BASELINE_HASH = EXPECTED_WEIGHT_HASH
CANONICAL_PARAM_COUNT = 3443136


def test_step153_reasoning_failure_autopsy():
    """Verify reasoning failure autopsy detects tokenization fragmentation and misalignment."""
    diagnosis: ReasoningFailureDiagnosis = ReasoningFailureAutopsy.perform_autopsy()
    assert isinstance(diagnosis, ReasoningFailureDiagnosis)
    assert len(diagnosis.root_cause_categories) >= 3
    assert "TOKENIZER_FRAGMENTATION" in diagnosis.root_cause_categories
    assert "TARGET_ALIGNMENT_MISMATCH" in diagnosis.root_cause_categories
    assert len(diagnosis.hypotheses) >= 3
    assert "target_id_mismatch" in diagnosis.evidence_summary
    assert diagnosis.evidence_summary["fragmentation_proven"] is True


def test_step154_curriculum_ladder_structure():
    """Verify 9-level curriculum ladder initialization, levels, and item assignment."""
    ladder = ReasoningCurriculumLadder()
    assert len(ladder.levels) == 9
    assert ReasoningLadderLevel.LEVEL_0_DIRECT_RELATION in ladder.levels
    assert ReasoningLadderLevel.LEVEL_8_MIXED_FAMILIES in ladder.levels

    item = ReasoningLadderItem(
        item_id="item_01",
        level=ReasoningLadderLevel.LEVEL_0_DIRECT_RELATION,
        prompt="A > B -> first: ",
        target="A",
        is_held_out=False,
    )
    ladder.add_item(item)
    assert len(ladder.levels[ReasoningLadderLevel.LEVEL_0_DIRECT_RELATION]) == 1
    assert ladder.get_total_items() == 1


def test_step155_procedural_generator_and_contamination_control():
    """Verify procedural reasoning generator covers required families with zero contamination."""
    data_gen = ProceduralReasoningGenerator(seed=42)
    symbols_train = ["A", "B", "C", "D", "E"]
    symbols_heldout = ["X", "Y", "Z", "U", "V"]

    dir_train, dir_held = data_gen.generate_direct_relation_data(symbols_train, symbols_heldout)
    trans_train, trans_held = data_gen.generate_transitive_data(symbols_train, symbols_heldout)

    assert len(dir_train) > 0
    assert len(dir_held) > 0
    assert len(trans_train) > 0
    assert len(trans_held) > 0

    contam_report = data_gen.check_contamination()
    assert contam_report.is_clean
    assert contam_report.contamination_rate == 0.0


def test_step156_neural_reasoning_training_experiment(tmp_path):
    """Verify CPU-first neural reasoning training runs on isolated candidate and updates weights."""
    obs_mgr = CandidateIsolationManager(tmp_path / "exp.db")
    candidate_model, base_h = obs_mgr.create_isolated_candidate()
    assert base_h == CANONICAL_BASELINE_HASH

    data_gen = ProceduralReasoningGenerator(seed=42)
    dir_train, dir_held = data_gen.generate_direct_relation_data(["A", "B"], ["X", "Y"])

    runner = NeuralReasoningExperimentRunner(device="cpu")
    metrics: ReasoningTrainingRunMetrics = runner.train_candidate_reasoning(
        candidate_model=candidate_model,
        train_items=dir_train,
        heldout_items=dir_held,
        steps=5,
        lr=1e-3,
    )
    assert isinstance(metrics, ReasoningTrainingRunMetrics)
    assert metrics.train_loss_final < metrics.train_loss_initial
    assert metrics.parameter_norm_delta > 0.0


def test_step157_systematic_and_compositional_generalization():
    """Verify evaluation across generalization tiers G1 to G8 produces separate honest scores."""
    baseline_model = instantiate_frozen_baseline()
    runner = NeuralReasoningExperimentRunner(device="cpu")
    evaluator = CompositionalGeneralizationEvaluator(runner)

    tier_items = {
        "G1": [ReasoningLadderItem("g1", ReasoningLadderLevel.LEVEL_0_DIRECT_RELATION, "X > Y -> first: ", "X", is_held_out=True)],
        "G2": [ReasoningLadderItem("g2", ReasoningLadderLevel.LEVEL_5_PARAPHRASED_RELATIONS, "A exceeds B -> first: ", "A", is_held_out=True)],
        "G3": [ReasoningLadderItem("g3", ReasoningLadderLevel.LEVEL_2_TWO_HOP_TRANSITIVE, "A > B, B > C -> first: ", "A", is_held_out=True)],
        "G4": [ReasoningLadderItem("g4", ReasoningLadderLevel.LEVEL_3_MULTI_HOP_TRANSITIVE, "chain: A>B,B>C,C>D -> first: ", "A", is_held_out=True)],
        "G5": [ReasoningLadderItem("g5", ReasoningLadderLevel.LEVEL_4_DISTRACTOR_ROBUSTNESS, "A>B, noise: P>Q -> first: ", "A", is_held_out=True)],
        "G6": [ReasoningLadderItem("g6", ReasoningLadderLevel.LEVEL_1_INVERSION, "A>B -> last: ", "B", is_held_out=True)],
        "G7": [ReasoningLadderItem("g7", ReasoningLadderLevel.LEVEL_8_MIXED_FAMILIES, "is A > B: ", "True", is_held_out=True)],
        "G8": [ReasoningLadderItem("g8", ReasoningLadderLevel.LEVEL_8_MIXED_FAMILIES, "conflict: A>B & B>A -> ", "CONFLICT", is_held_out=True)],
    }
    report: GeneralizationTiersReport = evaluator.evaluate_generalization_tiers(baseline_model, tier_items)
    assert isinstance(report, GeneralizationTiersReport)
    assert 0.0 <= report.g1_unseen_entities_score <= 1.0
    assert 0.0 <= report.g2_unseen_templates_score <= 1.0
    assert 0.0 <= report.systematic_generalization_score <= 1.0
    assert 0.0 <= report.compositional_generalization_score <= 1.0


def test_step158_multitask_neural_learning_and_anti_forgetting(tmp_path):
    """Verify multitask training coordinator compares candidates and computes retention deltas."""
    obs_mgr = CandidateIsolationManager(tmp_path / "exp.db")
    base_model = instantiate_frozen_baseline()
    cand_lang, _ = obs_mgr.create_isolated_candidate()
    cand_reason, _ = obs_mgr.create_isolated_candidate()
    cand_inter, _ = obs_mgr.create_isolated_candidate()

    lang_trainer = ControlledNeuralLanguageTrainer(device="cpu")
    train_lang, _, _ = lang_trainer.create_synthetic_language_corpus()
    lang_trainer.train_candidate(cand_lang, base_model, steps=3)
    lang_trainer.train_candidate(cand_inter, base_model, steps=3)

    data_gen = ProceduralReasoningGenerator(seed=42)
    dir_train, dir_held = data_gen.generate_direct_relation_data(["A", "B"], ["X", "Y"])
    reason_runner = NeuralReasoningExperimentRunner(device="cpu")
    reason_runner.train_candidate_reasoning(cand_reason, dir_train, dir_held, steps=3)
    reason_runner.train_candidate_reasoning(cand_inter, dir_train, dir_held, steps=3)

    coord = MultitaskNeuralCoordinator(lang_trainer, reason_runner)
    report = coord.evaluate_multitask_matrix(
        cand_lang_only=cand_lang,
        cand_reason_only=cand_reason,
        cand_interleaved=cand_inter,
        train_lang_data=train_lang,
        held_reason_items=dir_held,
    )
    assert report.language_retention_preserved is True


def test_step159_evidence_based_self_improvement_rejection_gate(tmp_path):
    """Verify governed self-improvement loop enforces mandatory capability gating and rejects on zero reasoning."""
    obs_mgr = CandidateIsolationManager(tmp_path / "exp.db")
    base_model = instantiate_frozen_baseline()
    cand_model, _ = obs_mgr.create_isolated_candidate()
    diagnosis = ReasoningFailureAutopsy.perform_autopsy()

    orchestrator = EvidenceSelfImprovementOrchestrator(obs_mgr)
    decision = orchestrator.execute_reasoning_improvement_cycle(
        cycle_id="test_cycle",
        diagnosis=diagnosis,
        baseline_model=base_model,
        candidate_model=cand_model,
        baseline_reasoning_acc=0.0,
        candidate_reasoning_acc=0.0,
        language_loss_delta=0.02,
        min_reasoning_acc_threshold=0.50,
    )
    assert isinstance(decision, EvidenceSelfImprovementDecision)
    assert decision.decision == "REJECT_CANDIDATE"
    assert decision.baseline_intact is True
    assert "mandatory" in decision.audit_rationale.lower()


def test_step160_baseline_immutability():
    """Verify canonical baseline remains strictly bit-exact and unchanged."""
    base_model = instantiate_frozen_baseline()
    assert sum(p.numel() for p in base_model.parameters()) == CANONICAL_PARAM_COUNT
    assert compute_model_hash(base_model) == CANONICAL_BASELINE_HASH
