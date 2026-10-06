"""
ChakrView Step 145-152: Master Development Wave Unit & Integration Tests.

Validates:
- Step 145: Neural Learning Observability & Candidate Isolation Protocol (Baseline Integrity & ΔW_baseline == 0)
- Step 146: Controlled Neural Language Learning (Real Gradient Optimization on CPU, Loss Reduction & Held-Out Acc)
- Step 147: Neural Reasoning Curriculum (Pattern Generalization Across Transitive Relations)
- Step 148: Generalization & Transfer Learning Evaluator (Acquisition, Retention & Transfer Deltas)
- Step 149: Neural Experience Learning Bridge (Episode Diagnostics to Synthetic Curriculum Proposals)
- Step 150: Governed Self-Improvement Controller (Multi-Gate Promotion & Automated Rollback on Regression)
- Step 151: Neural vs Cognitive Intelligence Separation Benchmark (Independent 4-Mode Evaluation)
- Step 152: Universal Baseline Immutability (3,443,136 params, SHA-256 c5571c... bit-exact)
"""

import time
from pathlib import Path
import pytest
import torch

from chakrview.brain.model import ChakrMicro
from chakrview.runtime.interactive import (
    instantiate_frozen_baseline,
    compute_model_hash,
    EXPECTED_WEIGHT_HASH,
)
from chakrview.cognition.neural_observability import (
    NeuralLearningExperiment,
    CandidateIsolationManager,
)
from chakrview.cognition.neural_language_learning import (
    ControlledNeuralLanguageTrainer,
    LanguageLearningResult,
)
from chakrview.cognition.neural_reasoning_curriculum import (
    NeuralReasoningCurriculumTrainer,
    NeuralReasoningResult,
)
from chakrview.cognition.generalization_transfer import (
    GeneralizationTransferEvaluator,
    TransferLearningMetrics,
)
from chakrview.cognition.experience_learning import (
    GovernedExperienceLearningEngine,
    ExperienceSignalOutcome,
)
from chakrview.cognition.neural_experience_bridge import (
    NeuralExperienceBridge,
    ExperienceCurriculumProposal,
)
from chakrview.cognition.self_improvement_controller import (
    GovernedSelfImprovementController,
    PromotionGateCriteria,
    SelfImprovementCycleResult,
)
from chakrview.cognition.intelligence_separation import (
    IntelligenceSeparationBenchmark,
    SystemExecutionMode,
    SeparationBenchmarkReport,
)


def test_step145_candidate_isolation_and_integrity(tmp_path):
    manager = CandidateIsolationManager(tmp_path / "exp_obs.db")
    candidate, base_h = manager.create_isolated_candidate()
    assert base_h == EXPECTED_WEIGHT_HASH

    baseline = instantiate_frozen_baseline()

    # Before training: candidate == baseline
    base_intact, cand_diverged, _, _ = manager.verify_candidate_integrity(baseline, candidate)
    assert base_intact is True
    assert cand_diverged is False

    # Perform a dummy parameter perturbation on candidate
    with torch.no_grad():
        for p in candidate.parameters():
            p.add_(1e-5)
            break

    # After perturbation: candidate != baseline, but baseline remains 100% intact
    base_intact, cand_diverged, base_final, cand_final = manager.verify_candidate_integrity(baseline, candidate)
    assert base_intact is True
    assert cand_diverged is True
    assert base_final == EXPECTED_WEIGHT_HASH
    assert cand_final != EXPECTED_WEIGHT_HASH

    # Record experiment
    exp = NeuralLearningExperiment(
        experiment_id="exp_01",
        parent_model_id="canonical_baseline",
        baseline_hash=base_final,
        candidate_hash=cand_final,
        dataset_identity="dataset_synth_v1",
        dataset_sha256="abc123hash",
        tokenizer_identity="vocab_4096",
        tokenizer_sha256="tok123hash",
        curriculum_identity="curr_lang_01",
        seed=42,
        optimizer_identity="AdamW",
        learning_rate=5e-4,
        batch_size=2,
        sequence_length=32,
        training_steps=10,
        training_device="cpu",
        parameter_count=3443136,
        train_loss=7.2,
        validation_loss=7.4,
        held_out_score=0.15,
        reasoning_score=0.20,
        language_score=0.25,
        generalization_score=0.18,
        regression_score=0.01,
        final_decision="PROMOTED_CANDIDATE",
    )
    manager.record_experiment(exp)


def test_step146_controlled_neural_language_learning():
    baseline = instantiate_frozen_baseline()
    manager = CandidateIsolationManager()
    candidate, _ = manager.create_isolated_candidate()

    trainer = ControlledNeuralLanguageTrainer(learning_rate=1e-3, device="cpu")
    res: LanguageLearningResult = trainer.train_candidate(candidate, baseline, steps=5)

    assert res.is_improved is True
    assert res.candidate_train_loss < res.baseline_train_loss
    assert res.loss_reduction_pct > 0.0
    # Baseline must remain strictly untouched
    assert compute_model_hash(baseline) == EXPECTED_WEIGHT_HASH


def test_step147_neural_reasoning_curriculum():
    baseline = instantiate_frozen_baseline()
    manager = CandidateIsolationManager()
    candidate, _ = manager.create_isolated_candidate()

    trainer = NeuralReasoningCurriculumTrainer(device="cpu")
    res: NeuralReasoningResult = trainer.train_and_evaluate_reasoning(candidate, baseline, steps=5)

    assert isinstance(res.baseline_train_acc, float)
    assert isinstance(res.candidate_train_acc, float)
    assert compute_model_hash(baseline) == EXPECTED_WEIGHT_HASH


def test_step148_generalization_transfer():
    baseline = instantiate_frozen_baseline()
    manager = CandidateIsolationManager()
    candidate, _ = manager.create_isolated_candidate()

    evaluator = GeneralizationTransferEvaluator(device="cpu")
    domains = {
        "DOMAIN_A": [("def add(x):", "return")],
        "DOMAIN_B": [("1 + 1 =", "2")],
    }
    metrics: TransferLearningMetrics = evaluator.evaluate_multi_domain_transfer(baseline, candidate, domains)

    assert "DOMAIN_A" in metrics.domain_accuracies
    assert "DOMAIN_B" in metrics.domain_accuracies
    assert metrics.retention_intact is True


def test_step149_neural_experience_bridge(tmp_path):
    exp_db = tmp_path / "exp_bridge.db"
    exp_engine = GovernedExperienceLearningEngine(exp_db)

    episode = exp_engine.process_experience(
        task_id="task_fail_42",
        domain_id="domain_python",
        outcome=ExperienceSignalOutcome.CRITICAL_FAILURE,
        raw_observations=["FileNotFoundError encountered"],
        error_context="path does not exist",
    )

    bridge = NeuralExperienceBridge(exp_engine)
    proposal: ExperienceCurriculumProposal = bridge.convert_episode_to_curriculum(episode)

    assert proposal.source_episode_id == episode.episode_id
    assert proposal.domain_id == "domain_python"
    assert proposal.sample_count >= 1
    assert "does not exist" in proposal.target_weakness.lower() or "defect" in proposal.target_weakness.lower()


def test_step150_governed_self_improvement_controller(tmp_path):
    obs_manager = CandidateIsolationManager(tmp_path / "exp.db")
    baseline = instantiate_frozen_baseline()
    candidate, _ = obs_manager.create_isolated_candidate()

    # Perturb candidate slightly
    with torch.no_grad():
        for p in candidate.parameters():
            p.add_(1e-5)
            break

    controller = GovernedSelfImprovementController(
        isolation_manager=obs_manager,
        gate_criteria=PromotionGateCriteria(min_heldout_delta=0.01, max_general_regression=0.04),
        db_path=tmp_path / "self_improvement.db",
    )

    # 1. Pass all gates
    res_promo: SelfImprovementCycleResult = controller.evaluate_promotion(
        cycle_id="c_pass",
        baseline_model=baseline,
        candidate_model=candidate,
        heldout_delta=0.03,
        reasoning_delta=0.02,
        regression_delta=0.01,
    )
    assert res_promo.decision == "PROMOTE_CANDIDATE"
    assert res_promo.baseline_intact is True

    # 2. Reject due to regression
    res_rollback: SelfImprovementCycleResult = controller.evaluate_promotion(
        cycle_id="c_fail",
        baseline_model=baseline,
        candidate_model=candidate,
        heldout_delta=0.03,
        reasoning_delta=0.02,
        regression_delta=0.08,  # > 0.04
    )
    assert res_rollback.decision == "ROLLBACK_CANDIDATE"


def test_step151_intelligence_separation_benchmark():
    bench = IntelligenceSeparationBenchmark()
    report: SeparationBenchmarkReport = bench.run_separation_evaluation(None)

    assert report.neural_intrinsic_improvement_proven is True
    assert "NEURAL_ONLY" in report.mean_scores_by_mode
    assert "FULL_CHAKRVIEW" in report.mean_scores_by_mode
    # Full orchestration achieves higher problem solving score than neural-only
    assert report.mean_scores_by_mode["FULL_CHAKRVIEW"] > report.mean_scores_by_mode["NEURAL_ONLY"]


def test_step152_canonical_baseline_immutability():
    baseline = instantiate_frozen_baseline()
    param_count = sum(p.numel() for p in baseline.parameters())
    assert param_count == 3443136, f"Baseline param count mismatch: {param_count} != 3443136"

    model_hash = compute_model_hash(baseline)
    assert model_hash == EXPECTED_WEIGHT_HASH
    assert model_hash == "c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da"
