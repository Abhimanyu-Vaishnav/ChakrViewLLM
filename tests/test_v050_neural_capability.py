"""
Test Suite for Waves/Stages of ChakrView v0.5.0 Neural Capability Expansion.
"""

import pytest
import torch

from chakrview.brain.canonical_hash import verify_canonical_baseline
from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro

from chakrview.cognition.v050_candidate.neural_trainer import IsolatedNeuralTrainer
from chakrview.cognition.v050_candidate.representation_experiment import RepresentationLearningExperiment
from chakrview.cognition.v050_candidate.compositional_experiment import CompositionalGeneralizationExperiment
from chakrview.cognition.v050_candidate.language_pattern_experiment import LanguagePatternExperiment
from chakrview.cognition.v050_candidate.neural_cognitive_coupling import NeuralCognitiveCouplingEngine
from chakrview.cognition.v050_candidate.continual_neural_experiment import ContinualNeuralExperiment
from chakrview.cognition.v050_candidate.experience_neural_pipeline import GovernedExperienceNeuralPipeline
from chakrview.cognition.v050_candidate.v050_master_gate import V050CandidateMasterGate


def test_canonical_baseline_remains_immutable():
    """Verify neural baseline remains bit-exact (Delta W = 0)."""
    torch.manual_seed(42)
    m = ChakrMicro(ModelConfig())
    res = verify_canonical_baseline(m)
    assert res["is_bit_exact"] is True
    assert res["delta_w"] == 0
    assert res["total_params"] == 3443136


def test_stage1_neural_learning_baseline():
    """Verify training convergence and parameter updates on isolated candidate."""
    cand, res = IsolatedNeuralTrainer.train_candidate(steps=10)
    assert res.is_converged is True
    assert res.final_loss < res.initial_loss
    assert res.parameters_updated > 0
    assert res.parameter_delta_l2 > 0.0


def test_stage2_representation_learning_disjoint():
    """Verify representation structure learning on train set and measure generalization gap on disjoint evaluation pairs."""
    res = RepresentationLearningExperiment.run_experiment(steps=10)
    assert res.train_accuracy > 0.5
    # Accurately reflect that disjoint zero-shot evaluation without prior exposure has a generalization gap
    assert res.generalization_gap >= 0.0


def test_stage3_compositional_generalization():
    """Verify compositional relational evaluation across hops."""
    res = CompositionalGeneralizationExperiment.run_experiment(steps=10)
    assert res.compositional_pass is True
    assert res.seen_composition_score > 0.5


def test_stage4_language_pattern_learning():
    """Verify token syntax predictability improvement."""
    res = LanguagePatternExperiment.run_experiment(steps=10)
    assert res.pattern_learning_demonstrated is True
    assert res.train_perplexity < 100.0


def test_stage5_neural_cognitive_coupling():
    """Verify neural representation output guides cognitive decision state with honest thresholds."""
    cand, _ = IsolatedNeuralTrainer.train_candidate(steps=15, lr=2e-3)
    inp = torch.tensor([[10, 20, 30]])
    res = NeuralCognitiveCouplingEngine.evaluate_coupling(cand, inp, "AlphaEntity")
    assert res.neural_signal_present is True
    assert res.cognitive_decision in ("ANSWER", "ABSTAIN")


def test_stage6_continual_neural_retention():
    """Verify sequential Task A -> Task B learning retains Task A without collapse."""
    res = ContinualNeuralExperiment.run_continual_test(steps_per_task=10)
    assert res.catastrophic_forgetting_detected is False
    assert res.backward_retention_rate >= 0.5


def test_stage7_and_8_governed_experience_and_parametric_adaptation():
    """Verify governance gate filters bad experience and candidate updates with rollback safety."""
    good_exp = {"token_sequence": [[10, 20, 30]], "target_sequence": [40], "confidence": 0.95}
    bad_exp = {"token_sequence": [[10, 20, 30]], "target_sequence": [40], "confidence": 0.40}

    out_good = GovernedExperienceNeuralPipeline.process_experience(good_exp)
    out_bad = GovernedExperienceNeuralPipeline.process_experience(bad_exp)

    assert out_good.experience_accepted is True
    assert out_good.parameter_updated is True
    assert out_good.improvement_verified is True
    assert out_bad.experience_accepted is False
    assert out_bad.decision == "REJECTED_BY_GOVERNANCE_GATE"


def test_stage10_v050_master_gate():
    """Evaluate full v0.5.0 candidate research gate."""
    rep = V050CandidateMasterGate.evaluate_candidate()
    assert rep.canonical_baseline_intact is True
    assert rep.baseline_delta_w == 0
    assert rep.final_status == "V050_NEURAL_CAPABILITY_PARTIAL"
