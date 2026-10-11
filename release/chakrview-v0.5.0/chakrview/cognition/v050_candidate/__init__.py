"""
ChakrView v0.5.0 Candidate: Neural Capability Expansion & Learning Experiments.

Encompasses Stages 1-8:
- Stage 1: Neural Learning Baseline & Optimization
- Stage 2: Representation Learning on Disjoint Relations
- Stage 3: Compositional Generalization on Unseen Pairs
- Stage 4: Language Pattern Learning & Perplexity Evaluation
- Stage 5: Language + Cognitive State Coupling & Gating
- Stage 6: Continual Learning & Retention Audit
- Stage 7: Governed Experience -> Candidate Update Engine
- Stage 8: Parametric Adaptation & Rollback Management
"""

from .neural_trainer import IsolatedNeuralTrainer, TrainingRunResult
from .representation_experiment import RepresentationLearningExperiment, RepresentationResult
from .compositional_experiment import CompositionalGeneralizationExperiment, CompositionalResult
from .language_pattern_experiment import LanguagePatternExperiment, LanguagePatternResult
from .neural_cognitive_coupling import NeuralCognitiveCouplingEngine, CouplingResult
from .continual_neural_experiment import ContinualNeuralExperiment, ContinualResult
from .experience_neural_pipeline import GovernedExperienceNeuralPipeline, ExperienceLearningOutcome
from .v050_master_gate import V050CandidateMasterGate, V050EvaluationReport

__all__ = [
    "IsolatedNeuralTrainer",
    "TrainingRunResult",
    "RepresentationLearningExperiment",
    "RepresentationResult",
    "CompositionalGeneralizationExperiment",
    "CompositionalResult",
    "LanguagePatternExperiment",
    "LanguagePatternResult",
    "NeuralCognitiveCouplingEngine",
    "CouplingResult",
    "ContinualNeuralExperiment",
    "ContinualResult",
    "GovernedExperienceNeuralPipeline",
    "ExperienceLearningOutcome",
    "V050CandidateMasterGate",
    "V050EvaluationReport",
]
