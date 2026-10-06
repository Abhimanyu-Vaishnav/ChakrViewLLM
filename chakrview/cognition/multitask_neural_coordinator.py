"""
ChakrView Step 158: Multitask Neural Learning & Anti-Forgetting.

Tests 3 distinct training conditions:
- Candidate A: Language Only
- Candidate B: Reasoning Only
- Candidate C: Language + Reasoning Interleaved

Evaluates whether reasoning training degrades previously acquired language representations,
or whether interleaved training preserves language retention while acquiring reasoning capabilities.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from chakrview.brain.model import ChakrMicro
from chakrview.cognition.neural_language_learning import ControlledNeuralLanguageTrainer
from chakrview.cognition.neural_reasoning_experiment import NeuralReasoningExperimentRunner
from chakrview.cognition.reasoning_curriculum_ladder import ReasoningLadderItem


@dataclass
class MultitaskLearningComparisonReport:
    candidate_a_lang_loss: float
    candidate_a_reason_acc: float
    candidate_b_lang_loss: float
    candidate_b_reason_acc: float
    candidate_c_lang_loss: float
    candidate_c_reason_acc: float
    language_retention_preserved: bool
    reasoning_acquired: bool


class MultitaskNeuralCoordinator:
    """
    Coordinates and compares language-only, reasoning-only, and interleaved multi-task candidates.
    """

    def __init__(
        self,
        lang_trainer: ControlledNeuralLanguageTrainer,
        reason_runner: NeuralReasoningExperimentRunner,
    ) -> None:
        self.lang_trainer = lang_trainer
        self.reason_runner = reason_runner

    def evaluate_multitask_matrix(
        self,
        cand_lang_only: ChakrMicro,
        cand_reason_only: ChakrMicro,
        cand_interleaved: ChakrMicro,
        train_lang_data: Any,
        held_reason_items: List[ReasoningLadderItem],
    ) -> MultitaskLearningComparisonReport:
        # Candidate A (Language only)
        l_loss_a, _ = self.lang_trainer.evaluate_loss_and_acc(cand_lang_only, train_lang_data)
        r_acc_a = self.reason_runner.evaluate_items(cand_lang_only, held_reason_items)

        # Candidate B (Reasoning only)
        l_loss_b, _ = self.lang_trainer.evaluate_loss_and_acc(cand_reason_only, train_lang_data)
        r_acc_b = self.reason_runner.evaluate_items(cand_reason_only, held_reason_items)

        # Candidate C (Interleaved)
        l_loss_c, _ = self.lang_trainer.evaluate_loss_and_acc(cand_interleaved, train_lang_data)
        r_acc_c = self.reason_runner.evaluate_items(cand_interleaved, held_reason_items)

        # Language retention is preserved if loss is below unadapted baseline ~8.39
        ret_ok = (l_loss_c < 8.25)
        # Reasoning acquired if accuracy is non-negative
        reason_ok = (r_acc_c >= r_acc_a)

        return MultitaskLearningComparisonReport(
            candidate_a_lang_loss=round(l_loss_a, 4),
            candidate_a_reason_acc=round(r_acc_a, 4),
            candidate_b_lang_loss=round(l_loss_b, 4),
            candidate_b_reason_acc=round(r_acc_b, 4),
            candidate_c_lang_loss=round(l_loss_c, 4),
            candidate_c_reason_acc=round(r_acc_c, 4),
            language_retention_preserved=ret_ok,
            reasoning_acquired=reason_ok,
        )
