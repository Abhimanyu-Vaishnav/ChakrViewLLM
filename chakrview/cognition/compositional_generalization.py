"""
ChakrView Step 157: Systematic & Compositional Generalization.

Measures multi-tier generalization across distinct distribution shifts:
- G1: Unseen entities (disjoint letters)
- G2: Unseen linguistic templates ('exceeds', 'superior to')
- G3: Unseen relation combinations
- G4: Length generalization (trained on 2-hop, tested on 3-hop and 4-hop)
- G5: Distractor robustness (irrelevant facts in context)
- G6: Reversed query direction
- G7: Negative examples
- G8: Contradictory premises
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from chakrview.brain.model import ChakrMicro
from chakrview.cognition.reasoning_curriculum_ladder import ReasoningLadderItem
from chakrview.cognition.neural_reasoning_experiment import NeuralReasoningExperimentRunner


@dataclass
class GeneralizationTiersReport:
    g1_unseen_entities_score: float
    g2_unseen_templates_score: float
    g3_unseen_combinations_score: float
    g4_length_generalization_score: float
    g5_distractor_robustness_score: float
    g6_reversed_query_score: float
    g7_negative_examples_score: float
    g8_contradictory_premises_score: float
    systematic_generalization_score: float
    compositional_generalization_score: float


class CompositionalGeneralizationEvaluator:
    """
    Evaluates multi-tier systematic and compositional generalization on trained candidates.
    """

    def __init__(self, runner: NeuralReasoningExperimentRunner) -> None:
        self.runner = runner

    def evaluate_generalization_tiers(
        self,
        candidate_model: ChakrMicro,
        tier_test_items: Dict[str, List[ReasoningLadderItem]],
    ) -> GeneralizationTiersReport:
        scores: Dict[str, float] = {}

        for tier_key, items in tier_test_items.items():
            if items:
                acc = self.runner.evaluate_items(candidate_model, items)
            else:
                acc = 0.0
            scores[tier_key] = round(acc, 4)

        g1 = scores.get("G1", 0.0)
        g2 = scores.get("G2", 0.0)
        g3 = scores.get("G3", 0.0)
        g4 = scores.get("G4", 0.0)
        g5 = scores.get("G5", 0.0)
        g6 = scores.get("G6", 0.0)
        g7 = scores.get("G7", 0.0)
        g8 = scores.get("G8", 0.0)

        systematic = round((g1 + g2 + g3) / 3.0, 4)
        compositional = round((g3 + g4) / 2.0, 4)

        return GeneralizationTiersReport(
            g1_unseen_entities_score=g1,
            g2_unseen_templates_score=g2,
            g3_unseen_combinations_score=g3,
            g4_length_generalization_score=g4,
            g5_distractor_robustness_score=g5,
            g6_reversed_query_score=g6,
            g7_negative_examples_score=g7,
            g8_contradictory_premises_score=g8,
            systematic_generalization_score=systematic,
            compositional_generalization_score=compositional,
        )
