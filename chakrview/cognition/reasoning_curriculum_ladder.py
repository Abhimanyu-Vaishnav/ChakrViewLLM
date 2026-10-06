"""
ChakrView Step 154: Curriculum Ladder for Neural Reasoning.

Defines a progressive 9-level curriculum ladder:
LEVEL 0: Direct relation recognition (A > B -> max is A)
LEVEL 1: One-step inversion / transformation (A > B -> min is B)
LEVEL 2: Two-hop transitive inference (A > B, B > C -> max is A)
LEVEL 3: Three/four-hop transitive inference (A > B, B > C, C > D -> max is A)
LEVEL 4: Distractor robustness (A > B, unrelated P > Q, B > C -> max is A)
LEVEL 5: Paraphrased relations (A exceeds B -> superior is A)
LEVEL 6: Variable/symbol substitution
LEVEL 7: Unseen compositional combinations
LEVEL 8: Mixed reasoning families (Order, Equality, Negation)

Tracks train, validation, and held-out sets per level.
Advancement requires held-out validation mastery.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple


class ReasoningLadderLevel(int, enum.Enum):
    LEVEL_0_DIRECT_RELATION = 0
    LEVEL_1_INVERSION = 1
    LEVEL_2_TWO_HOP_TRANSITIVE = 2
    LEVEL_3_MULTI_HOP_TRANSITIVE = 3
    LEVEL_4_DISTRACTOR_ROBUSTNESS = 4
    LEVEL_5_PARAPHRASED_RELATIONS = 5
    LEVEL_6_SYMBOL_SUBSTITUTION = 6
    LEVEL_7_UNSEEN_COMPOSITION = 7
    LEVEL_8_MIXED_FAMILIES = 8


@dataclass
class ReasoningLadderItem:
    item_id: str
    level: ReasoningLadderLevel
    prompt: str
    target: str
    is_held_out: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LevelMasteryReport:
    level: ReasoningLadderLevel
    train_accuracy: float
    held_out_accuracy: float
    mastery_achieved: bool
    threshold_required: float = 0.50


class ReasoningCurriculumLadder:
    """
    Manages progressive staged learning across reasoning levels.
    """

    def __init__(self) -> None:
        self.levels: Dict[ReasoningLadderLevel, List[ReasoningLadderItem]] = {
            lvl: [] for lvl in ReasoningLadderLevel
        }

    def add_item(self, item: ReasoningLadderItem) -> None:
        self.levels[item.level].append(item)

    def get_total_items(self) -> int:
        return sum(len(items) for items in self.levels.values())

    def get_level_data(
        self,
        level: ReasoningLadderLevel,
    ) -> Tuple[List[ReasoningLadderItem], List[ReasoningLadderItem]]:
        """Returns (train_items, heldout_items)."""
        items = self.levels[level]
        train = [i for i in items if not i.is_held_out]
        heldout = [i for i in items if i.is_held_out]
        return train, heldout

    def evaluate_level_mastery(
        self,
        level: ReasoningLadderLevel,
        train_acc: float,
        heldout_acc: float,
        mastery_threshold: float = 0.50,
    ) -> LevelMasteryReport:
        achieved = (heldout_acc >= mastery_threshold)
        return LevelMasteryReport(
            level=level,
            train_accuracy=round(train_acc, 4),
            held_out_accuracy=round(heldout_acc, 4),
            mastery_achieved=achieved,
            threshold_required=mastery_threshold,
        )
