"""
candidate_banks.py
==================
ChakrView Exp18 — Scaled Deterministic Candidate Banks.

Provides curated, verified, and completely disjoint candidate banks for:
- factual_qa (150 Train, 20 Val, 20 Test = 190 items)
- short_explanation (150 Train, 20 Val, 20 Test = 190 items)
- uncertainty_abstention (150 Train, 20 Val, 20 Test = 190 items, paired unanswerable + grounded controls)

All items strictly observe:
1. 100% token length compliance within hard bounds [3, 45] (mean ~15-28 tokens).
2. 0% lexical or entity overlap with frozen Normal Chat Gate v0.1 prompts.
3. Strict cross-split disjointness across Train, Val, and Test.
4. Full compatibility with TaskEvaluator v0.1 evaluation rules:
   - Factual QA targets aligned with gazetteer and SI standards.
   - Short explanations categorized as NEEDS_REVIEW (held out from scorable accuracy).
   - Abstention items pairing unanswerable queries (3:1 ratio) with grounded factual controls.
"""

from __future__ import annotations
from typing import Dict, List, Tuple

from experiments.research_v09_conversation_sft.exp18_scaled_instruction_format_v1.raw_fqa_data import FACTUAL_QA_DATA
from experiments.research_v09_conversation_sft.exp18_scaled_instruction_format_v1.raw_exp_data import EXPLANATION_DATA
from experiments.research_v09_conversation_sft.exp18_scaled_instruction_format_v1.raw_abstention_data import ABSTENTION_DATA

# Slice to exact required quotas: 150 Train, 20 Val, 20 Test
CANDIDATE_FACTUAL_BANKS: Dict[str, List[Tuple[str, str, str, str]]] = {
    "train": FACTUAL_QA_DATA["train"][:150],
    "val": FACTUAL_QA_DATA["val"][:20],
    "test": FACTUAL_QA_DATA["test"][:20],
}

CANDIDATE_EXPLANATION_BANKS: Dict[str, List[Tuple[str, str, str, str]]] = {
    "train": EXPLANATION_DATA["train"][:150],
    "val": EXPLANATION_DATA["val"][:20],
    "test": EXPLANATION_DATA["test"][:20],
}

CANDIDATE_ABSTENTION_BANKS: Dict[str, List[Tuple[str, str, str, str, str, str]]] = {
    "train": ABSTENTION_DATA["train"][:150],
    "val": ABSTENTION_DATA["val"][:20],
    "test": ABSTENTION_DATA["test"][:20],
}

__all__ = [
    "CANDIDATE_FACTUAL_BANKS",
    "CANDIDATE_EXPLANATION_BANKS",
    "CANDIDATE_ABSTENTION_BANKS",
]
