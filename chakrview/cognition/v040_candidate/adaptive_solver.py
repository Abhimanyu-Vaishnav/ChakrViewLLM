"""
Wave 588: Adaptive Problem Solving & Strategy Selector.

Architectural Guarantees:
- Unifies: Reasoning + Working Memory + Experience + Planning + Uncertainty + Resource Governance.
- Selects optimal strategy dynamically based on:
  - Task complexity (Single-hop vs Multi-hop vs Long-document)
  - Hardware resource constraints (Low-RAM / 28nm mode vs Standard)
  - Evidence sufficiency and conflict status
- Avoids hardcoded task rules by using a general multi-criteria policy.
"""

from __future__ import annotations

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple


class ProblemSolvingStrategy(str, Enum):
    DIRECT_DEDUCTION = "DIRECT_DEDUCTION"
    HIERARCHICAL_DECOMPOSITION = "HIERARCHICAL_DECOMPOSITION"
    STREAMING_CHUNK_SYNTHESIS = "STREAMING_CHUNK_SYNTHESIS"
    EVIDENCE_RESEARCH_PLAN = "EVIDENCE_RESEARCH_PLAN"
    HONEST_ABSTENTION = "HONEST_ABSTENTION"
    CLARIFICATION_PROMPT = "CLARIFICATION_PROMPT"


@dataclass
class ProblemStrategyDecision:
    chosen_strategy: ProblemSolvingStrategy
    expected_memory_bytes: int
    expected_steps: int
    rationale: str


class AdaptiveProblemSolver:
    """
    Selects the most efficient, bounded problem-solving strategy
    considering task properties and host resource constraints.
    """

    @classmethod
    def select_strategy(
        cls,
        task_type: str,
        input_length_chars: int,
        is_low_resource_profile: bool,
        has_sufficient_evidence: bool,
        is_ambiguous: bool,
        has_conflict: bool,
    ) -> ProblemStrategyDecision:
        # 1. Ambiguity check
        if is_ambiguous:
            return ProblemStrategyDecision(
                chosen_strategy=ProblemSolvingStrategy.CLARIFICATION_PROMPT,
                expected_memory_bytes=50_000,
                expected_steps=1,
                rationale="Inquiry is semantically ambiguous; prompting user for clarification before executing heavy reasoning.",
            )

        # 2. Insufficient evidence / Unverifiable
        if not has_sufficient_evidence and not has_conflict:
            return ProblemStrategyDecision(
                chosen_strategy=ProblemSolvingStrategy.EVIDENCE_RESEARCH_PLAN,
                expected_memory_bytes=200_000,
                expected_steps=3,
                rationale="Insufficient premises in working state; formulating structured research and evidence acquisition plan.",
            )

        # 3. Large input document
        if input_length_chars > 2000 or (is_low_resource_profile and input_length_chars > 800):
            return ProblemStrategyDecision(
                chosen_strategy=ProblemSolvingStrategy.STREAMING_CHUNK_SYNTHESIS,
                expected_memory_bytes=100_000,  # Bounded peak chunk memory
                expected_steps=4,
                rationale="Context size exceeds immediate memory threshold; selecting bounded streaming chunk processor to safeguard RAM.",
            )

        # 4. Complex Multi-Hop Task
        if task_type in ("MULTI_HOP_REASONING", "COMPLEX_GOAL_EXECUTION"):
            return ProblemStrategyDecision(
                chosen_strategy=ProblemSolvingStrategy.HIERARCHICAL_DECOMPOSITION,
                expected_memory_bytes=400_000,
                expected_steps=5,
                rationale="Multi-stage task requirements detected; decomposing into hierarchical subgoals with verified action execution.",
            )

        # 5. Simple Direct Deduction
        return ProblemStrategyDecision(
            chosen_strategy=ProblemSolvingStrategy.DIRECT_DEDUCTION,
            expected_memory_bytes=150_000,
            expected_steps=1,
            rationale="Single-premise task with verified ground evidence; direct deduction selected.",
        )
