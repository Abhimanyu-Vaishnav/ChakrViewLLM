"""
Wave 586: Experience <-> Reasoning Integration Bridge.

Architectural Guarantees:
- Reuses validated past strategies for structurally similar problems.
- Never blindly replays arbitrary solutions without matching task structure.
- Rejects strategies that accumulated negative experience or failure history.
- Zero neural weight mutation (Delta W = 0). Maintains PARAMETRIC_ADAPTATION = NOT_YET_JUSTIFIED.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple


@dataclass
class StrategyCandidate:
    strategy_id: str
    target_task_type: str
    success_count: int = 0
    failure_count: int = 0
    confidence: float = 0.5
    recommended_chunk_size: int = 500
    preferred_planner_depth: int = 3
    is_active: bool = True

    @property
    def success_rate(self) -> float:
        total = self.success_count + self.failure_count
        return (self.success_count / total) if total > 0 else 0.5


class ExperienceReasoningBridge:
    """
    Selects, scores, and updates high-level reasoning strategies based on
    accumulated empirical performance without mutating neural core weights.
    """

    def __init__(self):
        self.strategies: Dict[str, StrategyCandidate] = {}
        # Seed standard operational strategies
        self.register_strategy(StrategyCandidate(
            strategy_id="strat_multihop_divide",
            target_task_type="MULTI_HOP_REASONING",
            success_count=12,
            failure_count=1,
            confidence=0.92,
            preferred_planner_depth=4,
        ))
        self.register_strategy(StrategyCandidate(
            strategy_id="strat_low_ram_stream",
            target_task_type="LONG_DOCUMENT_SYNTHESIS",
            success_count=15,
            failure_count=0,
            confidence=0.98,
            recommended_chunk_size=300,
        ))
        self.register_strategy(StrategyCandidate(
            strategy_id="strat_naive_guess",
            target_task_type="UNKNOWN_HANDLING",
            success_count=0,
            failure_count=8,
            confidence=0.05,
            is_active=False,
        ))

    def register_strategy(self, strat: StrategyCandidate) -> None:
        self.strategies[strat.strategy_id] = strat

    def match_best_strategy(
        self,
        task_type: str,
        resource_constrained: bool = False,
    ) -> Tuple[Optional[StrategyCandidate], str]:
        matching = [
            s for s in self.strategies.values()
            if s.target_task_type == task_type and s.is_active and s.success_rate >= 0.70
        ]

        if not matching:
            return None, "No proven prior strategy found for this task type; falling back to first-principles reasoning."

        # Pick highest confidence
        best = max(matching, key=lambda x: (x.success_rate, x.confidence))
        return best, f"Adopted proven strategy '{best.strategy_id}' based on {best.success_count} historical successes."

    def record_strategy_outcome(self, strategy_id: str, success: bool) -> None:
        if strategy_id in self.strategies:
            strat = self.strategies[strategy_id]
            if success:
                strat.success_count += 1
            else:
                strat.failure_count += 1
            total = strat.success_count + strat.failure_count
            strat.confidence = round(strat.success_count / total, 3)
            # Deactivate if failure rate drops below acceptable threshold
            if total >= 5 and strat.success_rate < 0.5:
                strat.is_active = False
