"""
Strategy Lifecycle & Aging Engine.
Wave 530: Prevents strategies from remaining permanently trusted without revalidation.

Lifecycle States:
- ACTIVE: High trust, high recent success rate, actively routed.
- AGING: Decreasing usage or minor failure drift; monitored.
- REVALIDATION_REQUIRED: Recent failure rate exceeded tolerance (> 20%); blocked from default dispatch.
- DEPRECATED: Failed revalidation; fallback only while replacement is synthesized.
- RETIRED: Archived; never dispatched.

Transitions:
ACTIVE -> AGING (after inactivity or initial failure)
AGING -> REVALIDATION_REQUIRED (failure rate >= 0.25)
REVALIDATION_REQUIRED -> ACTIVE (passes revalidation benchmark)
REVALIDATION_REQUIRED -> DEPRECATED (fails revalidation benchmark)
DEPRECATED -> RETIRED (superseded by verified replacement)
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any, List

class StrategyState(str, Enum):
    ACTIVE = "ACTIVE"
    AGING = "AGING"
    REVALIDATION_REQUIRED = "REVALIDATION_REQUIRED"
    DEPRECATED = "DEPRECATED"
    RETIRED = "RETIRED"

@dataclass
class TrackedStrategy:
    strategy_id: str
    task_family: str
    state: StrategyState = StrategyState.ACTIVE
    total_uses: int = 0
    recent_successes: int = 0
    recent_failures: int = 0
    consecutive_failures: int = 0
    trust_score: float = 1.0

    def record_outcome(self, success: bool) -> None:
        self.total_uses += 1
        if success:
            self.recent_successes += 1
            self.consecutive_failures = 0
            self.trust_score = min(1.0, self.trust_score + 0.05)
        else:
            self.recent_failures += 1
            self.consecutive_failures += 1
            self.trust_score = max(0.0, self.trust_score - 0.2)

        # Evaluate state transitions
        recent_total = self.recent_successes + self.recent_failures
        if recent_total >= 4:
            failure_rate = self.recent_failures / recent_total
            if failure_rate >= 0.5 or self.consecutive_failures >= 2:
                self.state = StrategyState.REVALIDATION_REQUIRED
            elif failure_rate >= 0.2:
                self.state = StrategyState.AGING
            else:
                self.state = StrategyState.ACTIVE

    def perform_revalidation(self, benchmark_score: float, pass_threshold: float = 0.8) -> None:
        if benchmark_score >= pass_threshold:
            self.state = StrategyState.ACTIVE
            self.recent_failures = 0
            self.recent_successes = 5
            self.trust_score = 0.95
        else:
            self.state = StrategyState.DEPRECATED
            self.trust_score = 0.2

    def retire(self) -> None:
        self.state = StrategyState.RETIRED
        self.trust_score = 0.0
