"""
Wave 578: Evidence Budget & Resource-Aware Research Governor.

Architectural Guarantees:
- RESOURCE AVAILABILITY != PERMISSION != REQUIREMENT.
- USER WORKLOAD > OPTIONAL AI WORKLOAD.
- Decides: SEARCH_MORE, VERIFY_EXISTING, STOP_SEARCH, ABSTAIN, ANSWER, ASK_CLARIFICATION.
- Stopping reasons:
  SUFFICIENT_EVIDENCE, CONFLICT_REQUIRES_HUMAN_DECISION, NO_HIGHER_VALUE_SOURCE_AVAILABLE,
  BUDGET_EXHAUSTED, USER_WORKLOAD_ACTIVE, EVIDENCE_SATURATED.
- Prevents infinite retrieval loops and unnecessary resource consumption.
"""

from __future__ import annotations

import time
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple


class BudgetStoppingPolicy(str, Enum):
    SUFFICIENT_EVIDENCE = "SUFFICIENT_EVIDENCE"
    CONFLICT_REQUIRES_HUMAN_DECISION = "CONFLICT_REQUIRES_HUMAN_DECISION"
    NO_HIGHER_VALUE_SOURCE_AVAILABLE = "NO_HIGHER_VALUE_SOURCE_AVAILABLE"
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"
    USER_WORKLOAD_ACTIVE = "USER_WORKLOAD_ACTIVE"
    EVIDENCE_SATURATED = "EVIDENCE_SATURATED"
    CONTINUE_SEARCH = "CONTINUE_SEARCH"


class BudgetDecision(str, Enum):
    SEARCH_MORE = "SEARCH_MORE"
    VERIFY_EXISTING = "VERIFY_EXISTING"
    STOP_SEARCH = "STOP_SEARCH"
    ABSTAIN = "ABSTAIN"
    ANSWER = "ANSWER"
    ASK_CLARIFICATION = "ASK_CLARIFICATION"


@dataclass
class ResearchBudget:
    max_retrieval_calls: int = 5
    max_documents: int = 15
    max_bytes: int = 250_000
    max_elapsed_seconds: float = 10.0
    allowed_during_user_load: bool = False


@dataclass
class ResourceCostTracker:
    retrieval_calls_used: int = 0
    documents_ingested: int = 0
    bytes_consumed: int = 0
    start_time: float = field(default_factory=time.time)
    user_workload_active: bool = False


class ResourceAwareEvidenceGovernor:
    """
    Governs evidence gathering by calculating marginal information gain vs resource cost
    and respecting user system priorities.
    """

    @classmethod
    def evaluate_next_action(
        cls,
        budget: ResearchBudget,
        tracker: ResourceCostTracker,
        current_confidence: float,
        is_sufficient: bool,
        has_deadlock_conflict: bool,
        remaining_candidates_count: int,
    ) -> Tuple[BudgetDecision, BudgetStoppingPolicy, str]:
        elapsed = time.time() - tracker.start_time

        # 1. User Workload Priority Check
        if tracker.user_workload_active and not budget.allowed_during_user_load:
            return (
                BudgetDecision.STOP_SEARCH,
                BudgetStoppingPolicy.USER_WORKLOAD_ACTIVE,
                "User interactive workload active; pausing background retrieval to preserve system performance.",
            )

        # 2. Sufficiency Check (Information saturation)
        if is_sufficient and current_confidence >= 0.90:
            return (
                BudgetDecision.ANSWER,
                BudgetStoppingPolicy.SUFFICIENT_EVIDENCE,
                "Evidence threshold satisfied with high epistemic confidence; stopping search.",
            )

        # 3. Unresolvable Conflict Check
        if has_deadlock_conflict:
            return (
                BudgetDecision.ABSTAIN,
                BudgetStoppingPolicy.CONFLICT_REQUIRES_HUMAN_DECISION,
                "Direct epistemic deadlock between authoritative sources; stopping retrieval and escalating.",
            )

        # 4. Budget Exhaustion Checks
        if (
            tracker.retrieval_calls_used >= budget.max_retrieval_calls
            or tracker.documents_ingested >= budget.max_documents
            or tracker.bytes_consumed >= budget.max_bytes
            or elapsed >= budget.max_elapsed_seconds
        ):
            return (
                BudgetDecision.STOP_SEARCH,
                BudgetStoppingPolicy.BUDGET_EXHAUSTED,
                "Allocated retrieval resource budget reached limit.",
            )

        # 5. Diminishing Returns / No candidate sources
        if remaining_candidates_count <= 0:
            return (
                BudgetDecision.STOP_SEARCH,
                BudgetStoppingPolicy.NO_HIGHER_VALUE_SOURCE_AVAILABLE,
                "No additional high-value evidence sources available in discovery queue.",
            )

        # 6. Marginal Value Justifies Next Step
        if current_confidence < 0.80 and tracker.retrieval_calls_used < budget.max_retrieval_calls:
            return (
                BudgetDecision.SEARCH_MORE,
                BudgetStoppingPolicy.CONTINUE_SEARCH,
                "Confidence insufficient and remaining budget allows targeted retrieval.",
            )

        return (
            BudgetDecision.VERIFY_EXISTING,
            BudgetStoppingPolicy.CONTINUE_SEARCH,
            "Sufficient documents gathered; proceed to cross-validation phase.",
        )
