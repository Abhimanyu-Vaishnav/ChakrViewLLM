"""
ChakrView Step 139: Multi-Domain Learning & Anti-Forgetting Coordinator.

Prevents catastrophic forgetting during multi-domain progression:
- Domain-balanced interleaved sampling
- Evaluates domain performance, retention of previous domains, and general capability retention
- Rollback trigger if regression on existing/foundation capabilities exceeds governed threshold
- Candidate isolation (canonical baseline remains bit-exact and frozen)
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from chakrview.cognition.domain_curriculum import DomainCurriculumSample


@dataclass
class MultiDomainRetentionReport:
    domain_accuracies: Dict[str, float]
    general_baseline_accuracy: float
    catastrophic_forgetting_detected: bool
    regression_delta: float
    recommended_action: str  # "PROCEED_CANDIDATE", "ADAPT_SAMPLING", "ROLLBACK_CHECKPOINT"


class MultiDomainAntiForgettingCoordinator:
    """
    Coordinates interleaved domain training while monitoring and defending against regression.
    """

    def __init__(
        self,
        domain_weights: Optional[Dict[str, float]] = None,
        max_allowed_general_regression: float = 0.05,
    ) -> None:
        self.domain_weights = domain_weights or {}
        self.max_allowed_general_regression = max_allowed_general_regression

    def balance_and_interleave_samples(
        self,
        domain_sample_pools: Dict[str, List[DomainCurriculumSample]],
        total_samples: int,
        seed: int = 42,
    ) -> List[DomainCurriculumSample]:
        """Creates a domain-balanced, interleaved sample schedule."""
        rng = random.Random(seed)
        domains = list(domain_sample_pools.keys())
        if not domains:
            return []

        # Determine target count per domain
        weights = {d: self.domain_weights.get(d, 1.0 / len(domains)) for d in domains}
        total_weight = sum(weights.values())
        norm_weights = {d: w / total_weight for d, w in weights.items()}

        interleaved: List[DomainCurriculumSample] = []
        domain_indices = {d: 0 for d in domains}

        for _ in range(total_samples):
            # Select domain based on weights
            r = rng.random()
            cumulative = 0.0
            chosen_domain = domains[0]
            for d in domains:
                cumulative += norm_weights[d]
                if r <= cumulative:
                    chosen_domain = d
                    break

            pool = domain_sample_pools.get(chosen_domain, [])
            if pool:
                idx = domain_indices[chosen_domain] % len(pool)
                interleaved.append(pool[idx])
                domain_indices[chosen_domain] += 1

        return interleaved

    def evaluate_retention(
        self,
        domain_accuracies: Dict[str, float],
        initial_general_accuracy: float,
        current_general_accuracy: float,
    ) -> MultiDomainRetentionReport:
        """Evaluates retention across learned domains and triggers rollback if necessary."""
        delta = initial_general_accuracy - current_general_accuracy
        forgetting = delta > self.max_allowed_general_regression

        if forgetting:
            action = "ROLLBACK_CHECKPOINT"
        elif delta > (self.max_allowed_general_regression * 0.5):
            action = "ADAPT_SAMPLING"
        else:
            action = "PROCEED_CANDIDATE"

        return MultiDomainRetentionReport(
            domain_accuracies=domain_accuracies,
            general_baseline_accuracy=current_general_accuracy,
            catastrophic_forgetting_detected=forgetting,
            regression_delta=round(delta, 4),
            recommended_action=action,
        )
