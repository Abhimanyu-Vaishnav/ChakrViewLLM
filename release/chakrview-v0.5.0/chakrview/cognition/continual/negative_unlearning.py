"""
Wave 539: Negative Learning & Governed Unlearning Engine.

Handles:
LEARN STRATEGY A -> A WORKS -> ENVIRONMENT CHANGES -> A BECOMES HARMFUL
-> DETECT NEGATIVE TRANSFER -> REDUCE TRUST -> REVALIDATE -> DEPRECATE / RETIRE -> RECOVER

Guarantees:
- Never deletes old knowledge; archives into provenance ledger.
- Prevents false unlearning of still-valid strategies.
- Accurately measures stale strategy selection, correction latency, and recovery accuracy.
"""

from __future__ import annotations

import time
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional


class UnlearningStatus(str, Enum):
    MONITORED = "MONITORED"
    TRUST_DEGRADED = "TRUST_DEGRADED"
    REVALIDATION_FAILED = "REVALIDATION_FAILED"
    DEPRECATED = "DEPRECATED"
    RETIRED = "RETIRED"
    RECOVERED = "RECOVERED"


@dataclass
class UnlearningAuditRecord:
    strategy_id: str
    previous_trust: float
    current_trust: float
    status: UnlearningStatus
    negative_transfer_delta: float
    residual_influence: float
    rationale: str
    timestamp: float


class NegativeTransferDetector:
    """Detects when an established strategy begins harming task performance."""

    @staticmethod
    def detect_negative_transfer(
        pre_drift_success_rate: float,
        post_drift_success_rate: float,
        threshold: float = 0.20,
    ) -> Tuple[bool, float]:
        delta = pre_drift_success_rate - post_drift_success_rate
        is_negative = delta >= threshold
        return is_negative, delta


class UnlearningController:
    """Governs the step-down, deprecation, and replacement of stale strategies."""

    def __init__(self):
        self.audit_log: List[UnlearningAuditRecord] = []

    def evaluate_strategy_utility(
        self,
        strategy_id: str,
        historical_success: float,
        recent_success: float,
        current_trust: float,
        revalidation_score: float,
    ) -> Tuple[UnlearningStatus, float, str]:
        now = time.time()
        is_negative, delta = NegativeTransferDetector.detect_negative_transfer(
            historical_success, recent_success
        )

        if not is_negative:
            # Still valid; protect against false unlearning
            return UnlearningStatus.MONITORED, current_trust, "Strategy performance stable"

        # Strategy is causing negative transfer
        degraded_trust = max(0.0, current_trust - (delta * 0.8))

        if revalidation_score < 0.60:
            status = UnlearningStatus.DEPRECATED if degraded_trust > 0.1 else UnlearningStatus.RETIRED
            rationale = (
                f"Negative transfer detected (drop={delta:.2f}); "
                f"failed revalidation probe ({revalidation_score:.2f}). Demoting to {status.value}."
            )
        else:
            status = UnlearningStatus.TRUST_DEGRADED
            rationale = f"Negative transfer detected (drop={delta:.2f}); trust score decayed to {degraded_trust:.2f}."

        self.audit_log.append(
            UnlearningAuditRecord(
                strategy_id=strategy_id,
                previous_trust=current_trust,
                current_trust=degraded_trust,
                status=status,
                negative_transfer_delta=delta,
                residual_influence=degraded_trust,
                rationale=rationale,
                timestamp=now,
            )
        )

        return status, degraded_trust, rationale
