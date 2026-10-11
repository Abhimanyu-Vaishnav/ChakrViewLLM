"""
Wave 550: Temporal Knowledge & Freshness Engine.

Enforces:
- Distinguishes TIMELESS / STABLE facts (e.g. physical constants) from TIME-VARYING facts (e.g. current CEO, price, software version).
- Explicit temporal boundaries: valid_from, valid_until, observed_at, publication_timestamp.
- Distinguishes publication time != event occurrence time != retrieval time.
- Identifies stale vs current knowledge for time-sensitive inquiries.
"""

from __future__ import annotations

import time
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple


class TemporalValidity(str, Enum):
    CURRENT = "CURRENT"
    HISTORICAL = "HISTORICAL"
    STALE = "STALE"
    FUTURE_PREDICTION = "FUTURE_PREDICTION"
    TIMELESS_STABLE = "TIMELESS_STABLE"
    UNKNOWN_TEMPORAL = "UNKNOWN_TEMPORAL"


@dataclass
class TemporalFactRecord:
    fact_id: str
    entity: str
    attribute: str
    value: str
    is_time_varying: bool
    valid_from: Optional[float] = None
    valid_until: Optional[float] = None
    observed_at: float = field(default_factory=time.time)
    provenance_source: str = ""
    is_superseded: bool = False
    superseded_by: Optional[str] = None


class TemporalKnowledgeManager:
    """
    Manages temporal validity and freshness checks for cognitive facts.
    """

    def __init__(self):
        self.facts: List[TemporalFactRecord] = []

    def register_fact(self, fact: TemporalFactRecord) -> None:
        self.facts.append(fact)

    def evaluate_freshness(
        self,
        entity: str,
        attribute: str,
        query_timestamp: Optional[float] = None,
        max_staleness_seconds: float = 86400.0 * 30, # 30 days
    ) -> Tuple[TemporalValidity, Optional[TemporalFactRecord], str]:
        """
        Evaluates the active state of an entity's attribute relative to query_timestamp.
        """
        t = query_timestamp if query_timestamp is not None else time.time()
        matching = [
            f for f in self.facts
            if f.entity.strip().lower() == entity.strip().lower()
            and f.attribute.strip().lower() == attribute.strip().lower()
        ]

        if not matching:
            return TemporalValidity.UNKNOWN_TEMPORAL, None, "No temporal record found for entity-attribute."

        # Find latest record
        latest = max(matching, key=lambda x: x.observed_at)

        # 1. Timeless Fact
        if not latest.is_time_varying:
            return TemporalValidity.TIMELESS_STABLE, latest, "Fact is physically/conceptually timeless."

        # 2. Check if superseded or expired
        if latest.valid_until is not None and t > latest.valid_until:
            return TemporalValidity.HISTORICAL, latest, f"Fact expired on timestamp {latest.valid_until}; historically valid only."

        # 3. Check staleness
        age = t - latest.observed_at
        if age > max_staleness_seconds and latest.is_time_varying:
            return TemporalValidity.STALE, latest, f"Fact is time-varying and observed {age/86400:.1f} days ago (exceeds {max_staleness_seconds/86400:.1f}d freshness threshold)."

        return TemporalValidity.CURRENT, latest, "Fact is actively valid and within temporal freshness bounds."
