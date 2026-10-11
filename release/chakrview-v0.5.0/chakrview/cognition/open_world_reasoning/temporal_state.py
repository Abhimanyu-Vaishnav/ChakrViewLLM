"""
Wave 577: Temporal World-State Reasoning Engine.

Architectural Guarantees:
- Models:
  EVENT_TIME: When the real-world change occurred.
  PUBLICATION_TIME: When the document reported it.
  RETRIEVAL_TIME: When ChakrView ingested the document.
  VALID_FROM & VALID_UNTIL: Active interval of world-state truth.
- Resolves time-dependent queries (e.g., "Who was CEO in 2021?" vs "Who is CEO now?").
- Tracks state transitions: State A -> Event -> State B -> Event -> State C.
"""

from __future__ import annotations

import time
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple


class TemporalScope(str, Enum):
    HISTORICAL = "HISTORICAL"
    CURRENT = "CURRENT"
    FUTURE = "FUTURE"
    TIMELESS = "TIMELESS"


@dataclass
class TemporalStateTransition:
    entity: str
    attribute: str
    value: str
    event_timestamp: float
    valid_from: float
    valid_until: Optional[float]
    publication_timestamp: float
    retrieval_timestamp: float
    provenance_doc_id: str


@dataclass
class TemporalWorldStateOutcome:
    query_entity: str
    query_attribute: str
    target_query_time: float
    temporal_scope: TemporalScope
    resolved_value: Optional[str]
    transition_sequence: List[TemporalStateTransition]
    is_superseded_or_stale: bool
    confidence: float
    rationale: str


class TemporalWorldStateReasoner:
    """
    Evaluates temporal queries against chronological state transitions,
    preventing confusion between publication time, event time, and retrieval time.
    """

    def __init__(self):
        self.transitions: List[TemporalStateTransition] = []

    def record_transition(self, trans: TemporalStateTransition) -> None:
        self.transitions.append(trans)

    def query_state_at(
        self,
        entity: str,
        attribute: str,
        query_timestamp: float,
        is_current_query: bool = False,
    ) -> TemporalWorldStateOutcome:
        ent_clean = entity.strip().lower()
        attr_clean = attribute.strip().lower()

        matching = [
            t for t in self.transitions
            if t.entity.strip().lower() == ent_clean
            and t.attribute.strip().lower() == attr_clean
        ]

        if not matching:
            return TemporalWorldStateOutcome(
                query_entity=entity,
                query_attribute=attribute,
                target_query_time=query_timestamp,
                temporal_scope=TemporalScope.CURRENT if is_current_query else TemporalScope.HISTORICAL,
                resolved_value=None,
                transition_sequence=[],
                is_superseded_or_stale=False,
                confidence=0.0,
                rationale="No temporal transitions recorded for entity-attribute.",
            )

        # Sort transitions chronologically by valid_from
        sorted_trans = sorted(matching, key=lambda x: x.valid_from)

        # Find the active transition covering query_timestamp
        active_match: Optional[TemporalStateTransition] = None
        for t in sorted_trans:
            if t.valid_from <= query_timestamp:
                if t.valid_until is None or query_timestamp < t.valid_until:
                    active_match = t

        if active_match:
            is_latest = (active_match == sorted_trans[-1])
            is_stale = is_current_query and not is_latest
            scope = TemporalScope.CURRENT if (is_current_query and is_latest) else TemporalScope.HISTORICAL

            return TemporalWorldStateOutcome(
                query_entity=entity,
                query_attribute=attribute,
                target_query_time=query_timestamp,
                temporal_scope=scope,
                resolved_value=active_match.value,
                transition_sequence=sorted_trans,
                is_superseded_or_stale=is_stale,
                confidence=0.92,
                rationale=f"Active state at t={query_timestamp} identified as '{active_match.value}' from document {active_match.provenance_doc_id}.",
            )

        # Before any recorded history or in a gap
        if query_timestamp < sorted_trans[0].valid_from:
            return TemporalWorldStateOutcome(
                query_entity=entity,
                query_attribute=attribute,
                target_query_time=query_timestamp,
                temporal_scope=TemporalScope.HISTORICAL,
                resolved_value=None,
                transition_sequence=sorted_trans,
                is_superseded_or_stale=True,
                confidence=0.0,
                rationale="Query timestamp precedes the earliest recorded event window.",
            )

        # Historical state that was superseded
        last_t = sorted_trans[-1]
        return TemporalWorldStateOutcome(
            query_entity=entity,
            query_attribute=attribute,
            target_query_time=query_timestamp,
            temporal_scope=TemporalScope.HISTORICAL,
            resolved_value=last_t.value,
            transition_sequence=sorted_trans,
            is_superseded_or_stale=False,
            confidence=0.85,
            rationale=f"Resolved to latest available state '{last_t.value}' following past transitions.",
        )
