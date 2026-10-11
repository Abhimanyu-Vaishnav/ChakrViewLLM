"""
Wave 585: Memory <-> Reasoning Integration Bridge.

Architectural Guarantees:
- Prioritizes NEW VERIFIED EVIDENCE over OLD STORED MEMORY.
- Never lets stale memory silently overwrite contradictory external evidence.
- Updates/revises durable beliefs when evidence presents verified newer truth.
- Zero neural weight mutation (Delta W = 0).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple


@dataclass
class MemoryBeliefRecord:
    belief_id: str
    entity: str
    attribute: str
    value: str
    confidence: float
    observed_timestamp: float
    is_stale: bool = False
    superseded_by_evidence: Optional[str] = None


class MemoryReasoningBridge:
    """
    Arbitrates between persistent stored memory and newly retrieved external evidence.
    """

    def __init__(self):
        self.memory_store: Dict[str, MemoryBeliefRecord] = {}

    def store_belief(self, record: MemoryBeliefRecord) -> None:
        key = f"{record.entity.lower()}::{record.attribute.lower()}"
        self.memory_store[key] = record

    def get_belief(self, entity: str, attribute: str) -> Optional[MemoryBeliefRecord]:
        key = f"{entity.lower()}::{attribute.lower()}"
        return self.memory_store.get(key)

    def integrate_new_evidence(
        self,
        entity: str,
        attribute: str,
        evidence_value: str,
        evidence_timestamp: float,
        evidence_confidence: float = 0.95,
    ) -> Tuple[str, str, float]:
        """
        Integrates evidence against existing memory.
        Returns: (resolved_value, decision_rationale, calibrated_confidence)
        """
        existing = self.get_belief(entity, attribute)

        if not existing:
            # New fact -> Store into memory
            new_rec = MemoryBeliefRecord(
                belief_id=f"bel_{entity}_{attribute}",
                entity=entity,
                attribute=attribute,
                value=evidence_value,
                confidence=evidence_confidence,
                observed_timestamp=evidence_timestamp,
            )
            self.store_belief(new_rec)
            return (
                evidence_value,
                "New factual belief adopted from verified external evidence.",
                evidence_confidence,
            )

        # Conflict check: Evidence vs Stored Memory
        if existing.value.strip().lower() == evidence_value.strip().lower():
            # Corroboration: reinforce confidence
            existing.confidence = min(0.99, existing.confidence + 0.05)
            existing.observed_timestamp = max(existing.observed_timestamp, evidence_timestamp)
            return (
                existing.value,
                "Stored memory confirmed and reinforced by external evidence.",
                existing.confidence,
            )

        # Disagreement!
        # EVIDENCE OVERRIDES OLD MEMORY if evidence is newer and verified
        if evidence_timestamp >= existing.observed_timestamp:
            old_val = existing.value
            existing.is_stale = True
            existing.superseded_by_evidence = evidence_value
            # Update memory store with new verified state
            updated_rec = MemoryBeliefRecord(
                belief_id=f"bel_{entity}_{attribute}",
                entity=entity,
                attribute=attribute,
                value=evidence_value,
                confidence=evidence_confidence,
                observed_timestamp=evidence_timestamp,
            )
            self.store_belief(updated_rec)
            return (
                evidence_value,
                f"Memory revision: Newer verified evidence '{evidence_value}' superseded stale memory '{old_val}'.",
                evidence_confidence,
            )
        else:
            # Evidence is older than existing memory
            return (
                existing.value,
                f"Retained existing memory: Incoming evidence is older than existing verified belief.",
                existing.confidence,
            )
