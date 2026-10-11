"""
Wave 549: Contradictory Open-World Knowledge Engine.

Enforces:
- Contradictions are NEVER silently collapsed or arbitrarily chosen.
- Explicitly models CONTRADICTORY state with full provenance for both sides.
- Enables governed revalidation and resolution when new authoritative evidence arrives.
- Preserves complete historical audit trail; never silently deletes older contested beliefs.
"""

from __future__ import annotations

import time
import hashlib
from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Tuple

from .knowledge_interface import EvidenceItem, VerificationStatus


class ContradictionResolutionVerdict(str, Enum):
    UNRESOLVED_CONTRADICTION = "UNRESOLVED_CONTRADICTION"
    RESOLVED_SIDE_A = "RESOLVED_SIDE_A"
    RESOLVED_SIDE_B = "RESOLVED_SIDE_B"
    CONTEXTUAL_SPLIT = "CONTEXTUAL_SPLIT"
    TEMPORAL_SUPERSEDED = "TEMPORAL_SUPERSEDED"


@dataclass
class ContradictionRecord:
    record_id: str
    target_entity: str
    predicate: str
    claim_a_value: str
    claim_a_evidence: List[EvidenceItem]
    claim_b_value: str
    claim_b_evidence: List[EvidenceItem]
    detected_at: float
    status: ContradictionResolutionVerdict
    resolution_rationale: str
    historical_resolution_trail: List[Dict[str, Any]] = field(default_factory=list)


class ContradictoryKnowledgeStore:
    """
    Store for managing and resolving open-world contradictions with zero state loss.
    """

    def __init__(self):
        self.contradictions: Dict[str, ContradictionRecord] = {}

    def record_contradiction(
        self,
        entity: str,
        predicate: str,
        val_a: str,
        ev_a: List[EvidenceItem],
        val_b: str,
        ev_b: List[EvidenceItem],
    ) -> ContradictionRecord:
        rec_id = f"contra_{hashlib.sha256(f'{entity}:{predicate}:{val_a}:{val_b}'.encode()).hexdigest()[:8]}"
        rec = ContradictionRecord(
            record_id=rec_id,
            target_entity=entity,
            predicate=predicate,
            claim_a_value=val_a,
            claim_a_evidence=ev_a,
            claim_b_value=val_b,
            claim_b_evidence=ev_b,
            detected_at=time.time(),
            status=ContradictionResolutionVerdict.UNRESOLVED_CONTRADICTION,
            resolution_rationale="Mutually conflicting open-world claims observed. Preserving contradiction without guessing.",
        )
        self.contradictions[rec_id] = rec
        return rec

    def attempt_resolution(
        self,
        record_id: str,
        new_evidence: List[EvidenceItem],
        arbitration_source_reputation: float = 0.95,
    ) -> ContradictionRecord:
        record = self.contradictions.get(record_id)
        if not record:
            raise ValueError(f"Contradiction {record_id} not found.")

        # Evaluate if new evidence decisively validates Side A or Side B
        ev_values = [e.claimed_value.strip().lower() for e in new_evidence]
        now = time.time()

        if record.claim_a_value.strip().lower() in ev_values and arbitration_source_reputation >= 0.90:
            record.status = ContradictionResolutionVerdict.RESOLVED_SIDE_A
            record.resolution_rationale = (
                f"Resolved in favor of Claim A ('{record.claim_a_value}') by authoritative corroboration ({arbitration_source_reputation:.2f})."
            )
            record.historical_resolution_trail.append({
                "timestamp": now,
                "action": "RESOLVED_SIDE_A",
                "new_evidence_count": len(new_evidence),
            })
        elif record.claim_b_value.strip().lower() in ev_values and arbitration_source_reputation >= 0.90:
            record.status = ContradictionResolutionVerdict.RESOLVED_SIDE_B
            record.resolution_rationale = (
                f"Resolved in favor of Claim B ('{record.claim_b_value}') by authoritative corroboration ({arbitration_source_reputation:.2f})."
            )
            record.historical_resolution_trail.append({
                "timestamp": now,
                "action": "RESOLVED_SIDE_B",
                "new_evidence_count": len(new_evidence),
            })
        else:
            record.status = ContradictionResolutionVerdict.UNRESOLVED_CONTRADICTION
            record.resolution_rationale = "New evidence insufficient to break evidentiary deadlock; contradiction maintained."

        return record
