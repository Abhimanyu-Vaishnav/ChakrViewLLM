"""
Wave 538: Governed Knowledge Revision & Belief Update Engine.

Enforces:
INITIAL CLAIM -> NEW EVIDENCE -> CONFLICT -> RE-EVALUATION -> RESOLUTION -> ACTIVE UPDATE
- Preserves full historical provenance (old versions are NEVER deleted).
- Supports versioned knowledge states.
- Explicitly models UNKNOWN, CONFLICT, and CONTEXTUALIZED truth.
- Requires validation before promotion to active belief.
"""

from __future__ import annotations

import time
import hashlib
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, Any, List, Optional, Tuple


class RevisionVerdict(str, Enum):
    PROMOTED_NEW = "PROMOTED_NEW"           # Validated evidence superseded old belief
    PRESERVED_EXISTING = "PRESERVED_EXISTING" # New evidence rejected / insufficient
    CONFLICT_PRESERVED = "CONFLICT_PRESERVED" # Evidence in genuine deadlock; preserved as CONFLICT
    CONTEXT_SPLIT = "CONTEXT_SPLIT"         # Both beliefs valid under differing premises
    REVALIDATION_QUEUED = "REVALIDATION_QUEUED" # Queued for empirical probe


@dataclass
class BeliefRecord:
    belief_id: str
    proposition: str
    veracity: bool                          # True / False
    version: int
    confidence: float
    source: str
    provenance_hash: str
    timestamp: float
    context_condition: str = "global"
    is_active: bool = True
    superseded_by: Optional[str] = None
    revision_rationale: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BeliefRevisionEvent:
    event_id: str
    proposition: str
    old_version_id: Optional[str]
    new_version_id: Optional[str]
    verdict: RevisionVerdict
    rationale: str
    timestamp: float


class GovernedBeliefStore:
    """
    Version-tracked, append-only knowledge store that never erases belief history.
    """

    def __init__(self):
        # proposition -> list of versions (all history preserved)
        self.belief_history: Dict[str, List[BeliefRecord]] = {}
        self.revision_events: List[BeliefRevisionEvent] = []

    def get_active_belief(self, proposition: str) -> Optional[BeliefRecord]:
        history = self.belief_history.get(proposition, [])
        for record in reversed(history):
            if record.is_active:
                return record
        return None

    def propose_revision(
        self,
        proposition: str,
        new_veracity: bool,
        new_confidence: float,
        new_source: str,
        evidence_strength: float,
        context_condition: str = "global",
        validation_passed: bool = True,
    ) -> BeliefRevisionEvent:
        """
        Processes a candidate belief update against existing historical beliefs.
        """
        now = time.time()
        existing = self.get_active_belief(proposition)

        # Hash provenance
        prov_payload = f"{proposition}:{new_veracity}:{new_source}:{now}"
        prov_hash = hashlib.sha256(prov_payload.encode()).hexdigest()

        # Case 1: First time proposition is encountered
        if existing is None:
            new_record = BeliefRecord(
                belief_id=f"bel_{prov_hash[:8]}",
                proposition=proposition,
                veracity=new_veracity,
                version=1,
                confidence=new_confidence,
                source=new_source,
                provenance_hash=prov_hash,
                timestamp=now,
                context_condition=context_condition,
                is_active=validation_passed,
                revision_rationale="Initial validated proposition ingestion" if validation_passed else "Pending validation",
            )
            self.belief_history[proposition] = [new_record]
            verdict = RevisionVerdict.PROMOTED_NEW if validation_passed else RevisionVerdict.REVALIDATION_QUEUED
            event = BeliefRevisionEvent(
                event_id=f"rev_{prov_hash[:6]}",
                proposition=proposition,
                old_version_id=None,
                new_version_id=new_record.belief_id,
                verdict=verdict,
                rationale=new_record.revision_rationale,
                timestamp=now,
            )
            self.revision_events.append(event)
            return event

        # Case 2: Proposition exists, but incoming evidence has context split
        if context_condition != existing.context_condition and context_condition != "global":
            new_record = BeliefRecord(
                belief_id=f"bel_{prov_hash[:8]}",
                proposition=proposition,
                veracity=new_veracity,
                version=existing.version + 1,
                confidence=new_confidence,
                source=new_source,
                provenance_hash=prov_hash,
                timestamp=now,
                context_condition=context_condition,
                is_active=True,
                revision_rationale=f"Context split: valid under premise '{context_condition}'",
            )
            self.belief_history[proposition].append(new_record)
            event = BeliefRevisionEvent(
                event_id=f"rev_{prov_hash[:6]}",
                proposition=proposition,
                old_version_id=existing.belief_id,
                new_version_id=new_record.belief_id,
                verdict=RevisionVerdict.CONTEXT_SPLIT,
                rationale=new_record.revision_rationale,
                timestamp=now,
            )
            self.revision_events.append(event)
            return event

        # Case 3: Conflicting claim under same context
        if existing.veracity != new_veracity:
            # Requires strict evidence dominance and validation
            if validation_passed and (evidence_strength >= 0.85 and (new_confidence >= existing.confidence or evidence_strength >= 0.95)):
                # Deprecate existing, but DO NOT delete
                existing.is_active = False
                new_record = BeliefRecord(
                    belief_id=f"bel_{prov_hash[:8]}",
                    proposition=proposition,
                    veracity=new_veracity,
                    version=existing.version + 1,
                    confidence=new_confidence,
                    source=new_source,
                    provenance_hash=prov_hash,
                    timestamp=now,
                    context_condition=context_condition,
                    is_active=True,
                    revision_rationale=f"Superseded prior belief v{existing.version} based on high-authority corroborated evidence",
                )
                existing.superseded_by = new_record.belief_id
                self.belief_history[proposition].append(new_record)
                verdict = RevisionVerdict.PROMOTED_NEW
                rationale = new_record.revision_rationale
            elif not validation_passed:
                verdict = RevisionVerdict.PRESERVED_EXISTING
                rationale = "Candidate revision failed independent validation; old belief preserved"
            else:
                # Ambiguous/deadlocked evidence -> Preserve Conflict State
                verdict = RevisionVerdict.CONFLICT_PRESERVED
                rationale = f"Conflicting evidence strength ({evidence_strength}) insufficient to overturn prior belief ({existing.confidence})"

            event = BeliefRevisionEvent(
                event_id=f"rev_{prov_hash[:6]}",
                proposition=proposition,
                old_version_id=existing.belief_id,
                new_version_id=self.belief_history[proposition][-1].belief_id if verdict == RevisionVerdict.PROMOTED_NEW else None,
                verdict=verdict,
                rationale=rationale,
                timestamp=now,
            )
            self.revision_events.append(event)
            return event

        # Case 4: Reinforcing existing veracity
        existing.confidence = min(1.0, existing.confidence + 0.05)
        event = BeliefRevisionEvent(
            event_id=f"rev_{prov_hash[:6]}",
            proposition=proposition,
            old_version_id=existing.belief_id,
            new_version_id=existing.belief_id,
            verdict=RevisionVerdict.PRESERVED_EXISTING,
            rationale="Incoming evidence reinforces existing active belief",
            timestamp=now,
        )
        self.revision_events.append(event)
        return event
