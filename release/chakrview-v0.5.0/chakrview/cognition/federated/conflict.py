"""
Federated Conflict Detection & Resolution Subsystem (Step 26).

Enforces:
1. Explicit Conflict Detection: Disagreements between agent claims are identified
   and instantiated as auditable conflict records.
2. Minority Evidence Preservation: Minority viewpoints containing unresolved evidence
   or counter-evidence are NEVER discarded or outvoted.
3. Structured Resolution Lifecycle:
   AGREEMENT -> PARTIAL_AGREEMENT -> CONFLICT -> UNRESOLVED -> RESOLVED | INSUFFICIENT_EVIDENCE
"""

from typing import Dict, List, Optional, Any
import uuid

from chakrview.cognition.federated.models import (
    ConflictState,
    FederatedConflictRecord,
    AgentMessage,
)
from chakrview.cognition.critical.engine import CriticalThinkingEngine


class FederatedConflictResolver:
    """
    Evaluates agent outputs, identifies conflicting claims, and tracks conflict resolution.
    """

    def __init__(self, critical_engine: Optional[CriticalThinkingEngine] = None) -> None:
        self.critical_engine = critical_engine or CriticalThinkingEngine()

    def detect_conflicts(
        self,
        task_id: str,
        messages: List[AgentMessage],
        external_contradictions: Optional[List[Dict[str, Any]]] = None,
    ) -> List[FederatedConflictRecord]:
        """
        Scan agent messages and external contradictions to identify factual or logical conflicts.
        """
        conflicts: List[FederatedConflictRecord] = []
        claims: List[Dict[str, Any]] = []

        for msg in messages:
            p = msg.payload
            claim = p.get("claim") or p.get("synthesis")
            if claim:
                claims.append({
                    "agent_id": msg.sender_agent_id,
                    "claim": str(claim),
                    "confidence": p.get("confidence", 0.5),
                    "evidence": p.get("evidence_items", []),
                })

        # Check pairwise claims for mutual opposition or contradictory keywords
        for i in range(len(claims)):
            for j in range(i + 1, len(claims)):
                c1 = claims[i]
                c2 = claims[j]
                # Check for explicit opposition
                is_opposing = self._check_claim_opposition(c1["claim"], c2["claim"])
                if is_opposing:
                    cid = f"fcon_{uuid.uuid4().hex[:8]}"
                    record = FederatedConflictRecord(
                        conflict_id=cid,
                        task_id=task_id,
                        conflicting_agent_ids=[c1["agent_id"], c2["agent_id"]],
                        claims=[c1, c2],
                        supporting_evidence=c1["evidence"],
                        contradicting_evidence=c2["evidence"],
                        state=ConflictState.CONFLICT,
                        resolution_notes="Disagreement detected between agent claims. Both claims preserved.",
                    )
                    conflicts.append(record)

        # Ingest external contradictions from memory or critical thinking
        if external_contradictions:
            for ext in external_contradictions:
                cid = ext.get("contradiction_id", f"fcon_ext_{uuid.uuid4().hex[:6]}")
                record = FederatedConflictRecord(
                    conflict_id=cid,
                    task_id=task_id,
                    conflicting_agent_ids=ext.get("conflicting_memory_ids", ["memory_store"]),
                    claims=[{"claim": ext.get("notes", "Memory factual contradiction")}],
                    supporting_evidence=[],
                    contradicting_evidence=[],
                    state=ConflictState.UNRESOLVED,
                    resolution_notes="Imported memory/critical-thinking contradiction.",
                )
                conflicts.append(record)

        return conflicts

    def _check_claim_opposition(self, claim_a: str, claim_b: str) -> bool:
        """Heuristic check for contradictory assertions between two text claims."""
        ca = claim_a.lower()
        cb = claim_b.lower()
        # Direct negation
        if " not " in ca and " not " not in cb and ca.replace(" not ", " ") in cb:
            return True
        if " not " in cb and " not " not in ca and cb.replace(" not ", " ") in ca:
            return True
        # Contradictory polarities
        opposites = [
            ("enable", "disable"),
            ("allow", "deny"),
            ("true", "false"),
            ("supported", "unsupported"),
            ("valid", "invalid"),
            ("safe", "unsafe"),
            ("increase", "decrease"),
        ]
        for pos, neg in opposites:
            if (pos in ca and neg in cb) or (neg in ca and pos in cb):
                return True
        return False

    def evaluate_resolution(
        self,
        conflict: FederatedConflictRecord,
        counter_evidence_pool: List[Dict[str, Any]],
    ) -> ConflictState:
        """
        Evaluate whether new counter-evidence resolves or leaves the conflict unresolved.
        """
        # If counter-evidence confirms falsification of one side
        has_confirmed_ce = any(ce.get("status") == "CONFIRMED" for ce in counter_evidence_pool)
        if has_confirmed_ce:
            conflict.state = ConflictState.RESOLVED
            conflict.resolution_notes = "Conflict resolved by confirmed empirical counter-evidence."
            return ConflictState.RESOLVED

        # If evidence is absent on both sides
        if not conflict.supporting_evidence and not conflict.contradicting_evidence:
            conflict.state = ConflictState.INSUFFICIENT_EVIDENCE
            conflict.resolution_notes = "Insufficient evidence to resolve disagreement."
            return ConflictState.INSUFFICIENT_EVIDENCE

        # Unresolved conflict remains tracked
        conflict.state = ConflictState.UNRESOLVED
        conflict.resolution_notes = "Disagreement remains unresolved; epistemic uncertainty preserved."
        return ConflictState.UNRESOLVED
