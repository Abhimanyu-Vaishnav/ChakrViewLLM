"""
Memory Contradiction Subsystem for ChakrView Continual Cognition (Step 24).

Detects, tracks, and manages factual conflicts across memory stores:
- Avoids silent destructive overwrites.
- Preserves both conflicting records with explicit contradiction markers.
- Maintains formal resolution lifecycles: UNRESOLVED -> UNDER_REVIEW -> RESOLVED | PERSISTENT_CONFLICT.

CRITICAL PRINCIPLES:
1. DATA != AUTHORITY, REASONING != AUTHORITY.
   Contradiction detection flags conflicts for governance or critical review.
2. Contradicted memories have their verification status downgraded and
   retrieval score penalized, but are NOT deleted without explicit policy directive.
"""

from dataclasses import asdict
import re
import time
from typing import Dict, List, Optional, Any, Tuple
import uuid

from chakrview.memory.models import (
    MemoryContradiction,
    ContradictionResolutionState,
    MemoryProvenanceSource,
    MemoryVerificationState,
    SemanticMemory,
)
from chakrview.memory.semantic import SemanticMemoryStore


class ContradictionManager:
    """
    Manages contradiction detection, recording, and governed resolution.
    """

    def __init__(self, semantic_store: Optional[SemanticMemoryStore] = None) -> None:
        self.semantic_store = semantic_store
        # _contradictions[tenant_id][contradiction_id] = MemoryContradiction
        self._contradictions: Dict[str, Dict[str, MemoryContradiction]] = {}

    def record_contradiction(
        self,
        tenant_id: str,
        conflicting_memory_ids: List[str],
        contradiction_type: str = "attribute_conflict",
        severity: str = "MEDIUM",
        provenance: MemoryProvenanceSource = MemoryProvenanceSource.SYSTEM_OBSERVED,
        notes: str = "",
    ) -> MemoryContradiction:
        """Record a new factual conflict across memory records."""
        if not tenant_id:
            raise ValueError("tenant_id cannot be empty.")
        if len(conflicting_memory_ids) < 2:
            raise ValueError("A contradiction requires at least 2 conflicting memory IDs.")

        cid = f"con_{uuid.uuid4().hex[:12]}"
        contradiction = MemoryContradiction(
            contradiction_id=cid,
            tenant_id=tenant_id,
            conflicting_memory_ids=list(conflicting_memory_ids),
            contradiction_type=contradiction_type,
            severity=severity,
            provenance=provenance,
            timestamp=time.time(),
            resolution_state=ContradictionResolutionState.UNRESOLVED,
            resolution_evidence=[],
            resolved_memory_id=None,
            notes=notes,
        )

        if tenant_id not in self._contradictions:
            self._contradictions[tenant_id] = {}
        self._contradictions[tenant_id][cid] = contradiction

        # If semantic store is linked, mark affected memories
        if self.semantic_store:
            for mid in conflicting_memory_ids:
                self.semantic_store.link_contradiction(tenant_id, mid, cid)

        return contradiction

    def detect_semantic_conflicts(
        self,
        tenant_id: str,
        candidate_subject: str,
        candidate_predicate: str,
        candidate_value: str,
    ) -> List[Tuple[SemanticMemory, str]]:
        """
        Check existing semantic memories for a tenant to find conflicting values
        for the same subject and predicate.
        Returns list of (conflicting_memory, conflict_reason).
        """
        if not self.semantic_store:
            return []

        existing_mems = self.semantic_store.find_by_triple(
            tenant_id=tenant_id,
            subject=candidate_subject,
            predicate=candidate_predicate,
            active_only=True,
        )

        conflicts: List[Tuple[SemanticMemory, str]] = []
        c_val_norm = candidate_value.strip().lower()

        for mem in existing_mems:
            e_val_norm = mem.object_value.strip().lower()
            if e_val_norm != c_val_norm:
                reason = f"Conflicting object value for ({mem.subject} {mem.predicate}): '{mem.object_value}' vs '{candidate_value}'"
                conflicts.append((mem, reason))

        return conflicts

    def resolve_contradiction(
        self,
        tenant_id: str,
        contradiction_id: str,
        resolved_memory_id: Optional[str],
        resolution_evidence: List[str],
        strategy: str = "explicit_verification",
        notes: str = "",
    ) -> bool:
        """
        Formally resolve a detected contradiction with audit trail.
        """
        tenant_cons = self._contradictions.get(tenant_id, {})
        con = tenant_cons.get(contradiction_id)
        if not con:
            return False

        con.resolution_state = ContradictionResolutionState.RESOLVED
        con.resolved_memory_id = resolved_memory_id
        con.resolution_evidence = list(resolution_evidence)
        con.notes = f"{con.notes} | Resolved via {strategy}: {notes}".strip(" |")

        # Update semantic store states if memory was resolved
        if self.semantic_store and resolved_memory_id:
            for mid in con.conflicting_memory_ids:
                mem = self.semantic_store.get_memory(tenant_id, mid)
                if mem:
                    if mid == resolved_memory_id:
                        mem.verification_status = MemoryVerificationState.VERIFIED
                    else:
                        mem.verification_status = MemoryVerificationState.REJECTED

        return True

    def mark_persistent_conflict(
        self,
        tenant_id: str,
        contradiction_id: str,
        reason: str,
    ) -> bool:
        """Flag contradiction as persistent unresolvable disagreement."""
        tenant_cons = self._contradictions.get(tenant_id, {})
        con = tenant_cons.get(contradiction_id)
        if not con:
            return False
        con.resolution_state = ContradictionResolutionState.PERSISTENT_CONFLICT
        con.notes = f"{con.notes} | Marked persistent conflict: {reason}".strip(" |")
        return True

    def get_contradiction(self, tenant_id: str, contradiction_id: str) -> Optional[MemoryContradiction]:
        """Fetch contradiction enforcing tenant boundary."""
        return self._contradictions.get(tenant_id, {}) .get(contradiction_id)

    def list_unresolved(self, tenant_id: str) -> List[MemoryContradiction]:
        """List unresolved contradictions for a tenant."""
        tenant_cons = self._contradictions.get(tenant_id, {})
        return [
            c for c in tenant_cons.values()
            if c.resolution_state in (ContradictionResolutionState.UNRESOLVED, ContradictionResolutionState.UNDER_REVIEW)
        ]

    def list_all(self, tenant_id: str) -> List[MemoryContradiction]:
        """List all contradiction records for a tenant."""
        tenant_cons = self._contradictions.get(tenant_id, {})
        return sorted(list(tenant_cons.values()), key=lambda c: c.timestamp, reverse=True)

    def to_dict(self) -> Dict[str, Any]:
        """Export serialized contradiction data."""
        out: Dict[str, Any] = {}
        for t_id, con_dict in self._contradictions.items():
            out[t_id] = {cid: con.to_dict() for cid, con in con_dict.items()}
        return out

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        semantic_store: Optional[SemanticMemoryStore] = None,
    ) -> "ContradictionManager":
        """Reconstruct manager from serialized snapshot."""
        mgr = cls(semantic_store=semantic_store)
        for t_id, con_dict in data.items():
            mgr._contradictions[t_id] = {}
            for cid, c_data in con_dict.items():
                mgr._contradictions[t_id][cid] = MemoryContradiction.from_dict(c_data)
        return mgr
