"""
Semantic Memory Subsystem for ChakrView Continual Cognition (Step 24).

Maintains versioned declarative knowledge extracted from verified experiences
or trusted sources.

CRITICAL ARCHITECTURAL CONSTRAINTS:
1. Versioned & Non-Destructive: Conflicting or updated information increments
   version numbers and links ancestors rather than destructively erasing history.
2. Verification Lifecycle: Newly derived propositions start as CANDIDATE or UNVERIFIED.
   They NEVER auto-promote to VERIFIED without formal evaluation.
3. Strict Tenant Isolation: Tenant A can NEVER query or mutate Tenant B semantic memories.
"""

from dataclasses import asdict
import time
from typing import Dict, List, Optional, Any
import uuid

from chakrview.memory.models import (
    SemanticMemory,
    MemoryProvenanceSource,
    MemoryVerificationState,
    MemoryLifecycleStatus,
)


class SemanticMemoryStore:
    """
    In-memory semantic store structured by subject-predicate-object propositions,
    indexed strictly by tenant with revision lineage.
    """

    def __init__(self) -> None:
        # _store[tenant_id][memory_id] = SemanticMemory
        self._store: Dict[str, Dict[str, SemanticMemory]] = {}

    def add_memory(
        self,
        tenant_id: str,
        subject: str,
        predicate: str,
        object_value: str,
        session_id: str = "default_session",
        provenance: MemoryProvenanceSource = MemoryProvenanceSource.SYSTEM_OBSERVED,
        confidence: float = 0.8,
        verification_status: MemoryVerificationState = MemoryVerificationState.CANDIDATE,
        lifecycle_status: MemoryLifecycleStatus = MemoryLifecycleStatus.ACTIVE,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SemanticMemory:
        """
        Add a new semantic proposition.
        """
        if not tenant_id:
            raise ValueError("tenant_id cannot be empty.")
        if not subject.strip():
            raise ValueError("subject cannot be empty.")
        if not predicate.strip():
            raise ValueError("predicate cannot be empty.")
        if not object_value.strip():
            raise ValueError("object_value cannot be empty.")

        memory_id = f"sem_{uuid.uuid4().hex[:12]}"
        now = time.time()
        mem = SemanticMemory(
            memory_id=memory_id,
            tenant_id=tenant_id,
            session_id=session_id,
            subject=subject.strip(),
            predicate=predicate.strip(),
            object_value=object_value.strip(),
            provenance=provenance,
            confidence=max(0.0, min(1.0, confidence)),
            verification_status=verification_status,
            created_at=now,
            updated_at=now,
            version=1,
            previous_version_id=None,
            contradiction_refs=[],
            lifecycle_status=lifecycle_status,
            tags=tags or [],
            metadata=metadata or {},
        )

        if tenant_id not in self._store:
            self._store[tenant_id] = {}
        self._store[tenant_id][memory_id] = mem
        return mem

    def update_memory(
        self,
        tenant_id: str,
        existing_memory_id: str,
        new_object_value: str,
        confidence: Optional[float] = None,
        verification_status: Optional[MemoryVerificationState] = None,
        provenance: Optional[MemoryProvenanceSource] = None,
        notes: str = "",
    ) -> SemanticMemory:
        """
        Non-destructively updates a memory by archiving the prior version and creating version+1.
        """
        prior = self.get_memory(tenant_id, existing_memory_id)
        if not prior:
            raise KeyError(f"Semantic memory '{existing_memory_id}' not found for tenant '{tenant_id}'.")

        # Archive or supersede prior
        prior.lifecycle_status = MemoryLifecycleStatus.ARCHIVED
        prior.updated_at = time.time()

        new_id = f"sem_{uuid.uuid4().hex[:12]}"
        now = time.time()
        new_mem = SemanticMemory(
            memory_id=new_id,
            tenant_id=tenant_id,
            session_id=prior.session_id,
            subject=prior.subject,
            predicate=prior.predicate,
            object_value=new_object_value.strip(),
            provenance=provenance or prior.provenance,
            confidence=prior.confidence if confidence is None else max(0.0, min(1.0, confidence)),
            verification_status=verification_status or prior.verification_status,
            created_at=prior.created_at,
            updated_at=now,
            version=prior.version + 1,
            previous_version_id=prior.memory_id,
            contradiction_refs=list(prior.contradiction_refs),
            lifecycle_status=MemoryLifecycleStatus.ACTIVE,
            tags=list(prior.tags),
            metadata={**prior.metadata, "supersession_notes": notes, "superseded_from": prior.memory_id},
        )

        self._store[tenant_id][new_id] = new_mem
        return new_mem

    def get_memory(self, tenant_id: str, memory_id: str) -> Optional[SemanticMemory]:
        """Fetch semantic memory with strict tenant boundary."""
        return self._store.get(tenant_id, {}).get(memory_id)

    def find_by_subject(
        self,
        tenant_id: str,
        subject: str,
        active_only: bool = True,
    ) -> List[SemanticMemory]:
        """Retrieve semantic facts about a specific subject."""
        s_norm = subject.strip().lower()
        res: List[SemanticMemory] = []
        for mem in self._store.get(tenant_id, {}).values():
            if active_only and mem.lifecycle_status != MemoryLifecycleStatus.ACTIVE:
                continue
            if mem.subject.lower() == s_norm:
                res.append(mem)
        return res

    def find_by_triple(
        self,
        tenant_id: str,
        subject: str,
        predicate: str,
        active_only: bool = True,
    ) -> List[SemanticMemory]:
        """Retrieve semantic facts matching subject and predicate."""
        s_norm = subject.strip().lower()
        p_norm = predicate.strip().lower()
        res: List[SemanticMemory] = []
        for mem in self._store.get(tenant_id, {}).values():
            if active_only and mem.lifecycle_status != MemoryLifecycleStatus.ACTIVE:
                continue
            if mem.subject.lower() == s_norm and mem.predicate.lower() == p_norm:
                res.append(mem)
        return res

    def link_contradiction(self, tenant_id: str, memory_id: str, contradiction_id: str) -> bool:
        """Associate a contradiction record with a semantic memory."""
        mem = self.get_memory(tenant_id, memory_id)
        if not mem:
            return False
        if contradiction_id not in mem.contradiction_refs:
            mem.contradiction_refs.append(contradiction_id)
            mem.verification_status = MemoryVerificationState.CONTRADICTED
            mem.updated_at = time.time()
        return True

    def list_memories(
        self,
        tenant_id: str,
        limit: int = 100,
        active_only: bool = True,
    ) -> List[SemanticMemory]:
        """List semantic memories for a tenant sorted by recency."""
        tenant_mems = self._store.get(tenant_id, {})
        res: List[SemanticMemory] = []
        for mem in sorted(tenant_mems.values(), key=lambda m: m.updated_at, reverse=True):
            if active_only and mem.lifecycle_status != MemoryLifecycleStatus.ACTIVE:
                continue
            res.append(mem)
            if len(res) >= limit:
                break
        return res

    def count(self, tenant_id: str) -> int:
        """Count memories for a tenant."""
        return len(self._store.get(tenant_id, {}))

    def clear(self, tenant_id: Optional[str] = None) -> None:
        """Clear store for tenant or completely."""
        if tenant_id:
            if tenant_id in self._store:
                self._store[tenant_id].clear()
        else:
            self._store.clear()

    def to_dict(self) -> Dict[str, Any]:
        """Export serialized snapshot across all tenants."""
        out: Dict[str, Any] = {}
        for t_id, mem_dict in self._store.items():
            out[t_id] = {m_id: mem.to_dict() for m_id, mem in mem_dict.items()}
        return out

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SemanticMemoryStore":
        """Reconstruct store from serialized snapshot."""
        store = cls()
        for t_id, mem_dict in data.items():
            store._store[t_id] = {}
            for m_id, m_data in mem_dict.items():
                store._store[t_id][m_id] = SemanticMemory.from_dict(m_data)
        return store
