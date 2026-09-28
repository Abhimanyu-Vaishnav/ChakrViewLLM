"""
Memory Lifecycle & Controlled Forgetting Subsystem for ChakrView (Step 24).

Implements deterministic, bounded memory lifecycle transitions:
- Retention
- Controlled Archival
- Expiration Sweeps
- Safety Quarantine
- Explicit Policy-Governed Deletion with Audit Trail

CRITICAL PRINCIPLES:
1. No Silent Deletion: Contradictory, unverified, or historically significant facts
   are never silently erased.
2. Quarantined memories are isolated from both inference context and training pipelines.
3. Every lifecycle transition records a timestamped audit trail.
"""

from dataclasses import dataclass, field
import time
from typing import Dict, List, Optional, Any

from chakrview.memory.models import (
    MemoryLifecycleStatus,
    MemoryVerificationState,
)
from chakrview.memory.episodic import EpisodicMemoryStore
from chakrview.memory.semantic import SemanticMemoryStore


@dataclass
class LifecycleAuditEntry:
    """Audit log entry for a memory lifecycle transition."""
    tenant_id: str
    memory_id: str
    memory_type: str  # "episodic" or "semantic"
    previous_status: str
    new_status: str
    reason: str
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tenant_id": self.tenant_id,
            "memory_id": self.memory_id,
            "memory_type": self.memory_type,
            "previous_status": self.previous_status,
            "new_status": self.new_status,
            "reason": self.reason,
            "timestamp": self.timestamp,
        }


class MemoryLifecycleManager:
    """
    Orchestrates memory retention, expiration, archival, and governed deletion.
    """

    def __init__(
        self,
        episodic_store: EpisodicMemoryStore,
        semantic_store: SemanticMemoryStore,
    ) -> None:
        self.episodic_store = episodic_store
        self.semantic_store = semantic_store
        self._audit_log: List[LifecycleAuditEntry] = []

    def archive_memory(
        self,
        tenant_id: str,
        memory_id: str,
        memory_type: str = "semantic",
        reason: str = "Superseded or retired",
    ) -> bool:
        """Move memory from ACTIVE to ARCHIVED status."""
        if memory_type == "semantic":
            mem = self.semantic_store.get_memory(tenant_id, memory_id)
            if not mem:
                return False
            prev = mem.lifecycle_status.value
            mem.lifecycle_status = MemoryLifecycleStatus.ARCHIVED
            mem.updated_at = time.time()
            self._record_audit(tenant_id, memory_id, "semantic", prev, MemoryLifecycleStatus.ARCHIVED.value, reason)
            return True
        elif memory_type == "episodic":
            ep = self.episodic_store.get_episode(tenant_id, memory_id)
            if not ep:
                return False
            prev = ep.lifecycle_status.value
            ep.lifecycle_status = MemoryLifecycleStatus.ARCHIVED
            self._record_audit(tenant_id, memory_id, "episodic", prev, MemoryLifecycleStatus.ARCHIVED.value, reason)
            return True
        return False

    def quarantine_memory(
        self,
        tenant_id: str,
        memory_id: str,
        memory_type: str = "semantic",
        reason: str = "Safety flag or suspicious injection",
    ) -> bool:
        """Quarantine memory to prevent retrieval or training ingestion."""
        if memory_type == "semantic":
            mem = self.semantic_store.get_memory(tenant_id, memory_id)
            if not mem:
                return False
            prev = mem.lifecycle_status.value
            mem.lifecycle_status = MemoryLifecycleStatus.QUARANTINED
            mem.verification_status = MemoryVerificationState.QUARANTINED
            mem.metadata["quarantine_reason"] = reason
            mem.updated_at = time.time()
            self._record_audit(tenant_id, memory_id, "semantic", prev, MemoryLifecycleStatus.QUARANTINED.value, reason)
            return True
        elif memory_type == "episodic":
            ep = self.episodic_store.get_episode(tenant_id, memory_id)
            if not ep:
                return False
            prev = ep.lifecycle_status.value
            ep.lifecycle_status = MemoryLifecycleStatus.QUARANTINED
            ep.verification_status = MemoryVerificationState.QUARANTINED
            ep.metadata["quarantine_reason"] = reason
            self._record_audit(tenant_id, memory_id, "episodic", prev, MemoryLifecycleStatus.QUARANTINED.value, reason)
            return True
        return False

    def expire_memories(
        self,
        tenant_id: str,
        max_age_seconds: float,
    ) -> int:
        """
        Transition memories older than max_age_seconds to EXPIRED status.
        """
        expired_count = 0
        now = time.time()

        # Check episodic
        episodes = self.episodic_store.list_episodes(tenant_id, limit=500, lifecycle_filter=MemoryLifecycleStatus.ACTIVE)
        for ep in episodes:
            if (now - ep.timestamp) > max_age_seconds:
                prev = ep.lifecycle_status.value
                ep.lifecycle_status = MemoryLifecycleStatus.EXPIRED
                self._record_audit(
                    tenant_id, ep.episode_id, "episodic", prev,
                    MemoryLifecycleStatus.EXPIRED.value, f"Age exceeded {max_age_seconds}s"
                )
                expired_count += 1

        # Check semantic
        semantic_mems = self.semantic_store.list_memories(tenant_id, limit=500, active_only=True)
        for mem in semantic_mems:
            if (now - mem.updated_at) > max_age_seconds:
                prev = mem.lifecycle_status.value
                mem.lifecycle_status = MemoryLifecycleStatus.EXPIRED
                mem.updated_at = now
                self._record_audit(
                    tenant_id, mem.memory_id, "semantic", prev,
                    MemoryLifecycleStatus.EXPIRED.value, f"Age exceeded {max_age_seconds}s"
                )
                expired_count += 1

        return expired_count

    def delete_memory(
        self,
        tenant_id: str,
        memory_id: str,
        memory_type: str = "semantic",
        reason: str = "Explicit user deletion request",
    ) -> bool:
        """
        Governed deletion: marks as DELETED and records audit entry.
        """
        if memory_type == "semantic":
            mem = self.semantic_store.get_memory(tenant_id, memory_id)
            if not mem:
                return False
            prev = mem.lifecycle_status.value
            mem.lifecycle_status = MemoryLifecycleStatus.DELETED
            mem.updated_at = time.time()
            self._record_audit(tenant_id, memory_id, "semantic", prev, MemoryLifecycleStatus.DELETED.value, reason)
            return True
        elif memory_type == "episodic":
            ep = self.episodic_store.get_episode(tenant_id, memory_id)
            if not ep:
                return False
            prev = ep.lifecycle_status.value
            ep.lifecycle_status = MemoryLifecycleStatus.DELETED
            self._record_audit(tenant_id, memory_id, "episodic", prev, MemoryLifecycleStatus.DELETED.value, reason)
            return True
        return False

    def _record_audit(
        self,
        tenant_id: str,
        memory_id: str,
        memory_type: str,
        prev: str,
        new: str,
        reason: str,
    ) -> None:
        self._audit_log.append(
            LifecycleAuditEntry(
                tenant_id=tenant_id,
                memory_id=memory_id,
                memory_type=memory_type,
                previous_status=prev,
                new_status=new,
                reason=reason,
                timestamp=time.time(),
            )
        )

    def get_audit_trail(self, tenant_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Return audit trail entries, optionally filtered by tenant."""
        if tenant_id:
            return [e.to_dict() for e in self._audit_log if e.tenant_id == tenant_id]
        return [e.to_dict() for e in self._audit_log]
