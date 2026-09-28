"""
Episodic Memory Subsystem for ChakrView Continual Cognition (Step 24).

Represents structured experiences and events rather than treating raw logs
as permanent immutable truths.

CRITICAL ARCHITECTURAL CONSTRAINTS:
1. An Episode preserves what actually happened (situation, action, outcome)
   distinctly separated from semantic interpretations.
2. An episode is NEVER silently converted into a verified fact.
3. Strict tenant isolation: Tenant A can NEVER inspect or query Tenant B episodes.
"""

from dataclasses import asdict
import time
from typing import Dict, List, Optional, Any, Callable
import uuid

from chakrview.memory.models import (
    Episode,
    MemoryProvenanceSource,
    MemoryVerificationState,
    MemoryLifecycleStatus,
)


class EpisodicMemoryStore:
    """
    In-memory episodic store indexed strictly by tenant and session.
    """

    def __init__(self) -> None:
        # _store[tenant_id][episode_id] = Episode
        self._store: Dict[str, Dict[str, Episode]] = {}

    def record_episode(
        self,
        tenant_id: str,
        session_id: str,
        situation: str,
        action_or_response: str,
        outcome: str,
        task_id: Optional[str] = None,
        evidence_refs: Optional[List[str]] = None,
        confidence: float = 0.8,
        provenance: MemoryProvenanceSource = MemoryProvenanceSource.SYSTEM_OBSERVED,
        verification_status: MemoryVerificationState = MemoryVerificationState.UNVERIFIED,
        lifecycle_status: MemoryLifecycleStatus = MemoryLifecycleStatus.ACTIVE,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Episode:
        """
        Record a newly experienced episode.
        Enforces tenant isolation and validates confidence bounds.
        """
        if not tenant_id:
            raise ValueError("tenant_id cannot be empty.")
        if not session_id:
            raise ValueError("session_id cannot be empty.")

        episode_id = f"ep_{uuid.uuid4().hex[:12]}"
        episode = Episode(
            episode_id=episode_id,
            tenant_id=tenant_id,
            session_id=session_id,
            situation=situation,
            action_or_response=action_or_response,
            outcome=outcome,
            task_id=task_id,
            evidence_refs=evidence_refs or [],
            confidence=max(0.0, min(1.0, confidence)),
            provenance=provenance,
            verification_status=verification_status,
            lifecycle_status=lifecycle_status,
            tags=tags or [],
            metadata=metadata or {},
        )

        if tenant_id not in self._store:
            self._store[tenant_id] = {}
        self._store[tenant_id][episode_id] = episode
        return episode

    def get_episode(self, tenant_id: str, episode_id: str) -> Optional[Episode]:
        """Fetch episode enforcing tenant ownership."""
        tenant_episodes = self._store.get(tenant_id, {})
        return tenant_episodes.get(episode_id)

    def list_episodes(
        self,
        tenant_id: str,
        session_id: Optional[str] = None,
        limit: int = 50,
        lifecycle_filter: Optional[MemoryLifecycleStatus] = MemoryLifecycleStatus.ACTIVE,
    ) -> List[Episode]:
        """List episodes for a given tenant, optionally filtered by session and status."""
        tenant_episodes = self._store.get(tenant_id, {})
        results: List[Episode] = []

        # Sort reverse chronologically
        for ep in sorted(tenant_episodes.values(), key=lambda e: e.timestamp, reverse=True):
            if session_id and ep.session_id != session_id:
                continue
            if lifecycle_filter and ep.lifecycle_status != lifecycle_filter:
                continue
            results.append(ep)
            if len(results) >= limit:
                break
        return results

    def update_verification(
        self,
        tenant_id: str,
        episode_id: str,
        state: MemoryVerificationState,
    ) -> bool:
        """Update verification state of an existing episode."""
        ep = self.get_episode(tenant_id, episode_id)
        if not ep:
            return False
        ep.verification_status = state
        return True

    def mark_lifecycle(
        self,
        tenant_id: str,
        episode_id: str,
        status: MemoryLifecycleStatus,
    ) -> bool:
        """Transition lifecycle status (e.g. ARCHIVED, QUARANTINED)."""
        ep = self.get_episode(tenant_id, episode_id)
        if not ep:
            return False
        ep.lifecycle_status = status
        return True

    def count(self, tenant_id: str) -> int:
        """Return total stored episodes for a given tenant."""
        return len(self._store.get(tenant_id, {}))

    def clear(self, tenant_id: Optional[str] = None) -> None:
        """Clear episodes for a specific tenant or all stores."""
        if tenant_id:
            if tenant_id in self._store:
                self._store[tenant_id].clear()
        else:
            self._store.clear()

    def to_dict(self) -> Dict[str, Any]:
        """Export serialized snapshot across all tenants."""
        out: Dict[str, Any] = {}
        for t_id, ep_dict in self._store.items():
            out[t_id] = {ep_id: ep.to_dict() for ep_id, ep in ep_dict.items()}
        return out

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EpisodicMemoryStore":
        """Reconstruct store from serialized snapshot."""
        store = cls()
        for t_id, ep_dict in data.items():
            store._store[t_id] = {}
            for ep_id, ep_data in ep_dict.items():
                store._store[t_id][ep_id] = Episode.from_dict(ep_data)
        return store
