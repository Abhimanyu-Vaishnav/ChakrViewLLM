"""
Temporal Memory & Revision History Management for ChakrView (Step 16).

Tracks validity windows, revision sequencing, supersession chains, and
historical point-in-time state queries.
"""

from dataclasses import dataclass, field
import time
from typing import Dict, List, Optional, Any

from chakrview.memory.record import MemoryRecord, MemoryValidity
from chakrview.memory.store import MemoryStore


@dataclass
class TemporalLineageNode:
    """A single node in a memory revision history."""
    memory_id: str
    version: int
    content: str
    validity: MemoryValidity
    created_at: float
    updated_at: float
    superseded_by: Optional[str] = None


class TemporalMemoryManager:
    """
    Manages temporal state filtering, expiration transitions, and revision lineage.
    """

    @staticmethod
    def get_active_memories(
        records: List[MemoryRecord],
        as_of_timestamp: Optional[float] = None,
    ) -> List[MemoryRecord]:
        """
        Return records that are temporally valid at a given timestamp.
        """
        ts = time.time() if as_of_timestamp is None else as_of_timestamp
        return [r for r in records if r.is_valid_at(ts)]

    @staticmethod
    def sweep_expired_records(
        records: List[MemoryRecord],
        store: Optional[MemoryStore] = None,
        current_time: Optional[float] = None,
    ) -> List[str]:
        """
        Identify active records whose valid_until timestamp has elapsed,
        transition them to EXPIRED, and update store if provided.
        """
        now = time.time() if current_time is None else current_time
        expired_ids: List[str] = []

        for rec in records:
            if rec.validity == MemoryValidity.ACTIVE and rec.temporal.valid_until is not None:
                if now > rec.temporal.valid_until:
                    rec.mark_expired()
                    expired_ids.append(rec.memory_id)
                    if store is not None:
                        store.update(rec)

        return expired_ids

    @staticmethod
    def get_revision_history(
        memory_id: str,
        owner_id: str,
        store: MemoryStore,
    ) -> List[TemporalLineageNode]:
        """
        Traverse forward supersession links starting from a base memory to build
        the full evolutionary revision lineage.
        """
        lineage: List[TemporalLineageNode] = []
        curr_id: Optional[str] = memory_id
        visited = set()

        while curr_id and curr_id not in visited:
            visited.add(curr_id)
            rec = store.get(curr_id, owner_id)
            if not rec:
                break

            lineage.append(TemporalLineageNode(
                memory_id=rec.memory_id,
                version=rec.temporal.version,
                content=rec.content,
                validity=rec.validity,
                created_at=rec.temporal.created_at,
                updated_at=rec.temporal.updated_at,
                superseded_by=rec.temporal.superseded_by,
            ))
            curr_id = rec.temporal.superseded_by

        return lineage
