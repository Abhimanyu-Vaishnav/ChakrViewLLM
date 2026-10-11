"""
Bounded Audit Logger for Cross-Zone Peering Operations (Step 29).

CRITICAL ARCHITECTURAL AXIOM:
BOUNDED & SANITIZED TELEMETRY:
Audit logs are capped at MAX_AUDIT_LOG_ENTRIES (1000) using strict FIFO eviction.
Private chain-of-thought, hidden states, raw activations, and cryptographic secrets
are strictly excluded from audit entries.
"""

import collections
import threading
from typing import Dict, List, Optional, Any

from chakrview.cognition.peering.models import (
    AuditRecord,
    AuditEventType,
    MAX_AUDIT_LOG_ENTRIES,
)


class BoundedAuditLogger:
    """
    Thread-safe, bounded-memory audit logger for peering transactions.
    """

    def __init__(self, max_entries: int = MAX_AUDIT_LOG_ENTRIES) -> None:
        self.max_entries = min(max_entries, MAX_AUDIT_LOG_ENTRIES)
        self._lock = threading.Lock()
        self._records: collections.deque[AuditRecord] = collections.deque(maxlen=self.max_entries)

    def log(
        self,
        event_type: AuditEventType,
        epoch: int,
        peer_id: Optional[str] = None,
        zone_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        session_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> AuditRecord:
        """Record an auditable peering event."""
        event_id = f"aud_{event_type.value.lower()}_{len(self._records)}_{epoch}"
        record = AuditRecord(
            event_id=event_id,
            event_type=event_type,
            epoch=epoch,
            peer_id=peer_id,
            zone_id=zone_id,
            tenant_id=tenant_id,
            session_id=session_id,
            details=dict(details or {}),
        )

        with self._lock:
            self._records.append(record)

        return record

    def get_recent(self, limit: int = 100) -> List[AuditRecord]:
        with self._lock:
            records = list(self._records)
            return records[-limit:]

    def get_by_peer(self, peer_id: str) -> List[AuditRecord]:
        with self._lock:
            return [r for r in self._records if r.peer_id == peer_id]

    def get_by_event_type(self, event_type: AuditEventType) -> List[AuditRecord]:
        with self._lock:
            return [r for r in self._records if r.event_type == event_type]

    def clear(self) -> None:
        with self._lock:
            self._records.clear()

    @property
    def entry_count(self) -> int:
        with self._lock:
            return len(self._records)
