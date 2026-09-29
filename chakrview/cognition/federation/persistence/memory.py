"""
In-Memory Security State Store for Testing and Transient Nodes (Step 34).

Provides an isolated, thread-safe in-memory implementation of SecurityStateStore.
Copies on write/read to guarantee that mutations do not leak into stored representations.
"""

import threading
from typing import Dict, List, Optional

from chakrview.cognition.federation.persistence.base import SecurityStateStore
from chakrview.cognition.federation.persistence.models import (
    DurableSecuritySnapshot,
    JournalEntry,
)


class InMemorySecurityStateStore(SecurityStateStore):
    """
    In-memory, thread-safe implementation of SecurityStateStore.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._snapshots: Dict[str, Dict] = {}  # snapshot_id -> dict
        self._snapshot_order: List[str] = []
        self._journal: List[Dict] = []         # list of dicts

    def save_snapshot(self, snapshot: DurableSecuritySnapshot) -> None:
        if not snapshot.integrity_hash:
            snapshot.seal()

        with self._lock:
            # Store isolated dict copy
            snap_dict = snapshot.to_dict()
            self._snapshots[snapshot.snapshot_id] = snap_dict
            if snapshot.snapshot_id not in self._snapshot_order:
                self._snapshot_order.append(snapshot.snapshot_id)

    def load_latest_snapshot(self) -> Optional[DurableSecuritySnapshot]:
        with self._lock:
            if not self._snapshot_order:
                return None
            latest_id = self._snapshot_order[-1]
            return DurableSecuritySnapshot.from_dict(self._snapshots[latest_id])

    def load_snapshot(self, snapshot_id: str) -> Optional[DurableSecuritySnapshot]:
        with self._lock:
            if snapshot_id not in self._snapshots:
                return None
            return DurableSecuritySnapshot.from_dict(self._snapshots[snapshot_id])

    def list_snapshots(self) -> List[str]:
        with self._lock:
            return list(self._snapshot_order)

    def append_journal_entry(self, entry: JournalEntry) -> None:
        with self._lock:
            self._journal.append(entry.to_dict())

    def read_journal_entries(
        self,
        since_sequence: int = 0,
        limit: Optional[int] = None,
    ) -> List[JournalEntry]:
        with self._lock:
            results = []
            for d in self._journal:
                if d["sequence_num"] > since_sequence:
                    results.append(JournalEntry.from_dict(d))
                    if limit is not None and len(results) >= limit:
                        break
            return results

    def get_last_journal_sequence(self) -> int:
        with self._lock:
            if not self._journal:
                return 0
            return self._journal[-1]["sequence_num"]

    def truncate_journal(self, before_sequence: int) -> int:
        with self._lock:
            orig_len = len(self._journal)
            self._journal = [d for d in self._journal if d["sequence_num"] >= before_sequence]
            return orig_len - len(self._journal)

    def close(self) -> None:
        pass
