"""
Abstract Base Interface for Durable Security State Stores (Step 34).

Defines the contract for snapshot storage and append-only write-ahead journals.
"""

from abc import ABC, abstractmethod
from typing import List, Optional

from chakrview.cognition.federation.persistence.models import (
    DurableSecuritySnapshot,
    JournalEntry,
)


class SecurityStateStore(ABC):
    """
    Abstract storage backend for snapshots and write-ahead security journals.
    """

    @abstractmethod
    def save_snapshot(self, snapshot: DurableSecuritySnapshot) -> None:
        """Atomically persist a durable security snapshot."""
        pass

    @abstractmethod
    def load_latest_snapshot(self) -> Optional[DurableSecuritySnapshot]:
        """Retrieve the most recent verified snapshot, or None if empty."""
        pass

    @abstractmethod
    def load_snapshot(self, snapshot_id: str) -> Optional[DurableSecuritySnapshot]:
        """Retrieve a specific snapshot by ID."""
        pass

    @abstractmethod
    def list_snapshots(self) -> List[str]:
        """List all available snapshot IDs ordered by version ascending."""
        pass

    @abstractmethod
    def append_journal_entry(self, entry: JournalEntry) -> None:
        """Atomically append a validated journal entry."""
        pass

    @abstractmethod
    def read_journal_entries(
        self,
        since_sequence: int = 0,
        limit: Optional[int] = None,
    ) -> List[JournalEntry]:
        """Read journal entries with sequence strictly greater than since_sequence."""
        pass

    @abstractmethod
    def get_last_journal_sequence(self) -> int:
        """Return the highest recorded journal sequence number, or 0 if empty."""
        pass

    @abstractmethod
    def truncate_journal(self, before_sequence: int) -> int:
        """Prune journal entries strictly before sequence number."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Cleanly flush and close the storage backend."""
        pass
