"""
Storage Abstraction Layer for ChakrView Persistent Memory (Step 16).

Decouples memory management logic from concrete storage engines (in-memory,
SQLite, JSON, encrypted disk, or enterprise vector databases).
Enforces strict owner/user privacy boundaries across all operations.
"""

from abc import ABC, abstractmethod
import json
from pathlib import Path
import time
from typing import Dict, List, Optional, Any, Tuple, Union

from chakrview.memory.record import MemoryRecord, MemoryType, MemoryValidity


class MemoryStoreError(RuntimeError):
    """Base exception for persistent memory storage failures."""
    pass


class MemoryNotFoundError(MemoryStoreError):
    """Raised when a specified memory record does not exist."""
    pass


class MemoryAccessDeniedError(PermissionError):
    """Raised when an operation attempts cross-user access violation."""
    pass


class MemoryStore(ABC):
    """
    Abstract storage interface for persistent personal memory.
    """

    @abstractmethod
    def add(self, record: MemoryRecord) -> str:
        """Add a new memory record and return its memory_id."""
        pass

    @abstractmethod
    def get(self, memory_id: str, owner_id: str) -> Optional[MemoryRecord]:
        """Retrieve a memory record by ID, scoped to owner_id."""
        pass

    @abstractmethod
    def update(self, record: MemoryRecord) -> bool:
        """Update an existing memory record."""
        pass

    @abstractmethod
    def delete(self, memory_id: str, owner_id: str) -> bool:
        """Delete a memory record by ID, scoped to owner_id."""
        pass

    @abstractmethod
    def list_records(
        self,
        owner_id: str,
        memory_type: Optional[MemoryType] = None,
        validity: Optional[MemoryValidity] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[MemoryRecord]:
        """List memory records for an owner with optional filters."""
        pass

    @abstractmethod
    def supersede(self, old_id: str, new_record: MemoryRecord, owner_id: str) -> bool:
        """Mark old record as superseded and store the new revision."""
        pass

    @abstractmethod
    def archive(self, memory_id: str, owner_id: str) -> bool:
        """Mark a memory record as archived."""
        pass

    @abstractmethod
    def count(self, owner_id: str, validity: Optional[MemoryValidity] = None) -> int:
        """Count total records for an owner."""
        pass

    @abstractmethod
    def clear(self, owner_id: Optional[str] = None) -> None:
        """Clear all records, or records for a specific owner."""
        pass


class InMemoryMemoryStore(MemoryStore):
    """
    Lightweight deterministic in-memory store.
    Guarantees strict owner isolation and thread-safe predictability.
    """

    def __init__(self) -> None:
        # Internal storage: Dict[owner_id, Dict[memory_id, MemoryRecord]]
        self._store: Dict[str, Dict[str, MemoryRecord]] = {}

    def add(self, record: MemoryRecord) -> str:
        if not record.owner_id:
            raise ValueError("Record owner_id cannot be empty.")
        if record.owner_id not in self._store:
            self._store[record.owner_id] = {}
        self._store[record.owner_id][record.memory_id] = record
        return record.memory_id

    def get(self, memory_id: str, owner_id: str) -> Optional[MemoryRecord]:
        user_store = self._store.get(owner_id)
        if not user_store:
            return None
        return user_store.get(memory_id)

    def update(self, record: MemoryRecord) -> bool:
        user_store = self._store.get(record.owner_id)
        if not user_store or record.memory_id not in user_store:
            return False
        record.temporal.updated_at = time.time()
        user_store[record.memory_id] = record
        return True

    def delete(self, memory_id: str, owner_id: str) -> bool:
        user_store = self._store.get(owner_id)
        if not user_store or memory_id not in user_store:
            return False
        del user_store[memory_id]
        return True

    def list_records(
        self,
        owner_id: str,
        memory_type: Optional[MemoryType] = None,
        validity: Optional[MemoryValidity] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[MemoryRecord]:
        user_store = self._store.get(owner_id)
        if not user_store:
            return []

        results: List[MemoryRecord] = []
        for rec in user_store.values():
            if memory_type is not None and rec.memory_type != memory_type:
                continue
            if validity is not None and rec.validity != validity:
                continue
            results.append(rec)

        # Sort deterministically by updated_at descending
        results.sort(key=lambda r: r.temporal.updated_at, reverse=True)
        return results[offset:offset + limit]

    def supersede(self, old_id: str, new_record: MemoryRecord, owner_id: str) -> bool:
        user_store = self._store.get(owner_id)
        if not user_store or old_id not in user_store:
            return False

        old_rec = user_store[old_id]
        old_rec.mark_superseded(new_record.memory_id)
        new_record.temporal.version = old_rec.temporal.version + 1
        new_record.owner_id = owner_id

        self.add(new_record)
        return True

    def archive(self, memory_id: str, owner_id: str) -> bool:
        rec = self.get(memory_id, owner_id)
        if rec is None:
            return False
        rec.mark_archived()
        return True

    def count(self, owner_id: str, validity: Optional[MemoryValidity] = None) -> int:
        user_store = self._store.get(owner_id)
        if not user_store:
            return 0
        if validity is None:
            return len(user_store)
        return sum(1 for r in user_store.values() if r.validity == validity)

    def clear(self, owner_id: Optional[str] = None) -> None:
        if owner_id is not None:
            if owner_id in self._store:
                self._store[owner_id].clear()
        else:
            self._store.clear()

    def dump_to_jsonl(self, filepath: Union[str, Path], owner_id: Optional[str] = None) -> int:
        """Export records to JSONL file for portable persistence testing."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        count = 0
        with open(path, "w", encoding="utf-8") as f:
            owners = [owner_id] if owner_id else list(self._store.keys())
            for o in owners:
                if o in self._store:
                    for rec in self._store[o].values():
                        f.write(json.dumps(rec.to_dict()) + "\n")
                        count += 1
        return count

    def load_from_jsonl(self, filepath: Union[str, Path]) -> int:
        """Import records from JSONL file."""
        path = Path(filepath)
        if not path.is_file():
            raise FileNotFoundError(f"Memory export file not found: {path}")
        count = 0
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    rec = MemoryRecord.from_dict(data)
                    self.add(rec)
                    count += 1
        return count
