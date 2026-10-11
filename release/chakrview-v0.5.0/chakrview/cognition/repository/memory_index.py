from __future__ import annotations

import json
import time
import threading
from typing import Dict, List, Optional, Any
from .semantic_record import RepositorySemanticRecord


class RepositoryMemoryIndex:
    """Deterministic in‑memory index for repository semantic memories.

    • Supports insertion, deletion, versioning, and lookup by ID.
    • Provides filtered candidate retrieval based on explicit fields.
    • All operations are thread‑safe and produce reproducible ordering
      (sorted by ``memory_id``) to guarantee deterministic behaviour.
    """

    def __init__(self) -> None:
        self._store: Dict[str, RepositorySemanticRecord] = {}
        self._lock = threading.RLock()
        self._version_counter: int = 1

    # ---------------------------------------------------------------------
    # Core CRUD operations
    # ---------------------------------------------------------------------
    def insert(self, record: RepositorySemanticRecord) -> None:
        """Insert a new ``RepositorySemanticRecord``.

        If a record with the same ``memory_id`` already exists, it is
        treated as a versioned update – the existing record is marked as
        ``active_version=False`` and a new version is inserted with an
        incremented ``version`` field.
        """
        with self._lock:
            if record.memory_id in self._store:
                # versioning path
                old = self._store[record.memory_id]
                old.active_version = False
                old.superseded_by = f"{record.memory_id}_v{self._version_counter}"
                record.version = self._version_counter
                record.supersedes = old.memory_id
                self._version_counter += 1
            self._store[record.memory_id] = record

    def delete(self, memory_id: str) -> bool:
        """Delete a record by ``memory_id``. Returns ``True`` if removed."""
        with self._lock:
            return self._store.pop(memory_id, None) is not None

    def lookup(self, memory_id: str) -> Optional[RepositorySemanticRecord]:
        """Retrieve a record by ID (or ``None``)."""
        with self._lock:
            return self._store.get(memory_id)

    # ---------------------------------------------------------------------
    # Search / Retrieval helpers
    # ---------------------------------------------------------------------
    def _match_field(self, record: RepositorySemanticRecord, field: str, value: Any) -> bool:
        """Exact match helper used by ``candidate_set``.
        ``value`` may be ``None`` meaning "do not filter on this field".
        """
        if value is None:
            return True
        record_value = getattr(record, field)
        # For list fields we require any overlap
        if isinstance(record_value, list):
            return any(item == value or (isinstance(item, str) and isinstance(value, str) and value in item) for item in record_value)
        return record_value == value

    def candidate_set(
        self,
        *,
        task_family: Optional[str] = None,
        language: Optional[str] = None,
        framework: Optional[str] = None,
        symptom_signature: Optional[str] = None,
        root_cause_signature: Optional[str] = None,
        dependency_signature: Optional[str] = None,
        affected_modules: Optional[List[str]] = None,
    ) -> List[RepositorySemanticRecord]:
        """Return a *deterministic* list of candidate memories matching the supplied
        filters. The list is sorted by ``memory_id`` to guarantee reproducibility.
        """
        with self._lock:
            candidates: List[RepositorySemanticRecord] = []
            for rec in self._store.values():
                if not self._match_field(rec, "task_family", task_family):
                    continue
                if not self._match_field(rec, "language", language):
                    continue
                if not self._match_field(rec, "framework", framework):
                    continue
                if not self._match_field(rec, "symptom_signature", symptom_signature):
                    continue
                if not self._match_field(rec, "root_cause_signature", root_cause_signature):
                    continue
                if not self._match_field(rec, "dependency_signature", dependency_signature):
                    continue
                if affected_modules is not None:
                    if not any(mod in rec.affected_modules for mod in affected_modules):
                        continue
                candidates.append(rec)
            candidates.sort(key=lambda r: r.memory_id)
            return candidates

    # ---------------------------------------------------------------------
    # Serialisation
    # ---------------------------------------------------------------------
    def to_json(self) -> str:
        """Serialise the entire index to a JSON string (ordered for determinism)."""
        with self._lock:
            ordered = {mid: rec.to_dict() for mid, rec in sorted(self._store.items())}
            return json.dumps(ordered, sort_keys=True, indent=2)

    @classmethod
    def from_json(cls, data: str) -> "RepositoryMemoryIndex":
        """Deserialize an index produced by ``to_json``.

        Returns a new ``RepositoryMemoryIndex`` instance with the same internal
        ordering and version counter.
        """
        obj = cls()
        raw = json.loads(data)
        for mid, rec_dict in raw.items():
            rec = RepositorySemanticRecord.from_dict(rec_dict)
            obj._store[mid] = rec
        max_version = max((rec.version for rec in obj._store.values()), default=0)
        obj._version_counter = max_version + 1
        return obj

    # ---------------------------------------------------------------------
    # Convenience utilities for experiments
    # ---------------------------------------------------------------------
    def count(self) -> int:
        """Return the total number of stored memories (including superseded versions)."""
        with self._lock:
            return len(self._store)

    def all_records(self) -> Dict[str, RepositorySemanticRecord]:
        """Return a copy of all stored records."""
        with self._lock:
            return dict(self._store)


