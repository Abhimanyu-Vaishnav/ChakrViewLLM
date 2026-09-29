"""
Append-Only Security State Journal for ChakrView (Step 34).

Provides cryptographic hash-chaining, monotonic sequence enforcement,
tamper detection, and replay verification for all security-critical mutations.
"""

import json
import threading
from typing import Dict, List, Optional, Any, Tuple

from chakrview.cognition.federation.persistence.errors import (
    JournalCorruptionError,
    JournalSequenceError,
    JournalTruncationError,
)
from chakrview.cognition.federation.persistence.models import (
    JournalEntry,
    JournalEntryType,
    JOURNAL_GENESIS_DIGEST,
    MAX_JOURNAL_PAYLOAD_BYTES,
)


class SecurityStateJournal:
    """
    Append-only, cryptographically chained write-ahead security journal.
    """

    def __init__(self, initial_entries: Optional[List[JournalEntry]] = None) -> None:
        self._lock = threading.Lock()
        self._entries: List[JournalEntry] = []
        self._last_digest: str = JOURNAL_GENESIS_DIGEST
        self._last_sequence: int = 0

        if initial_entries:
            self.load_entries(initial_entries)

    @property
    def last_sequence(self) -> int:
        with self._lock:
            return self._last_sequence

    @property
    def last_digest(self) -> str:
        with self._lock:
            return self._last_digest

    @property
    def entry_count(self) -> int:
        with self._lock:
            return len(self._entries)

    def append(
        self,
        entry_type: JournalEntryType,
        epoch: int,
        payload: Dict[str, Any],
    ) -> JournalEntry:
        """
        Create and append a new cryptographically chained journal entry.
        """
        # Validate size bounds
        payload_bytes = len(json.dumps(payload).encode("utf-8"))
        if payload_bytes > MAX_JOURNAL_PAYLOAD_BYTES:
            raise ValueError(
                f"Journal entry payload ({payload_bytes} bytes) exceeds ceiling "
                f"({MAX_JOURNAL_PAYLOAD_BYTES} bytes)."
            )

        with self._lock:
            next_seq = self._last_sequence + 1
            prev_dig = self._last_digest

            entry = JournalEntry.create(
                sequence_num=next_seq,
                epoch=epoch,
                entry_type=entry_type,
                payload=payload,
                prev_digest=prev_dig,
            )

            self._entries.append(entry)
            self._last_sequence = next_seq
            self._last_digest = entry.digest
            return entry

    def get_entries(
        self,
        since_sequence: int = 0,
        limit: Optional[int] = None,
    ) -> List[JournalEntry]:
        """Retrieve journal entries after a specified sequence number."""
        with self._lock:
            filtered = [e for e in self._entries if e.sequence_num > since_sequence]
            if limit is not None:
                return filtered[:limit]
            return list(filtered)

    def load_entries(self, entries: List[JournalEntry]) -> None:
        """
        Bulk load and verify an existing series of journal entries.
        Fails closed on any sequence gap, broken hash chain, or corruption.
        """
        with self._lock:
            self.verify_chain(entries)
            self._entries = list(entries)
            if entries:
                self._last_sequence = entries[-1].sequence_num
                self._last_digest = entries[-1].digest
            else:
                self._last_sequence = 0
                self._last_digest = JOURNAL_GENESIS_DIGEST

    @staticmethod
    def verify_chain(entries: List[JournalEntry]) -> Tuple[bool, Optional[str]]:
        """
        Verify cryptographic hash-chaining, monotonic sequence numbers,
        and integrity across a list of journal entries.
        """
        if not entries:
            return True, None

        expected_prev = JOURNAL_GENESIS_DIGEST
        expected_seq = entries[0].sequence_num

        for idx, entry in enumerate(entries):
            # 1. Sequence number check
            if entry.sequence_num != expected_seq:
                raise JournalSequenceError(
                    f"Journal sequence regression/gap at index {idx}: expected {expected_seq}, "
                    f"got {entry.sequence_num}."
                )

            # 2. Previous digest continuity check (for entries starting after genesis)
            if idx == 0 and entry.sequence_num == 1:
                if entry.prev_digest != JOURNAL_GENESIS_DIGEST:
                    raise JournalCorruptionError(
                        f"First journal entry must chain from genesis digest, got: {entry.prev_digest[:16]}..."
                    )
            elif idx > 0:
                if entry.prev_digest != expected_prev:
                    raise JournalCorruptionError(
                        f"Hash chain broken at sequence {entry.sequence_num}: expected prev {expected_prev[:16]}..., "
                        f"got {entry.prev_digest[:16]}..."
                    )

            # 3. Payload integrity check
            if not entry.verify_integrity(expected_prev_digest=entry.prev_digest):
                raise JournalCorruptionError(
                    f"Cryptographic digest mismatch on entry {entry.sequence_num} ({entry.event_id}). "
                    f"Payload tampering or bit corruption detected."
                )

            expected_prev = entry.digest
            expected_seq += 1

        return True, None

    def truncate_before(self, sequence_num: int) -> int:
        """
        Safely prune entries strictly before sequence_num (e.g. after snapshot creation).
        Returns number of removed entries.
        """
        with self._lock:
            original_len = len(self._entries)
            self._entries = [e for e in self._entries if e.sequence_num >= sequence_num]
            return original_len - len(self._entries)
