"""
SQLite-Backed Durable Security State Store for ChakrView (Step 34).

Provides durable, crash-resilient disk storage for snapshots and write-ahead journals.
Uses standard JSON columns with zero pickle serialization.
"""

import json
from pathlib import Path
import sqlite3
import threading
from typing import Dict, List, Optional, Union

from chakrview.cognition.federation.persistence.base import SecurityStateStore
from chakrview.cognition.federation.persistence.errors import PersistenceError
from chakrview.cognition.federation.persistence.models import (
    DurableSecuritySnapshot,
    JournalEntry,
    JournalEntryType,
)


class SqliteSecurityStateStore(SecurityStateStore):
    """
    SQLite-backed transactional persistence for federation snapshots and security journals.
    """

    def __init__(self, db_path: Union[str, Path] = ":memory:") -> None:
        self.db_path = str(db_path)
        self._lock = threading.Lock()
        self._conn: Optional[sqlite3.Connection] = None
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        if self._conn is None:
            conn = sqlite3.connect(
                self.db_path,
                check_same_thread=False,
                timeout=30.0,
            )
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            self._conn = conn
        return self._conn

    def _init_db(self) -> None:
        with self._lock:
            conn = self._get_connection()
            with conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS snapshots (
                        snapshot_id TEXT PRIMARY KEY,
                        snapshot_version INTEGER NOT NULL,
                        journal_offset INTEGER NOT NULL,
                        created_at REAL NOT NULL,
                        payload TEXT NOT NULL,
                        integrity_hash TEXT NOT NULL
                    );
                    """
                )
                conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_snapshots_version
                    ON snapshots(snapshot_version ASC);
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS journal (
                        sequence_num INTEGER PRIMARY KEY,
                        epoch INTEGER NOT NULL,
                        entry_type TEXT NOT NULL,
                        event_id TEXT NOT NULL,
                        payload TEXT NOT NULL,
                        prev_digest TEXT NOT NULL,
                        digest TEXT NOT NULL,
                        timestamp REAL NOT NULL
                    );
                    """
                )

    def save_snapshot(self, snapshot: DurableSecuritySnapshot) -> None:
        if not snapshot.integrity_hash:
            snapshot.seal()

        snap_json = json.dumps(snapshot.to_dict(), sort_keys=True, separators=(",", ":"))
        with self._lock:
            conn = self._get_connection()
            with conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO snapshots (
                        snapshot_id, snapshot_version, journal_offset,
                        created_at, payload, integrity_hash
                    ) VALUES (?, ?, ?, ?, ?, ?);
                    """,
                    (
                        snapshot.snapshot_id,
                        snapshot.snapshot_version,
                        snapshot.journal_offset,
                        snapshot.created_at,
                        snap_json,
                        snapshot.integrity_hash,
                    ),
                )

    def load_latest_snapshot(self) -> Optional[DurableSecuritySnapshot]:
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT payload FROM snapshots
                ORDER BY snapshot_version DESC, created_at DESC
                LIMIT 1;
                """
            )
            row = cursor.fetchone()
            if not row:
                return None
            data = json.loads(row[0])
            return DurableSecuritySnapshot.from_dict(data)

    def load_snapshot(self, snapshot_id: str) -> Optional[DurableSecuritySnapshot]:
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(
                "SELECT payload FROM snapshots WHERE snapshot_id = ? LIMIT 1;",
                (snapshot_id,),
            )
            row = cursor.fetchone()
            if not row:
                return None
            data = json.loads(row[0])
            return DurableSecuritySnapshot.from_dict(data)

    def list_snapshots(self) -> List[str]:
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT snapshot_id FROM snapshots ORDER BY snapshot_version ASC;")
            return [r[0] for r in cursor.fetchall()]

    def append_journal_entry(self, entry: JournalEntry) -> None:
        payload_json = json.dumps(entry.payload, sort_keys=True, separators=(",", ":"))
        with self._lock:
            conn = self._get_connection()
            with conn:
                conn.execute(
                    """
                    INSERT INTO journal (
                        sequence_num, epoch, entry_type, event_id,
                        payload, prev_digest, digest, timestamp
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?);
                    """,
                    (
                        entry.sequence_num,
                        entry.epoch,
                        entry.entry_type.value,
                        entry.event_id,
                        payload_json,
                        entry.prev_digest,
                        entry.digest,
                        entry.timestamp,
                    ),
                )

    def read_journal_entries(
        self,
        since_sequence: int = 0,
        limit: Optional[int] = None,
    ) -> List[JournalEntry]:
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            query = """
                SELECT sequence_num, epoch, entry_type, event_id,
                       payload, prev_digest, digest, timestamp
                FROM journal
                WHERE sequence_num > ?
                ORDER BY sequence_num ASC
            """
            params: List[Any] = [since_sequence]
            if limit is not None:
                query += " LIMIT ?"
                params.append(limit)

            cursor.execute(query, tuple(params))
            results = []
            for row in cursor.fetchall():
                entry_dict = {
                    "sequence_num": row[0],
                    "epoch": row[1],
                    "entry_type": row[2],
                    "event_id": row[3],
                    "payload": json.loads(row[4]),
                    "prev_digest": row[5],
                    "digest": row[6],
                    "timestamp": row[7],
                }
                results.append(JournalEntry.from_dict(entry_dict))
            return results

    def get_last_journal_sequence(self) -> int:
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT MAX(sequence_num) FROM journal;")
            row = cursor.fetchone()
            if not row or row[0] is None:
                return 0
            return int(row[0])

    def truncate_journal(self, before_sequence: int) -> int:
        with self._lock:
            conn = self._get_connection()
            with conn:
                cursor = conn.cursor()
                cursor.execute(
                    "DELETE FROM journal WHERE sequence_num < ?;",
                    (before_sequence,),
                )
                return cursor.rowcount

    def close(self) -> None:
        with self._lock:
            if self._conn:
                self._conn.close()
                self._conn = None
