"""
ChakrView Step 123: Distributed Cognitive State Synchronization.

Enables multiple cognitive nodes to share targeted state without broadcasting
the entire repository or project brain:
- VersionedCognitiveStateRecord: state_key, revision, author_node_id, state_category, payload
- DistributedStateSynchronizer: Durable SQLite state sync with optimistic concurrency and conflict detection
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class CognitiveStateCategory(str, Enum):
    TASK_STATE = "TASK_STATE"
    PROJECT_UNDERSTANDING = "PROJECT_UNDERSTANDING"
    WORKER_RESULT = "WORKER_RESULT"
    REVIEWER_CRITIQUE = "REVIEWER_CRITIQUE"
    FAILURE_STATE = "FAILURE_STATE"
    STRATEGY_STATE = "STRATEGY_STATE"


@dataclass
class VersionedCognitiveStateRecord:
    state_key: str
    revision: int
    author_node_id: str
    category: CognitiveStateCategory
    payload: Dict[str, Any]
    payload_hash: str
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["category"] = self.category.value
        return d


class DistributedStateSynchronizer:
    """
    Manages deterministic synchronization of partitioned cognitive state
    across distributed nodes using revision tracking.
    """

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self._init_sync_tables()

    def _init_sync_tables(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS distributed_cognitive_state (
                    state_key TEXT PRIMARY KEY,
                    revision INTEGER,
                    author_node_id TEXT,
                    category TEXT,
                    payload_json TEXT,
                    payload_hash TEXT,
                    timestamp REAL
                )
            """)
            conn.commit()

    def publish_state_update(
        self,
        record: VersionedCognitiveStateRecord,
    ) -> bool:
        """
        Applies update if revision > current revision. Rejects stale writes.
        Returns True if updated, False if stale write rejected.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT revision FROM distributed_cognitive_state WHERE state_key = ?", (record.state_key,))
            row = cursor.fetchone()
            if row:
                current_rev = row[0]
                if record.revision <= current_rev:
                    return False  # Reject stale write

            conn.execute("""
                INSERT OR REPLACE INTO distributed_cognitive_state
                (state_key, revision, author_node_id, category, payload_json, payload_hash, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                record.state_key,
                record.revision,
                record.author_node_id,
                record.category.value,
                json.dumps(record.payload),
                record.payload_hash,
                record.timestamp,
            ))
            conn.commit()
            return True

    def get_state(self, state_key: str) -> Optional[VersionedCognitiveStateRecord]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT revision, author_node_id, category, payload_json, payload_hash, timestamp
                FROM distributed_cognitive_state WHERE state_key = ?
            """, (state_key,))
            row = cursor.fetchone()
            if not row:
                return None
            return VersionedCognitiveStateRecord(
                state_key=state_key,
                revision=row[0],
                author_node_id=row[1],
                category=CognitiveStateCategory(row[2]),
                payload=json.loads(row[3]),
                payload_hash=row[4],
                timestamp=row[5],
            )
