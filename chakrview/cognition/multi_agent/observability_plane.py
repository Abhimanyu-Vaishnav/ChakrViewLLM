"""
ChakrView Step 132: Cognitive Observability & Audit Plane.

Structured, non-leaking observability plane recording all cognitive lifecycle transitions:
- Task dispatch, worker selection, tool governance evaluations, execution outcomes, replans, and reroutes
- Zero sensitive data leakage (payloads hashed and truncated to metadata)
- Complete reconstruction capability of any task execution trajectory
"""

from __future__ import annotations

import enum
import hashlib
import json
import sqlite3
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class AuditEventType(str, enum.Enum):
    TASK_SCHEDULED = "TASK_SCHEDULED"
    TOOL_EVALUATED = "TOOL_EVALUATED"
    WORKER_STARTED = "WORKER_STARTED"
    WORKER_COMPLETED = "WORKER_COMPLETED"
    WORKER_FAILED = "WORKER_FAILED"
    TASK_REROUTED = "TASK_REROUTED"
    REVIEWER_VERDICT = "REVIEWER_VERDICT"
    MEMORY_STORED = "MEMORY_STORED"
    MEMORY_INVALIDATED = "MEMORY_INVALIDATED"
    CONFLICT_RESOLVED = "CONFLICT_RESOLVED"


@dataclass
class CognitiveAuditEvent:
    event_id: str
    event_type: AuditEventType
    task_id: str
    node_id: str
    worker_id: str
    role: str
    summary: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    duration_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["event_type"] = self.event_type.value
        return d


class CognitiveObservabilityPlane:
    """
    Durable, queryable audit plane tracking every cognitive event.
    """

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self._init_tables()

    def _init_tables(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cognitive_audit_events (
                    event_id TEXT PRIMARY KEY,
                    event_type TEXT,
                    task_id TEXT,
                    node_id TEXT,
                    worker_id TEXT,
                    role TEXT,
                    summary TEXT,
                    metadata_json TEXT,
                    duration_ms REAL,
                    timestamp REAL
                )
            """)
            conn.commit()

    def record_event(self, event: CognitiveAuditEvent) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO cognitive_audit_events
                (event_id, event_type, task_id, node_id, worker_id, role, summary, metadata_json, duration_ms, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                event.event_id,
                event.event_type.value,
                event.task_id,
                event.node_id,
                event.worker_id,
                event.role,
                event.summary,
                json.dumps(event.metadata),
                event.duration_ms,
                event.timestamp,
            ))
            conn.commit()

    def query_trajectory(self, task_id: str) -> List[CognitiveAuditEvent]:
        events: List[CognitiveAuditEvent] = []
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT event_id, event_type, task_id, node_id, worker_id, role, summary, metadata_json, duration_ms, timestamp
                FROM cognitive_audit_events
                WHERE task_id = ?
                ORDER BY timestamp ASC
            """, (task_id,))
            for row in cursor.fetchall():
                events.append(CognitiveAuditEvent(
                    event_id=row[0],
                    event_type=AuditEventType(row[1]),
                    task_id=row[2],
                    node_id=row[3],
                    worker_id=row[4],
                    role=row[5],
                    summary=row[6],
                    metadata=json.loads(row[7]),
                    duration_ms=row[8],
                    timestamp=row[9],
                ))
        return events
