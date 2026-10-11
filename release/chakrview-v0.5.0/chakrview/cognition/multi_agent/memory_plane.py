"""
ChakrView Step 124: Federated Persistent Memory & Knowledge Plane.

Maintains structured, compact, provenance-aware persistent memory:
- ProvenanceMemoryRecord: fact_id, subject, predicate, object_value, author_node_id, task_id, confidence
- FederatedMemoryPlane: Durable indexing, query by subject/task, delta updates
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class ProvenanceMemoryRecord:
    fact_id: str
    subject: str
    predicate: str
    object_value: str
    author_node_id: str
    task_id: str
    confidence: float = 1.0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class FederatedMemoryPlane:
    """
    Manages durable structured facts and project knowledge with explicit provenance.
    Never stores unbounded conversational transcripts or full repository text.
    """

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self._init_memory_tables()

    def _init_memory_tables(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS federated_provenance_memory (
                    fact_id TEXT PRIMARY KEY,
                    subject TEXT,
                    predicate TEXT,
                    object_value TEXT,
                    author_node_id TEXT,
                    task_id TEXT,
                    confidence REAL,
                    timestamp REAL
                )
            """)
            conn.commit()

    def store_fact(self, record: ProvenanceMemoryRecord) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO federated_provenance_memory
                (fact_id, subject, predicate, object_value, author_node_id, task_id, confidence, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record.fact_id,
                record.subject,
                record.predicate,
                record.object_value,
                record.author_node_id,
                record.task_id,
                record.confidence,
                record.timestamp,
            ))
            conn.commit()

    def query_facts(self, subject: Optional[str] = None, task_id: Optional[str] = None) -> List[ProvenanceMemoryRecord]:
        query = "SELECT fact_id, subject, predicate, object_value, author_node_id, task_id, confidence, timestamp FROM federated_provenance_memory"
        clauses = []
        params = []
        if subject:
            clauses.append("subject = ?")
            params.append(subject)
        if task_id:
            clauses.append("task_id = ?")
            params.append(task_id)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)

        results: List[ProvenanceMemoryRecord] = []
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(query, tuple(params))
            for row in cursor.fetchall():
                results.append(ProvenanceMemoryRecord(
                    fact_id=row[0],
                    subject=row[1],
                    predicate=row[2],
                    object_value=row[3],
                    author_node_id=row[4],
                    task_id=row[5],
                    confidence=row[6],
                    timestamp=row[7],
                ))
        return results
