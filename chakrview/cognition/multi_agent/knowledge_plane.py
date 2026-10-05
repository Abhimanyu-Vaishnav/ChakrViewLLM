"""
ChakrView Step 131: Federated Memory Consistency & Knowledge Provenance.

Expands memory plane with:
- Full provenance chain: author node, worker role, generating task ID, causal parent fact IDs
- Explicit tombstone / invalidation semantics (soft-delete with reason)
- Logical partition namespaces: LOCAL_MEMORY, FEDERATED_MEMORY, PROJECT_KNOWLEDGE, TASK_HISTORY
- Invalidation propagation without destructive loss of audit history
"""

from __future__ import annotations

import enum
import json
import sqlite3
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


class MemoryPartition(str, enum.Enum):
    LOCAL_MEMORY = "LOCAL_MEMORY"
    FEDERATED_MEMORY = "FEDERATED_MEMORY"
    PROJECT_KNOWLEDGE = "PROJECT_KNOWLEDGE"
    TASK_HISTORY = "TASK_HISTORY"


@dataclass
class KnowledgeFactRecord:
    fact_id: str
    partition: MemoryPartition
    subject: str
    predicate: str
    object_value: str
    author_node_id: str
    author_worker_id: str
    task_id: str
    confidence: float = 1.0
    parent_fact_ids: List[str] = field(default_factory=list)
    is_invalidated: bool = False
    invalidation_reason: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["partition"] = self.partition.value
        return d


class GovernedKnowledgePlane:
    """
    Durable knowledge store enforcing explicit provenance, partitioning, and tombstone invalidations.
    """

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self._init_tables()

    def _init_tables(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS knowledge_facts (
                    fact_id TEXT PRIMARY KEY,
                    partition TEXT,
                    subject TEXT,
                    predicate TEXT,
                    object_value TEXT,
                    author_node_id TEXT,
                    author_worker_id TEXT,
                    task_id TEXT,
                    confidence REAL,
                    parent_fact_ids_json TEXT,
                    is_invalidated INTEGER,
                    invalidation_reason TEXT,
                    created_at REAL,
                    updated_at REAL
                )
            """)
            conn.commit()

    def store_fact(self, fact: KnowledgeFactRecord) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO knowledge_facts
                (fact_id, partition, subject, predicate, object_value, author_node_id, author_worker_id,
                 task_id, confidence, parent_fact_ids_json, is_invalidated, invalidation_reason, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                fact.fact_id,
                fact.partition.value,
                fact.subject,
                fact.predicate,
                fact.object_value,
                fact.author_node_id,
                fact.author_worker_id,
                fact.task_id,
                fact.confidence,
                json.dumps(fact.parent_fact_ids),
                1 if fact.is_invalidated else 0,
                fact.invalidation_reason,
                fact.created_at,
                fact.updated_at,
            ))
            conn.commit()

    def invalidate_fact(self, fact_id: str, reason: str) -> bool:
        """Applies tombstone invalidation."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT fact_id FROM knowledge_facts WHERE fact_id = ?", (fact_id,))
            if not cursor.fetchone():
                return False
            conn.execute("""
                UPDATE knowledge_facts
                SET is_invalidated = 1, invalidation_reason = ?, updated_at = ?
                WHERE fact_id = ?
            """, (reason, time.time(), fact_id))
            conn.commit()
            return True

    def query_active_facts(
        self,
        subject: Optional[str] = None,
        partition: Optional[MemoryPartition] = None,
    ) -> List[KnowledgeFactRecord]:
        query = "SELECT fact_id, partition, subject, predicate, object_value, author_node_id, author_worker_id, task_id, confidence, parent_fact_ids_json, is_invalidated, invalidation_reason, created_at, updated_at FROM knowledge_facts WHERE is_invalidated = 0"
        clauses = []
        params = []
        if subject:
            clauses.append("subject = ?")
            params.append(subject)
        if partition:
            clauses.append("partition = ?")
            params.append(partition.value)
        if clauses:
            query += " AND " + " AND ".join(clauses)

        results: List[KnowledgeFactRecord] = []
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(query, tuple(params))
            for row in cursor.fetchall():
                results.append(KnowledgeFactRecord(
                    fact_id=row[0],
                    partition=MemoryPartition(row[1]),
                    subject=row[2],
                    predicate=row[3],
                    object_value=row[4],
                    author_node_id=row[5],
                    author_worker_id=row[6],
                    task_id=row[7],
                    confidence=row[8],
                    parent_fact_ids=json.loads(row[9]),
                    is_invalidated=bool(row[10]),
                    invalidation_reason=row[11],
                    created_at=row[12],
                    updated_at=row[13],
                ))
        return results
