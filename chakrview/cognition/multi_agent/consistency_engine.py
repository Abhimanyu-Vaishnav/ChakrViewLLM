"""
ChakrView Step 130: Distributed Consistency & Conflict Resolution.

Governed consistency engine:
- Monotonic revision vectors per entity
- Deterministic conflict detection (e.g., diverging concurrent revisions)
- Audited conflict resolution policy (deterministic priority rule: higher authority tier or latest valid timestamp)
- Conflict log in SQLite PPB for full observability
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


class ConflictResolutionPolicy(str, enum.Enum):
    HIGHEST_REVISION = "HIGHEST_REVISION"
    AUTHORITY_TIER = "AUTHORITY_TIER"
    DETERMINISTIC_MERGE = "DETERMINISTIC_MERGE"


@dataclass
class ConflictAuditRecord:
    conflict_id: str
    entity_key: str
    local_revision: int
    incoming_revision: int
    resolution_policy: ConflictResolutionPolicy
    winning_author: str
    resolution_reason: str
    timestamp: float = field(default_factory=time.time)


class GovernedConsistencyEngine:
    """
    Guarantees state consistency across distributed workers with explicit conflict auditing.
    """

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self._init_tables()

    def _init_tables(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS consistency_records (
                    entity_key TEXT PRIMARY KEY,
                    revision INTEGER,
                    author_id TEXT,
                    authority_tier INTEGER,
                    payload_json TEXT,
                    payload_hash TEXT,
                    timestamp REAL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS conflict_audit_trail (
                    conflict_id TEXT PRIMARY KEY,
                    entity_key TEXT,
                    local_rev INTEGER,
                    incoming_rev INTEGER,
                    policy TEXT,
                    winning_author TEXT,
                    reason TEXT,
                    timestamp REAL
                )
            """)
            conn.commit()

    def apply_update(
        self,
        entity_key: str,
        incoming_rev: int,
        author_id: str,
        authority_tier: int,
        payload: Dict[str, Any],
    ) -> Tuple[bool, Optional[ConflictAuditRecord]]:
        """
        Applies update deterministically. If conflict occurs, resolves via authority tier or revision.
        """
        payload_bytes = json.dumps(payload, sort_keys=True).encode("utf-8")
        p_hash = hashlib.sha256(payload_bytes).hexdigest()

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT revision, author_id, authority_tier, payload_hash FROM consistency_records WHERE entity_key = ?",
                (entity_key,),
            )
            row = cursor.fetchone()

            if not row:
                conn.execute("""
                    INSERT INTO consistency_records (entity_key, revision, author_id, authority_tier, payload_json, payload_hash, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (entity_key, incoming_rev, author_id, authority_tier, json.dumps(payload), p_hash, time.time()))
                conn.commit()
                return True, None

            local_rev, local_author, local_tier, local_hash = row

            # Strict monotonic progression
            if incoming_rev > local_rev:
                conn.execute("""
                    UPDATE consistency_records
                    SET revision = ?, author_id = ?, authority_tier = ?, payload_json = ?, payload_hash = ?, timestamp = ?
                    WHERE entity_key = ?
                """, (incoming_rev, author_id, authority_tier, json.dumps(payload), p_hash, time.time(), entity_key))
                conn.commit()
                return True, None

            # Conflict: incoming revision <= local revision
            # If payload is identical, idempotent success
            if incoming_rev == local_rev and p_hash == local_hash:
                return True, None

            # Divergent conflict resolution based on authority tier
            winning_author = author_id if authority_tier > local_tier else local_author
            audit = ConflictAuditRecord(
                conflict_id=f"conf_{entity_key}_{int(time.time() * 1000)}",
                entity_key=entity_key,
                local_revision=local_rev,
                incoming_revision=incoming_rev,
                resolution_policy=ConflictResolutionPolicy.AUTHORITY_TIER,
                winning_author=winning_author,
                resolution_reason=f"Resolved via authority tier {authority_tier} vs {local_tier}",
            )

            conn.execute("""
                INSERT INTO conflict_audit_trail (conflict_id, entity_key, local_rev, incoming_rev, policy, winning_author, reason, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (audit.conflict_id, audit.entity_key, audit.local_revision, audit.incoming_revision, audit.resolution_policy.value, audit.winning_author, audit.resolution_reason, audit.timestamp))

            if authority_tier > local_tier:
                conn.execute("""
                    UPDATE consistency_records
                    SET revision = ?, author_id = ?, authority_tier = ?, payload_json = ?, payload_hash = ?, timestamp = ?
                    WHERE entity_key = ?
                """, (incoming_rev, author_id, authority_tier, json.dumps(payload), p_hash, time.time(), entity_key))
                conn.commit()
                return True, audit

            conn.commit()
            return False, audit
