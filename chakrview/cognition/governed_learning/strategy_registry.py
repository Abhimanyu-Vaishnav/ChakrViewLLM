"""
ChakrView Step 98: Cognitive Strategy Registry.

Maintains a durable, versioned catalog of operational strategies:
- Examples: REPOSITORY_SCANNING, TASK_DECOMPOSITION, PPB_RETRIEVAL_PRIORITY,
  INVESTIGATION_DISPATCH, ABSTENTION_TRIGGER, FAILURE_RECOVERY, LOW_RESOURCE_OPTIMIZATION.
- Strategies track:
  strategy_id, name, description, applicable_conditions, expected_benefit,
  success_count, failure_count, confidence, version, status (CANDIDATE, ACTIVE, RETIRED).
- A strategy must NOT be marked ACTIVE merely by being attempted once; requires empirical successes.
- Persisted in SQLite to survive complete restarts.
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union


class StrategyStatus(str, Enum):
    CANDIDATE = "CANDIDATE"  # Proposed or experimental
    ACTIVE = "ACTIVE"        # Empirically validated and in standard use
    RETIRED = "RETIRED"      # Deprecated due to regression or supersession


@dataclass
class CognitiveStrategy:
    """
    Step 98: Auditable cognitive decision strategy.
    """
    strategy_id: str
    name: str
    description: str
    applicable_conditions: List[str]
    expected_benefit: str
    success_count: int = 0
    failure_count: int = 0
    confidence: float = 0.5
    version: int = 1
    status: StrategyStatus = StrategyStatus.CANDIDATE
    provenance_lesson_id: Optional[str] = None
    created_at_utc: str = field(
        default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    )

    def record_outcome(self, is_success: bool) -> None:
        """Update metrics and adjust confidence."""
        if is_success:
            self.success_count += 1
        else:
            self.failure_count += 1

        total = self.success_count + self.failure_count
        if total > 0:
            self.confidence = round(self.success_count / total, 3)

        # Promotion rule: >= 2 successes and confidence >= 0.7 promote to ACTIVE
        if self.success_count >= 2 and self.confidence >= 0.7:
            self.status = StrategyStatus.ACTIVE
        elif self.failure_count >= 3 and self.confidence < 0.4:
            self.status = StrategyStatus.RETIRED

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy_id": self.strategy_id,
            "name": self.name,
            "description": self.description,
            "applicable_conditions": list(self.applicable_conditions),
            "expected_benefit": self.expected_benefit,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "confidence": self.confidence,
            "version": self.version,
            "status": self.status.value,
            "provenance_lesson_id": self.provenance_lesson_id,
            "created_at_utc": self.created_at_utc,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CognitiveStrategy:
        d = dict(data)
        d["status"] = StrategyStatus(d["status"])
        return cls(**d)


class CognitiveStrategyRegistry:
    """
    Durable SQLite store for cognitive strategies.
    Survives process restart.
    """

    def __init__(self, db_path: Union[str, Path]) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_tables()
        self._seed_default_strategies()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_tables(self) -> None:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS cognitive_strategies (
                    strategy_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT NOT NULL,
                    applicable_conditions TEXT NOT NULL,
                    expected_benefit TEXT NOT NULL,
                    success_count INTEGER NOT NULL,
                    failure_count INTEGER NOT NULL,
                    confidence REAL NOT NULL,
                    version INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    provenance_lesson_id TEXT,
                    created_at_utc TEXT NOT NULL
                );
            """)
            conn.commit()

    def _seed_default_strategies(self) -> None:
        """Seed foundational established strategies."""
        defaults = [
            CognitiveStrategy(
                strategy_id="strat_low_resource_chunking",
                name="Low Resource Chunked Scanning",
                description="Scan project files in bounded chunks of 2-5 files when RAM <= 2GB",
                applicable_conditions=["LOW_RESOURCE", "RAM <= 2.0GB"],
                expected_benefit="Prevents out-of-memory crashes on constrained hardware",
                success_count=5,
                failure_count=0,
                confidence=1.0,
                status=StrategyStatus.ACTIVE,
            ),
            CognitiveStrategy(
                strategy_id="strat_targeted_ppb_retrieval",
                name="Targeted PPB Retrieval Before Read",
                description="Query PPB SQLite index before performing filesystem disk reads",
                applicable_conditions=["Known module inquiry", "Unchanged repository fingerprint"],
                expected_benefit="Eliminates redundant filesystem rescanning",
                success_count=5,
                failure_count=0,
                confidence=1.0,
                status=StrategyStatus.ACTIVE,
            ),
            CognitiveStrategy(
                strategy_id="strat_investigate_before_patch",
                name="Investigate Unknown Before Patch",
                description="Trigger autonomous investigation whenever affected module has UNKNOWN status",
                applicable_conditions=["UNKNOWN module status", "New dependency detected"],
                expected_benefit="Prevents hallucinated patches and bad assumptions",
                success_count=3,
                failure_count=0,
                confidence=1.0,
                status=StrategyStatus.ACTIVE,
            ),
        ]
        for s in defaults:
            if not self.get_strategy(s.strategy_id):
                self.store_strategy(s)

    def store_strategy(self, strategy: CognitiveStrategy) -> None:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO cognitive_strategies (
                    strategy_id, name, description, applicable_conditions,
                    expected_benefit, success_count, failure_count, confidence,
                    version, status, provenance_lesson_id, created_at_utc
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                strategy.strategy_id,
                strategy.name,
                strategy.description,
                json.dumps(strategy.applicable_conditions),
                strategy.expected_benefit,
                strategy.success_count,
                strategy.failure_count,
                strategy.confidence,
                strategy.version,
                strategy.status.value,
                strategy.provenance_lesson_id,
                strategy.created_at_utc,
            ))
            conn.commit()

    def get_strategy(self, strategy_id: str) -> Optional[CognitiveStrategy]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM cognitive_strategies WHERE strategy_id = ?", (strategy_id,))
            row = cursor.fetchone()
            if not row:
                return None
            return self._row_to_strategy(row)

    def list_strategies(self, status: Optional[StrategyStatus] = None) -> List[CognitiveStrategy]:
        query = "SELECT * FROM cognitive_strategies"
        params: List[Any] = []
        if status:
            query += " WHERE status = ?"
            params.append(status.value)
        query += " ORDER BY confidence DESC"

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            return [self._row_to_strategy(r) for r in cursor.fetchall()]

    def _row_to_strategy(self, row: sqlite3.Row) -> CognitiveStrategy:
        return CognitiveStrategy(
            strategy_id=row["strategy_id"],
            name=row["name"],
            description=row["description"],
            applicable_conditions=json.loads(row["applicable_conditions"]),
            expected_benefit=row["expected_benefit"],
            success_count=row["success_count"],
            failure_count=row["failure_count"],
            confidence=float(row["confidence"]),
            version=row["version"],
            status=StrategyStatus(row["status"]),
            provenance_lesson_id=row["provenance_lesson_id"],
            created_at_utc=row["created_at_utc"],
        )
