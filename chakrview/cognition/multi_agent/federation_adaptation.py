"""
ChakrView Step 127: Governed Self-Evaluation, Learning & Federation Adaptation.

Adapts federation scheduling policies based on empirical failure analysis:
- Failure Pattern Extraction: Tracks worker failure rates per task resource type
- Adaptive Routing Policy: Excludes or deprioritizes workers with proven failure history
- Preserves governance: System NEVER modifies core weights or bypasses invariants
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from chakrview.cognition.multi_agent.contracts import WorkerRole
from chakrview.cognition.multi_agent.resource_federation import WorkerResourceProfile


@dataclass
class WorkerReliabilityRecord:
    worker_id: str
    task_role: str
    total_executions: int = 0
    failure_count: int = 0
    crash_count: int = 0
    timeout_count: int = 0

    @property
    def reliability_score(self) -> float:
        if self.total_executions == 0:
            return 1.0
        return max(0.0, 1.0 - (self.failure_count / float(self.total_executions)))


class FederationAdaptationManager:
    """
    Learns empirical worker reliability from execution history and dynamically
    adapts worker selection weights.
    """

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self._init_adaptation_tables()

    def _init_adaptation_tables(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS worker_reliability (
                    worker_id TEXT,
                    task_role TEXT,
                    total_executions INTEGER,
                    failure_count INTEGER,
                    crash_count INTEGER,
                    timeout_count INTEGER,
                    PRIMARY KEY (worker_id, task_role)
                )
            """)
            conn.commit()

    def record_outcome(self, worker_id: str, role: WorkerRole, is_success: bool, failure_type: Optional[str] = None) -> None:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT total_executions, failure_count, crash_count, timeout_count FROM worker_reliability WHERE worker_id = ? AND task_role = ?",
                (worker_id, role.value),
            )
            row = cursor.fetchone()
            if row:
                tot, fail, crash, tout = row
            else:
                tot, fail, crash, tout = 0, 0, 0, 0

            tot += 1
            if not is_success:
                fail += 1
                if failure_type == "PROCESS_CRASH":
                    crash += 1
                elif failure_type == "TIMEOUT":
                    tout += 1

            conn.execute("""
                INSERT OR REPLACE INTO worker_reliability
                (worker_id, task_role, total_executions, failure_count, crash_count, timeout_count)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (worker_id, role.value, tot, fail, crash, tout))
            conn.commit()

    def get_worker_reliability(self, worker_id: str, role: WorkerRole) -> float:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT total_executions, failure_count FROM worker_reliability WHERE worker_id = ? AND task_role = ?",
                (worker_id, role.value),
            )
            row = cursor.fetchone()
            if not row or row[0] == 0:
                return 1.0
            return max(0.0, 1.0 - (row[1] / float(row[0])))
