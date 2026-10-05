"""
ChakrView Step 134: Cognitive Recovery & Self-Healing Wave.

Structured cognitive recovery pipeline:
DETECT -> CLASSIFY -> PERSIST -> DIAGNOSE -> REPLAN -> REPAIR -> VERIFY -> RECORD LESSON

Bounds recovery attempts to avoid infinite loops and updates cognitive strategy records.
"""

from __future__ import annotations

import enum
import json
import sqlite3
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from chakrview.cognition.multi_agent.contracts import WorkerRole, WorkerContract, WorkerPackage
from chakrview.cognition.multi_agent.result import WorkerResult, WorkerExecutionStatus
from chakrview.cognition.governed_learning.self_evaluator import GovernedFailureAnalysis, FailureClass


class SelfHealingAction(str, enum.Enum):
    RETRY_SAME_WORKER = "RETRY_SAME_WORKER"
    REROUTE_FALLBACK_WORKER = "REROUTE_FALLBACK_WORKER"
    REPLAN_TASK_DECOMPOSITION = "REPLAN_TASK_DECOMPOSITION"
    ABORT_WITH_GOVERNED_ERROR = "ABORT_WITH_GOVERNED_ERROR"


@dataclass
class SelfHealingRecord:
    healing_id: str
    task_id: str
    failure_class: str
    root_cause: str
    action_taken: SelfHealingAction
    rerouted_worker: Optional[str]
    success: bool
    lesson_learned: str
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["action_taken"] = self.action_taken.value
        return d


class CognitiveSelfHealingEngine:
    """
    Drives bounded cognitive self-healing loops with explicit diagnosis and lesson recording.
    """

    def __init__(self, db_path: Path, max_attempts_per_task: int = 3) -> None:
        self.db_path = Path(db_path)
        self.max_attempts_per_task = max_attempts_per_task
        self._init_tables()

    def _init_tables(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cognitive_self_healing_records (
                    healing_id TEXT PRIMARY KEY,
                    task_id TEXT,
                    failure_class TEXT,
                    root_cause TEXT,
                    action_taken TEXT,
                    rerouted_worker TEXT,
                    success INTEGER,
                    lesson_learned TEXT,
                    timestamp REAL
                )
            """)
            conn.commit()

    def determine_healing_action(
        self,
        attempt_number: int,
        error_msg: str,
        has_fallback_worker: bool,
    ) -> SelfHealingAction:
        if attempt_number >= self.max_attempts_per_task:
            return SelfHealingAction.ABORT_WITH_GOVERNED_ERROR
        if "timeout" in error_msg.lower() or "crash" in error_msg.lower():
            return SelfHealingAction.REROUTE_FALLBACK_WORKER if has_fallback_worker else SelfHealingAction.RETRY_SAME_WORKER
        if "verification" in error_msg.lower() or "test" in error_msg.lower():
            return SelfHealingAction.REPLAN_TASK_DECOMPOSITION
        return SelfHealingAction.REROUTE_FALLBACK_WORKER if has_fallback_worker else SelfHealingAction.ABORT_WITH_GOVERNED_ERROR

    def record_healing_event(self, record: SelfHealingRecord) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO cognitive_self_healing_records
                (healing_id, task_id, failure_class, root_cause, action_taken, rerouted_worker, success, lesson_learned, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record.healing_id,
                record.task_id,
                record.failure_class,
                record.root_cause,
                record.action_taken.value,
                record.rerouted_worker,
                1 if record.success else 0,
                record.lesson_learned,
                record.timestamp,
            ))
            conn.commit()
