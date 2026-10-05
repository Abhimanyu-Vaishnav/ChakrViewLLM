"""
ChakrView Step 99: Persistent Evaluation & Regression Memory.

Maintains historical records of verified capabilities and regression status:
- Tracks:
  test_identity, capability_name, input_condition, expected_result, actual_result,
  passed, regression_status, resource_profile, model_weight_hash, timestamp_utc.
- Answers:
  - "What capabilities have actually been proven?"
  - "What capabilities are only partially proven?"
  - "What remains unproven?"
  - "Did a recent change regress anything?"
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union


class CapabilityProofStatus(str, Enum):
    PROVEN = "PROVEN"
    PARTIALLY_PROVEN = "PARTIALLY_PROVEN"
    UNPROVEN = "UNPROVEN"
    BLOCKED = "BLOCKED"


@dataclass
class EvaluationMemoryRecord:
    """
    Step 99: Historical test and evaluation verification entry.
    """
    eval_id: str
    capability_name: str
    test_identity: str
    input_condition: str
    expected_result: str
    actual_result: str
    passed: bool
    proof_status: CapabilityProofStatus
    resource_profile: str
    model_weight_hash: str
    timestamp_utc: str = field(
        default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "eval_id": self.eval_id,
            "capability_name": self.capability_name,
            "test_identity": self.test_identity,
            "input_condition": self.input_condition,
            "expected_result": self.expected_result,
            "actual_result": self.actual_result,
            "passed": self.passed,
            "proof_status": self.proof_status.value,
            "resource_profile": self.resource_profile,
            "model_weight_hash": self.model_weight_hash,
            "timestamp_utc": self.timestamp_utc,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EvaluationMemoryRecord:
        d = dict(data)
        d["proof_status"] = CapabilityProofStatus(d["proof_status"])
        return cls(**d)


class EvaluationRegressionMemory:
    """
    Durable SQLite store for tracking capability proofs and regression history.
    """

    def __init__(self, db_path: Union[str, Path]) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_tables()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_tables(self) -> None:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS evaluation_memory (
                    eval_id TEXT PRIMARY KEY,
                    capability_name TEXT NOT NULL,
                    test_identity TEXT NOT NULL,
                    input_condition TEXT NOT NULL,
                    expected_result TEXT NOT NULL,
                    actual_result TEXT NOT NULL,
                    passed INTEGER NOT NULL,
                    proof_status TEXT NOT NULL,
                    resource_profile TEXT NOT NULL,
                    model_weight_hash TEXT NOT NULL,
                    timestamp_utc TEXT NOT NULL
                );
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_eval_capability
                ON evaluation_memory(capability_name, passed);
            """)
            conn.commit()

    def record_evaluation(self, record: EvaluationMemoryRecord) -> None:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO evaluation_memory (
                    eval_id, capability_name, test_identity, input_condition,
                    expected_result, actual_result, passed, proof_status,
                    resource_profile, model_weight_hash, timestamp_utc
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record.eval_id,
                record.capability_name,
                record.test_identity,
                record.input_condition,
                record.expected_result,
                record.actual_result,
                1 if record.passed else 0,
                record.proof_status.value,
                record.resource_profile,
                record.model_weight_hash,
                record.timestamp_utc,
            ))
            conn.commit()

    def get_capability_status(self, capability_name: str) -> CapabilityProofStatus:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT passed, proof_status FROM evaluation_memory WHERE capability_name = ? ORDER BY timestamp_utc DESC",
                (capability_name,),
            )
            rows = cursor.fetchall()
            if not rows:
                return CapabilityProofStatus.UNPROVEN
            # Check latest
            latest = rows[0]
            if latest["passed"] == 1:
                return CapabilityProofStatus(latest["proof_status"])
            return CapabilityProofStatus.BLOCKED

    def get_summary_report(self) -> Dict[str, Any]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT capability_name FROM evaluation_memory")
            caps = [r["capability_name"] for r in cursor.fetchall()]

            proven: List[str] = []
            partial: List[str] = []
            unproven: List[str] = []
            blocked: List[str] = []

            for cap in caps:
                st = self.get_capability_status(cap)
                if st == CapabilityProofStatus.PROVEN:
                    proven.append(cap)
                elif st == CapabilityProofStatus.PARTIALLY_PROVEN:
                    partial.append(cap)
                elif st == CapabilityProofStatus.BLOCKED:
                    blocked.append(cap)
                else:
                    unproven.append(cap)

            return {
                "proven_count": len(proven),
                "partially_proven_count": len(partial),
                "unproven_count": len(unproven),
                "blocked_count": len(blocked),
                "proven_capabilities": sorted(proven),
                "partially_proven_capabilities": sorted(partial),
                "blocked_capabilities": sorted(blocked),
            }
