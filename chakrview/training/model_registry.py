"""
ChakrView Governed Model and Checkpoint Registry.

Maintains an immutable, auditable catalog of all neural models:
- ModelStatus: CANONICAL_BASELINE, EXPERIMENTAL_CANDIDATE, VALIDATED_CANDIDATE, RELEASED_MODEL, REJECTED_MODEL
- ModelRecord: Tracks model_hash, architecture identity, parameter count, tokenizer identity,
  dataset identity, metrics, evaluation status, parent model, creation metadata.
- Prevents silent replacement of the canonical baseline.
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


class ModelStatus(str, Enum):
    CANONICAL_BASELINE = "CANONICAL_BASELINE"
    EXPERIMENTAL_CANDIDATE = "EXPERIMENTAL_CANDIDATE"
    VALIDATED_CANDIDATE = "VALIDATED_CANDIDATE"
    RELEASED_MODEL = "RELEASED_MODEL"
    REJECTED_MODEL = "REJECTED_MODEL"


@dataclass
class ModelRecord:
    """
    Immutable metadata record for a trained or canonical ChakrMicro model checkpoint.
    """
    model_id: str
    model_hash: str
    architecture: str
    parameter_count: int
    tokenizer_checksum: str
    dataset_identity: str
    status: ModelStatus
    checkpoint_path: str
    parent_model_id: Optional[str] = None
    validation_loss: Optional[float] = None
    validation_ppl: Optional[float] = None
    top5_syntactic_acc: Optional[float] = None
    experiment_id: str = ""
    creation_timestamp_utc: str = field(
        default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    )
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "model_hash": self.model_hash,
            "architecture": self.architecture,
            "parameter_count": self.parameter_count,
            "tokenizer_checksum": self.tokenizer_checksum,
            "dataset_identity": self.dataset_identity,
            "status": self.status.value,
            "checkpoint_path": self.checkpoint_path,
            "parent_model_id": self.parent_model_id,
            "validation_loss": self.validation_loss,
            "validation_ppl": self.validation_ppl,
            "top5_syntactic_acc": self.top5_syntactic_acc,
            "experiment_id": self.experiment_id,
            "creation_timestamp_utc": self.creation_timestamp_utc,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ModelRecord:
        d = dict(data)
        d["status"] = ModelStatus(d["status"])
        return cls(**d)


class ModelRegistry:
    """
    Durable, auditable registry storing neural model records.
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
                CREATE TABLE IF NOT EXISTS model_registry (
                    model_id TEXT PRIMARY KEY,
                    model_hash TEXT NOT NULL,
                    architecture TEXT NOT NULL,
                    parameter_count INTEGER NOT NULL,
                    tokenizer_checksum TEXT NOT NULL,
                    dataset_identity TEXT NOT NULL,
                    status TEXT NOT NULL,
                    checkpoint_path TEXT NOT NULL,
                    parent_model_id TEXT,
                    validation_loss REAL,
                    validation_ppl REAL,
                    top5_syntactic_acc REAL,
                    experiment_id TEXT,
                    creation_timestamp_utc TEXT NOT NULL,
                    metadata_json TEXT NOT NULL
                )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_model_hash ON model_registry (model_hash)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_model_status ON model_registry (status)")
            conn.commit()

    def register_model(self, record: ModelRecord) -> None:
        """Register a new model record. Rejects overwriting the canonical baseline."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT status FROM model_registry WHERE model_id = ?", (record.model_id,))
            row = cursor.fetchone()
            if row and row["status"] == ModelStatus.CANONICAL_BASELINE.value:
                raise ValueError("Cannot overwrite immutable CANONICAL_BASELINE model in registry.")

            cursor.execute("""
                INSERT OR REPLACE INTO model_registry (
                    model_id, model_hash, architecture, parameter_count, tokenizer_checksum,
                    dataset_identity, status, checkpoint_path, parent_model_id,
                    validation_loss, validation_ppl, top5_syntactic_acc,
                    experiment_id, creation_timestamp_utc, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record.model_id,
                record.model_hash,
                record.architecture,
                record.parameter_count,
                record.tokenizer_checksum,
                record.dataset_identity,
                record.status.value,
                record.checkpoint_path,
                record.parent_model_id,
                record.validation_loss,
                record.validation_ppl,
                record.top5_syntactic_acc,
                record.experiment_id,
                record.creation_timestamp_utc,
                json.dumps(record.metadata),
            ))
            conn.commit()

    def get_model(self, model_id: str) -> Optional[ModelRecord]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM model_registry WHERE model_id = ?", (model_id,))
            row = cursor.fetchone()
            if not row:
                return None
            return self._row_to_record(row)

    def get_by_hash(self, model_hash: str) -> List[ModelRecord]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM model_registry WHERE model_hash = ?", (model_hash,))
            rows = cursor.fetchall()
            return [self._row_to_record(r) for r in rows]

    def list_by_status(self, status: ModelStatus) -> List[ModelRecord]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM model_registry WHERE status = ? ORDER BY creation_timestamp_utc DESC", (status.value,))
            rows = cursor.fetchall()
            return [self._row_to_record(r) for r in rows]

    def _row_to_record(self, row: sqlite3.Row) -> ModelRecord:
        return ModelRecord(
            model_id=row["model_id"],
            model_hash=row["model_hash"],
            architecture=row["architecture"],
            parameter_count=row["parameter_count"],
            tokenizer_checksum=row["tokenizer_checksum"],
            dataset_identity=row["dataset_identity"],
            status=ModelStatus(row["status"]),
            checkpoint_path=row["checkpoint_path"],
            parent_model_id=row["parent_model_id"],
            validation_loss=row["validation_loss"],
            validation_ppl=row["validation_ppl"],
            top5_syntactic_acc=row["top5_syntactic_acc"],
            experiment_id=row["experiment_id"],
            creation_timestamp_utc=row["creation_timestamp_utc"],
            metadata=json.loads(row["metadata_json"]) if row["metadata_json"] else {},
        )
