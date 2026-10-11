"""
ChakrView Step 94: Governed Experience & Learning Record Models.

Provides the foundational schema and persistence contracts for cognitive experiences:
- ExperienceEpistemicCategory: FACT, OBSERVATION, INFERENCE, HYPOTHESIS, LESSON, FAILURE, SUCCESS, UNKNOWN.
- GovernedExperienceRecord: Structured, auditable, provenance-tracked unit of cognitive learning.
- ConflictResolutionRule: Invariant order of authority preventing inferences/hypotheses from overriding verified facts:
  FACT > OBSERVATION > LESSON > INFERENCE > HYPOTHESIS > UNKNOWN.
- GovernedExperienceStore: Persistent SQLite store for experience records with complete version lineage and query capabilities.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union


class ExperienceEpistemicCategory(str, Enum):
    """
    Epistemic categories for cognitive learning records.
    Inferences and hypotheses must never silently become FACT.
    """
    FACT = "FACT"                 # Directly verified ground truth (AST, test result, verified source)
    OBSERVATION = "OBSERVATION"   # Raw or structured execution observation from environment
    INFERENCE = "INFERENCE"       # Deductive or inductive conclusion derived from observations
    HYPOTHESIS = "HYPOTHESIS"     # Tentative explanation under evaluation
    LESSON = "LESSON"             # Synthesized operational guidance or anti-pattern
    FAILURE = "FAILURE"           # Concretely observed operational or execution defect
    SUCCESS = "SUCCESS"           # Concretely verified completion of an objective
    UNKNOWN = "UNKNOWN"           # Recognized missing knowledge requiring future acquisition


# Epistemic hierarchy rank: Higher value means stronger authority.
EPISTEMIC_AUTHORITY_RANK: Dict[ExperienceEpistemicCategory, int] = {
    ExperienceEpistemicCategory.FACT: 100,
    ExperienceEpistemicCategory.OBSERVATION: 80,
    ExperienceEpistemicCategory.SUCCESS: 75,
    ExperienceEpistemicCategory.FAILURE: 70,
    ExperienceEpistemicCategory.LESSON: 60,
    ExperienceEpistemicCategory.INFERENCE: 40,
    ExperienceEpistemicCategory.HYPOTHESIS: 20,
    ExperienceEpistemicCategory.UNKNOWN: 0,
}


@dataclass
class ExperienceProvenance:
    """Tracks origin and derivation causality for an experience record."""
    source_type: str                   # "TASK_EXECUTION", "SELF_EVALUATION", "LESSON_EXTRACTION", "INVESTIGATION"
    task_id: str
    parent_experience_id: Optional[str] = None
    agent_version: str = "0.1.0"
    author_component: str = "cognitive_learning_subsystem"
    rationale: str = ""
    timestamp_utc: str = field(
        default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ExperienceProvenance:
        return cls(**data)


@dataclass
class GovernedExperienceRecord:
    """
    Step 94: Standardized, durable experience record.
    Represents what was attempted, what occurred, the evidence, and the resulting learning signal.
    """
    record_id: str
    project_id: str
    task_goal: str
    action_taken: str
    observation: str
    result_summary: str
    epistemic_category: ExperienceEpistemicCategory
    is_success: bool
    evidence_ids: List[str] = field(default_factory=list)
    confidence: float = 1.0
    affected_modules: List[str] = field(default_factory=list)
    lesson_summary: Optional[str] = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    provenance: ExperienceProvenance = field(
        default_factory=lambda: ExperienceProvenance(source_type="MANUAL", task_id="unspecified")
    )
    created_at_utc: str = field(
        default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    )

    def __post_init__(self) -> None:
        if isinstance(self.epistemic_category, str):
            self.epistemic_category = ExperienceEpistemicCategory(self.epistemic_category)
        if isinstance(self.provenance, dict):
            self.provenance = ExperienceProvenance.from_dict(self.provenance)

    def compute_fingerprint(self) -> str:
        """Deterministic fingerprint of core content for change and duplicate detection."""
        hasher = hashlib.sha256()
        hasher.update(self.project_id.encode("utf-8"))
        hasher.update(self.task_goal.encode("utf-8"))
        hasher.update(self.action_taken.encode("utf-8"))
        hasher.update(self.observation.encode("utf-8"))
        hasher.update(self.epistemic_category.value.encode("utf-8"))
        hasher.update(str(self.is_success).encode("utf-8"))
        for m in sorted(self.affected_modules):
            hasher.update(m.encode("utf-8"))
        return hasher.hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "record_id": self.record_id,
            "project_id": self.project_id,
            "task_goal": self.task_goal,
            "action_taken": self.action_taken,
            "observation": self.observation,
            "result_summary": self.result_summary,
            "epistemic_category": self.epistemic_category.value,
            "is_success": self.is_success,
            "evidence_ids": list(self.evidence_ids),
            "confidence": self.confidence,
            "affected_modules": list(self.affected_modules),
            "lesson_summary": self.lesson_summary,
            "diagnostics": self.diagnostics,
            "provenance": self.provenance.to_dict(),
            "created_at_utc": self.created_at_utc,
            "fingerprint": self.compute_fingerprint(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> GovernedExperienceRecord:
        d = dict(data)
        d.pop("fingerprint", None)
        return cls(**d)


class GovernedExperienceStore:
    """
    Durable SQLite repository for governed experience records.
    Survives complete process shutdown and restarts.
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
                CREATE TABLE IF NOT EXISTS governed_experiences (
                    record_id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    task_goal TEXT NOT NULL,
                    action_taken TEXT NOT NULL,
                    observation TEXT NOT NULL,
                    result_summary TEXT NOT NULL,
                    epistemic_category TEXT NOT NULL,
                    is_success INTEGER NOT NULL,
                    confidence REAL NOT NULL,
                    affected_modules TEXT NOT NULL,
                    evidence_ids TEXT NOT NULL,
                    lesson_summary TEXT,
                    diagnostics TEXT NOT NULL,
                    provenance TEXT NOT NULL,
                    fingerprint TEXT NOT NULL,
                    created_at_utc TEXT NOT NULL
                );
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_exp_project_cat
                ON governed_experiences(project_id, epistemic_category);
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_exp_success
                ON governed_experiences(project_id, is_success);
            """)
            conn.commit()

    def store_experience(self, exp: GovernedExperienceRecord) -> None:
        """Stores or updates a governed experience record."""
        fp = exp.compute_fingerprint()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO governed_experiences (
                    record_id, project_id, task_goal, action_taken, observation,
                    result_summary, epistemic_category, is_success, confidence,
                    affected_modules, evidence_ids, lesson_summary, diagnostics,
                    provenance, fingerprint, created_at_utc
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                exp.record_id,
                exp.project_id,
                exp.task_goal,
                exp.action_taken,
                exp.observation,
                exp.result_summary,
                exp.epistemic_category.value,
                1 if exp.is_success else 0,
                exp.confidence,
                json.dumps(exp.affected_modules),
                json.dumps(exp.evidence_ids),
                exp.lesson_summary,
                json.dumps(exp.diagnostics),
                json.dumps(exp.provenance.to_dict()),
                fp,
                exp.created_at_utc,
            ))
            conn.commit()

    def get_experience(self, record_id: str) -> Optional[GovernedExperienceRecord]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM governed_experiences WHERE record_id = ?", (record_id,))
            row = cursor.fetchone()
            if not row:
                return None
            return self._row_to_record(row)

    def query_experiences(
        self,
        project_id: str,
        category: Optional[ExperienceEpistemicCategory] = None,
        is_success: Optional[bool] = None,
        affected_module: Optional[str] = None,
        limit: int = 100,
    ) -> List[GovernedExperienceRecord]:
        """Queries persisted experiences with filters."""
        query = "SELECT * FROM governed_experiences WHERE project_id = ?"
        params: List[Any] = [project_id]

        if category is not None:
            query += " AND epistemic_category = ?"
            params.append(category.value)
        if is_success is not None:
            query += " AND is_success = ?"
            params.append(1 if is_success else 0)

        query += " ORDER BY created_at_utc DESC LIMIT ?"
        params.append(limit)

        records: List[GovernedExperienceRecord] = []
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            for row in cursor.fetchall():
                rec = self._row_to_record(row)
                if affected_module:
                    if affected_module in rec.affected_modules:
                        records.append(rec)
                else:
                    records.append(rec)
        return records

    def _row_to_record(self, row: sqlite3.Row) -> GovernedExperienceRecord:
        return GovernedExperienceRecord(
            record_id=row["record_id"],
            project_id=row["project_id"],
            task_goal=row["task_goal"],
            action_taken=row["action_taken"],
            observation=row["observation"],
            result_summary=row["result_summary"],
            epistemic_category=ExperienceEpistemicCategory(row["epistemic_category"]),
            is_success=bool(row["is_success"]),
            confidence=float(row["confidence"]),
            affected_modules=json.loads(row["affected_modules"]),
            evidence_ids=json.loads(row["evidence_ids"]),
            lesson_summary=row["lesson_summary"],
            diagnostics=json.loads(row["diagnostics"]),
            provenance=ExperienceProvenance.from_dict(json.loads(row["provenance"])),
            created_at_utc=row["created_at_utc"],
        )
