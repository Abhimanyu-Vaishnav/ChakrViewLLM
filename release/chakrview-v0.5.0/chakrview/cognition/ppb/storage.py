"""
ChakrView Step 78: Persistent Storage Engine for Persistent Project Brain.

Provides durable, SQLite-based and JSON-auditable storage for project knowledge:
- Deterministic SQLite database with ACID transactions, WAL mode, foreign keys.
- Complete versioning and historical knowledge preservation (historical records marked superseded, never deleted).
- Provenance and epistemic status preservation.
- Schema migration mechanism.
- Process restart / crash resilience.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from chakrview.cognition.ppb.models import (
    CURRENT_SCHEMA_VERSION,
    EpistemicStatus,
    KnowledgeRecord,
    KnowledgeRecordType,
    PPBStorageSchemaError,
    ProjectBrainState,
    ProjectIdentity,
)


class PersistentBrainStorage:
    """
    Durable, deterministic persistence engine for Persistent Project Brain.
    Thread-safe and process-safe with SQLite WAL mode.
    """

    def __init__(self, db_path: str, project_identity: Optional[ProjectIdentity] = None) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._init_db()

        if project_identity is not None:
            self.set_project_identity(project_identity)

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _init_db(self) -> None:
        with self._lock:
            with self._get_connection() as conn:
                # Schema version table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS schema_info (
                        version INTEGER PRIMARY KEY,
                        applied_at TEXT NOT NULL
                    );
                """)

                # Check existing schema version
                cur = conn.execute("SELECT MAX(version) FROM schema_info;")
                row = cur.fetchone()
                existing_ver = row[0] if (row and row[0] is not None) else None

                if existing_ver is None:
                    self._create_schema_v1(conn)
                    conn.execute(
                        "INSERT INTO schema_info (version, applied_at) VALUES (?, ?);",
                        (CURRENT_SCHEMA_VERSION, time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())),
                    )
                elif existing_ver < CURRENT_SCHEMA_VERSION:
                    self._migrate(conn, existing_ver, CURRENT_SCHEMA_VERSION)
                elif existing_ver > CURRENT_SCHEMA_VERSION:
                    raise PPBStorageSchemaError(
                        f"Database schema version {existing_ver} is newer than supported {CURRENT_SCHEMA_VERSION}"
                    )

    def _create_schema_v1(self, conn: sqlite3.Connection) -> None:
        # Project identity table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS project_meta (
                project_id TEXT PRIMARY KEY,
                project_root TEXT NOT NULL,
                language TEXT NOT NULL,
                framework TEXT NOT NULL,
                created_at_utc TEXT NOT NULL,
                last_accessed_utc TEXT NOT NULL,
                last_scan_fingerprint TEXT NOT NULL DEFAULT ''
            );
        """)

        # Knowledge records table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS knowledge_records (
                record_id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL,
                record_type TEXT NOT NULL,
                file_path TEXT NOT NULL,
                symbol_name TEXT,
                summary TEXT NOT NULL,
                details_json TEXT NOT NULL,
                dependencies_json TEXT NOT NULL,
                related_symbols_json TEXT NOT NULL,
                evidence_ids_json TEXT NOT NULL,
                epistemic_status TEXT NOT NULL,
                confidence REAL NOT NULL,
                repo_fingerprint TEXT NOT NULL,
                source_chunk TEXT NOT NULL,
                version INTEGER NOT NULL,
                active_version INTEGER NOT NULL DEFAULT 1,
                superseded_by TEXT,
                supersedes TEXT,
                created_at_utc TEXT NOT NULL,
                updated_at_utc TEXT NOT NULL
            );
        """)

        # Indexes for fast deterministic lookup and retrieval
        conn.execute("CREATE INDEX IF NOT EXISTS idx_kr_project ON knowledge_records(project_id);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_kr_file ON knowledge_records(file_path);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_kr_symbol ON knowledge_records(symbol_name);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_kr_active ON knowledge_records(active_version);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_kr_status ON knowledge_records(epistemic_status);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_kr_type ON knowledge_records(record_type);")

        # Scan state / chunk progress table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS scan_progress (
                project_id TEXT NOT NULL,
                file_path TEXT NOT NULL,
                content_sha256 TEXT NOT NULL,
                source_chunk TEXT NOT NULL,
                scanned_at_utc TEXT NOT NULL,
                scan_status TEXT NOT NULL,
                PRIMARY KEY (project_id, file_path)
            );
        """)

    def _migrate(self, conn: sqlite3.Connection, from_ver: int, to_ver: int) -> None:
        # Schema migration strategy hook
        cur_v = from_ver
        while cur_v < to_ver:
            next_v = cur_v + 1
            # Perform atomic step migration when future schema versions exist
            conn.execute(
                "INSERT INTO schema_info (version, applied_at) VALUES (?, ?);",
                (next_v, time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())),
            )
            cur_v = next_v

    # -------------------------------------------------------------------------
    # Project Identity Operations
    # -------------------------------------------------------------------------

    def set_project_identity(self, identity: ProjectIdentity) -> None:
        with self._lock:
            with self._get_connection() as conn:
                conn.execute("""
                    INSERT INTO project_meta (
                        project_id, project_root, language, framework,
                        created_at_utc, last_accessed_utc, last_scan_fingerprint
                    ) VALUES (?, ?, ?, ?, ?, ?, '')
                    ON CONFLICT(project_id) DO UPDATE SET
                        project_root = excluded.project_root,
                        language = excluded.language,
                        framework = excluded.framework,
                        last_accessed_utc = excluded.last_accessed_utc;
                """, (
                    identity.project_id,
                    identity.project_root,
                    identity.language,
                    identity.framework,
                    identity.created_at_utc,
                    identity.last_accessed_utc,
                ))

    def get_project_identity(self, project_id: Optional[str] = None) -> Optional[ProjectIdentity]:
        with self._lock:
            with self._get_connection() as conn:
                if project_id:
                    cur = conn.execute(
                        "SELECT * FROM project_meta WHERE project_id = ? LIMIT 1;", (project_id,)
                    )
                else:
                    cur = conn.execute("SELECT * FROM project_meta LIMIT 1;")
                row = cur.fetchone()
                if not row:
                    return None
                return ProjectIdentity(
                    project_id=row["project_id"],
                    project_root=row["project_root"],
                    language=row["language"],
                    framework=row["framework"],
                    created_at_utc=row["created_at_utc"],
                    last_accessed_utc=row["last_accessed_utc"],
                )

    def update_last_scan_fingerprint(self, project_id: str, fingerprint: str) -> None:
        with self._lock:
            with self._get_connection() as conn:
                conn.execute(
                    "UPDATE project_meta SET last_scan_fingerprint = ?, last_accessed_utc = ? WHERE project_id = ?;",
                    (fingerprint, time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), project_id),
                )

    def get_last_scan_fingerprint(self, project_id: str) -> str:
        with self._lock:
            with self._get_connection() as conn:
                cur = conn.execute("SELECT last_scan_fingerprint FROM project_meta WHERE project_id = ?;", (project_id,))
                row = cur.fetchone()
                return row["last_scan_fingerprint"] if row else ""

    # -------------------------------------------------------------------------
    # Knowledge Record Operations
    # -------------------------------------------------------------------------

    def insert_record(self, record: KnowledgeRecord) -> None:
        """
        Store a knowledge record. If record with same record_id exists,
        version it deterministically, marking the old one superseded.
        """
        with self._lock:
            with self._get_connection() as conn:
                cur = conn.execute("SELECT * FROM knowledge_records WHERE record_id = ?;", (record.record_id,))
                existing = cur.fetchone()

                if existing:
                    # Versioning: mark existing as superseded
                    new_version = existing["version"] + 1
                    superseded_id = f"{record.record_id}_v{existing['version']}"
                    conn.execute("""
                        UPDATE knowledge_records
                        SET active_version = 0,
                            superseded_by = ?
                        WHERE record_id = ?;
                    """, (record.record_id, record.record_id))

                    # Insert copy of old as historical
                    conn.execute("""
                        INSERT INTO knowledge_records (
                            record_id, project_id, record_type, file_path, symbol_name,
                            summary, details_json, dependencies_json, related_symbols_json,
                            evidence_ids_json, epistemic_status, confidence, repo_fingerprint,
                            source_chunk, version, active_version, superseded_by, supersedes,
                            created_at_utc, updated_at_utc
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?, ?);
                    """, (
                        superseded_id,
                        existing["project_id"],
                        existing["record_type"],
                        existing["file_path"],
                        existing["symbol_name"],
                        existing["summary"],
                        existing["details_json"],
                        existing["dependencies_json"],
                        existing["related_symbols_json"],
                        existing["evidence_ids_json"],
                        existing["epistemic_status"],
                        existing["confidence"],
                        existing["repo_fingerprint"],
                        existing["source_chunk"],
                        existing["version"],
                        record.record_id,
                        existing["supersedes"],
                        existing["created_at_utc"],
                        existing["updated_at_utc"],
                    ))

                    # Update the record object with next version
                    record.version = new_version
                    record.supersedes = superseded_id
                    record.active_version = True

                    # Overwrite primary record_id with new version
                    conn.execute("""
                        UPDATE knowledge_records SET
                            project_id = ?, record_type = ?, file_path = ?, symbol_name = ?,
                            summary = ?, details_json = ?, dependencies_json = ?, related_symbols_json = ?,
                            evidence_ids_json = ?, epistemic_status = ?, confidence = ?, repo_fingerprint = ?,
                            source_chunk = ?, version = ?, active_version = 1, superseded_by = NULL,
                            supersedes = ?, created_at_utc = ?, updated_at_utc = ?
                        WHERE record_id = ?;
                    """, (
                        record.project_id,
                        record.record_type.value,
                        record.file_path,
                        record.symbol_name,
                        record.summary,
                        json.dumps(record.details, sort_keys=True),
                        json.dumps(record.dependencies, sort_keys=True),
                        json.dumps(record.related_symbols, sort_keys=True),
                        json.dumps(record.evidence_ids, sort_keys=True),
                        record.epistemic_status.value,
                        record.confidence,
                        record.repo_fingerprint,
                        record.source_chunk,
                        record.version,
                        record.supersedes,
                        record.created_at_utc,
                        record.updated_at_utc,
                        record.record_id,
                    ))
                else:
                    # Fresh insert
                    conn.execute("""
                        INSERT INTO knowledge_records (
                            record_id, project_id, record_type, file_path, symbol_name,
                            summary, details_json, dependencies_json, related_symbols_json,
                            evidence_ids_json, epistemic_status, confidence, repo_fingerprint,
                            source_chunk, version, active_version, superseded_by, supersedes,
                            created_at_utc, updated_at_utc
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                    """, (
                        record.record_id,
                        record.project_id,
                        record.record_type.value,
                        record.file_path,
                        record.symbol_name,
                        record.summary,
                        json.dumps(record.details, sort_keys=True),
                        json.dumps(record.dependencies, sort_keys=True),
                        json.dumps(record.related_symbols, sort_keys=True),
                        json.dumps(record.evidence_ids, sort_keys=True),
                        record.epistemic_status.value,
                        record.confidence,
                        record.repo_fingerprint,
                        record.source_chunk,
                        record.version,
                        1 if record.active_version else 0,
                        record.superseded_by,
                        record.supersedes,
                        record.created_at_utc,
                        record.updated_at_utc,
                    ))

    def get_record(self, record_id: str) -> Optional[KnowledgeRecord]:
        with self._lock:
            with self._get_connection() as conn:
                cur = conn.execute("SELECT * FROM knowledge_records WHERE record_id = ?;", (record_id,))
                row = cur.fetchone()
                return self._row_to_record(row) if row else None

    def query_records(
        self,
        *,
        project_id: Optional[str] = None,
        file_path: Optional[str] = None,
        symbol_name: Optional[str] = None,
        record_type: Optional[KnowledgeRecordType] = None,
        epistemic_status: Optional[EpistemicStatus] = None,
        active_only: bool = True,
        limit: int = 100,
    ) -> List[KnowledgeRecord]:
        with self._lock:
            with self._get_connection() as conn:
                clauses: List[str] = []
                params: List[Any] = []

                if project_id:
                    clauses.append("project_id = ?")
                    params.append(project_id)
                if file_path:
                    clauses.append("file_path = ?")
                    params.append(file_path)
                if symbol_name:
                    clauses.append("symbol_name = ?")
                    params.append(symbol_name)
                if record_type:
                    clauses.append("record_type = ?")
                    params.append(record_type.value)
                if epistemic_status:
                    clauses.append("epistemic_status = ?")
                    params.append(epistemic_status.value)
                if active_only:
                    clauses.append("active_version = 1")

                where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
                sql = f"SELECT * FROM knowledge_records {where} ORDER BY record_id ASC LIMIT ?;"
                params.append(limit)

                cur = conn.execute(sql, tuple(params))
                return [self._row_to_record(row) for row in cur.fetchall()]

    def update_epistemic_status_by_file(
        self,
        file_path: str,
        new_status: EpistemicStatus,
        project_id: Optional[str] = None,
    ) -> int:
        """Mark all active records for a file path with a new epistemic status (e.g. STALE)."""
        with self._lock:
            with self._get_connection() as conn:
                clauses = ["file_path = ?", "active_version = 1"]
                params: List[Any] = [file_path]
                if project_id:
                    clauses.append("project_id = ?")
                    params.append(project_id)

                sql = f"""
                    UPDATE knowledge_records
                    SET epistemic_status = ?,
                        updated_at_utc = ?
                    WHERE {' AND '.join(clauses)};
                """
                params = [new_status.value, time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())] + params
                cur = conn.execute(sql, tuple(params))
                return cur.rowcount

    def update_record_status(self, record_id: str, new_status: EpistemicStatus) -> bool:
        with self._lock:
            with self._get_connection() as conn:
                cur = conn.execute("""
                    UPDATE knowledge_records
                    SET epistemic_status = ?,
                        updated_at_utc = ?
                    WHERE record_id = ?;
                """, (new_status.value, time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), record_id))
                return cur.rowcount > 0

    def get_all_active_files(self, project_id: Optional[str] = None) -> List[str]:
        with self._lock:
            with self._get_connection() as conn:
                if project_id:
                    cur = conn.execute(
                        "SELECT DISTINCT file_path FROM knowledge_records WHERE project_id = ? AND active_version = 1 ORDER BY file_path ASC;",
                        (project_id,),
                    )
                else:
                    cur = conn.execute(
                        "SELECT DISTINCT file_path FROM knowledge_records WHERE active_version = 1 ORDER BY file_path ASC;"
                    )
                return [row["file_path"] for row in cur.fetchall()]

    # -------------------------------------------------------------------------
    # Progress Tracking Operations
    # -------------------------------------------------------------------------

    def record_scan_progress(
        self,
        project_id: str,
        file_path: str,
        content_sha256: str,
        source_chunk: str,
        status: str = "COMPLETED",
    ) -> None:
        with self._lock:
            with self._get_connection() as conn:
                conn.execute("""
                    INSERT INTO scan_progress (
                        project_id, file_path, content_sha256, source_chunk, scanned_at_utc, scan_status
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(project_id, file_path) DO UPDATE SET
                        content_sha256 = excluded.content_sha256,
                        source_chunk = excluded.source_chunk,
                        scanned_at_utc = excluded.scanned_at_utc,
                        scan_status = excluded.scan_status;
                """, (
                    project_id,
                    file_path,
                    content_sha256,
                    source_chunk,
                    time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    status,
                ))

    def get_scanned_files(self, project_id: str) -> Dict[str, str]:
        """Returns dict of {file_path: content_sha256} for all successfully completed files."""
        with self._lock:
            with self._get_connection() as conn:
                cur = conn.execute(
                    "SELECT file_path, content_sha256 FROM scan_progress WHERE project_id = ? AND scan_status = 'COMPLETED';",
                    (project_id,),
                )
                return {row["file_path"]: row["content_sha256"] for row in cur.fetchall()}

    def get_state_summary(self, project_id: str) -> ProjectBrainState:
        with self._lock:
            ident = self.get_project_identity(project_id)
            if not ident:
                ident = ProjectIdentity(project_id=project_id, project_root="")

            with self._get_connection() as conn:
                cur = conn.execute("SELECT COUNT(*) FROM knowledge_records WHERE project_id = ?;", (project_id,))
                total = cur.fetchone()[0]

                cur = conn.execute(
                    "SELECT COUNT(*) FROM knowledge_records WHERE project_id = ? AND active_version = 1;",
                    (project_id,),
                )
                active = cur.fetchone()[0]

                cur = conn.execute(
                    "SELECT COUNT(*) FROM knowledge_records WHERE project_id = ? AND active_version = 1 AND epistemic_status = 'STALE';",
                    (project_id,),
                )
                stale = cur.fetchone()[0]

                cur = conn.execute(
                    "SELECT COUNT(*) FROM scan_progress WHERE project_id = ? AND scan_status = 'COMPLETED';",
                    (project_id,),
                )
                scanned_count = cur.fetchone()[0]

                fp = self.get_last_scan_fingerprint(project_id)

                return ProjectBrainState(
                    project_identity=ident,
                    schema_version=CURRENT_SCHEMA_VERSION,
                    total_records=total,
                    active_records=active,
                    stale_records=stale,
                    scanned_files_count=scanned_count,
                    total_files_known=scanned_count,
                    last_scan_fingerprint=fp,
                )

    # -------------------------------------------------------------------------
    # Internal Helpers
    # -------------------------------------------------------------------------

    def _row_to_record(self, row: sqlite3.Row) -> KnowledgeRecord:
        return KnowledgeRecord(
            record_id=row["record_id"],
            project_id=row["project_id"],
            record_type=KnowledgeRecordType(row["record_type"]),
            file_path=row["file_path"],
            symbol_name=row["symbol_name"],
            summary=row["summary"],
            details=json.loads(row["details_json"]),
            dependencies=json.loads(row["dependencies_json"]),
            related_symbols=json.loads(row["related_symbols_json"]),
            evidence_ids=json.loads(row["evidence_ids_json"]),
            epistemic_status=EpistemicStatus(row["epistemic_status"]),
            confidence=row["confidence"],
            repo_fingerprint=row["repo_fingerprint"],
            source_chunk=row["source_chunk"],
            version=row["version"],
            active_version=bool(row["active_version"]),
            superseded_by=row["superseded_by"],
            supersedes=row["supersedes"],
            created_at_utc=row["created_at_utc"],
            updated_at_utc=row["updated_at_utc"],
        )
