"""
ChakrView Step 78: Persistent Project Brain (PPB) Data Models & Contracts.

Defines:
- EpistemicStatus: Grounded truth tracking (FACT, INFERRED, HYPOTHESIS, STALE, UNKNOWN, INSUFFICIENT, CONTESTED, REVERIFIED).
- KnowledgeRecordType: Taxonomy of project knowledge (MODULE, SYMBOL, DEPENDENCY, ARCHITECTURE, CONSTRAINT, TEST_RELATION, TASK_HISTORY, OBSERVATION).
- KnowledgeRecord: Strongly typed, provenance-traceable, versioned unit of project memory.
- ProjectIdentity: Project identifier, path, language, framework, creation and last-accessed metadata.
- ProjectBrainState: High-level state summary and statistics of the persistent project brain.
- PPBStorageSchemaError: Exception raised when schema migration or format mismatch occurs.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Set


CURRENT_SCHEMA_VERSION = 1


class EpistemicStatus(str, Enum):
    """
    Epistemic classification for stored project knowledge.
    Enforces the Step 71-76 principle: Inferences never become FACT silently;
    uninspected is UNKNOWN; inadequate evidence is INSUFFICIENT; conflicts are CONTESTED;
    outdated knowledge is STALE.
    """
    FACT = "FACT"                     # Directly observed via AST inspection, dependency scan, or passing test
    INFERRED = "INFERRED"             # Derived logically from facts, pending explicit re-verification
    HYPOTHESIS = "HYPOTHESIS"         # Tentative architectural or structural proposition
    STALE = "STALE"                   # Underlying file, symbol, or dependency modified; needs refresh
    UNKNOWN = "UNKNOWN"               # Uninspected or unverified entity
    INSUFFICIENT = "INSUFFICIENT"     # Partial or inconclusive evidence
    CONTESTED = "CONTESTED"           # Conflicting evidence across scans or changes
    REVERIFIED = "REVERIFIED"         # Stale entity explicitly re-analyzed and confirmed valid


class KnowledgeRecordType(str, Enum):
    """Taxonomy of structured knowledge captured in the Persistent Project Brain."""
    MODULE = "MODULE"
    SYMBOL = "SYMBOL"
    DEPENDENCY = "DEPENDENCY"
    ARCHITECTURE = "ARCHITECTURE"
    CONSTRAINT = "CONSTRAINT"
    TEST_RELATION = "TEST_RELATION"
    TASK_HISTORY = "TASK_HISTORY"
    OBSERVATION = "OBSERVATION"


@dataclass(frozen=True)
class ProjectIdentity:
    """Canonical identity of a project indexed by the Persistent Project Brain."""
    project_id: str
    project_root: str
    language: str = "python"
    framework: str = "standard_library"
    created_at_utc: str = field(
        default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    )
    last_accessed_utc: str = field(
        default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ProjectIdentity:
        return cls(**data)


@dataclass
class KnowledgeRecord:
    """
    Fundamental unit of persistent project intelligence.
    Must always maintain strict provenance, epistemic status, and versioning.
    """
    record_id: str
    project_id: str
    record_type: KnowledgeRecordType
    file_path: str
    symbol_name: Optional[str] = None
    summary: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    related_symbols: List[str] = field(default_factory=list)
    evidence_ids: List[str] = field(default_factory=list)
    epistemic_status: EpistemicStatus = EpistemicStatus.FACT
    confidence: float = 1.0
    repo_fingerprint: str = ""
    source_chunk: str = ""
    version: int = 1
    active_version: bool = True
    superseded_by: Optional[str] = None
    supersedes: Optional[str] = None
    created_at_utc: str = field(
        default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    )
    updated_at_utc: str = field(
        default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    )

    def compute_content_hash(self) -> str:
        """Deterministic digest of this knowledge record's core factual content."""
        hasher = hashlib.sha256()
        hasher.update(self.project_id.encode("utf-8"))
        hasher.update(self.record_type.value.encode("utf-8"))
        hasher.update(self.file_path.encode("utf-8"))
        hasher.update((self.symbol_name or "").encode("utf-8"))
        hasher.update(self.summary.encode("utf-8"))
        hasher.update(json.dumps(self.details, sort_keys=True).encode("utf-8"))
        hasher.update(",".join(sorted(self.dependencies)).encode("utf-8"))
        hasher.update(",".join(sorted(self.related_symbols)).encode("utf-8"))
        return hasher.hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "record_id": self.record_id,
            "project_id": self.project_id,
            "record_type": self.record_type.value,
            "file_path": self.file_path,
            "symbol_name": self.symbol_name,
            "summary": self.summary,
            "details": self.details,
            "dependencies": list(self.dependencies),
            "related_symbols": list(self.related_symbols),
            "evidence_ids": list(self.evidence_ids),
            "epistemic_status": self.epistemic_status.value,
            "confidence": round(self.confidence, 4),
            "repo_fingerprint": self.repo_fingerprint,
            "source_chunk": self.source_chunk,
            "version": self.version,
            "active_version": self.active_version,
            "superseded_by": self.superseded_by,
            "supersedes": self.supersedes,
            "created_at_utc": self.created_at_utc,
            "updated_at_utc": self.updated_at_utc,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> KnowledgeRecord:
        d = dict(data)
        d["record_type"] = KnowledgeRecordType(d["record_type"])
        d["epistemic_status"] = EpistemicStatus(d["epistemic_status"])
        return cls(**d)


@dataclass
class ProjectBrainState:
    """Summary snapshot of the persistent project brain."""
    project_identity: ProjectIdentity
    schema_version: int = CURRENT_SCHEMA_VERSION
    total_records: int = 0
    active_records: int = 0
    stale_records: int = 0
    scanned_files_count: int = 0
    total_files_known: int = 0
    last_scan_fingerprint: str = ""
    last_updated_utc: str = field(
        default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "project_identity": self.project_identity.to_dict(),
            "schema_version": self.schema_version,
            "total_records": self.total_records,
            "active_records": self.active_records,
            "stale_records": self.stale_records,
            "scanned_files_count": self.scanned_files_count,
            "total_files_known": self.total_files_known,
            "last_scan_fingerprint": self.last_scan_fingerprint,
            "last_updated_utc": self.last_updated_utc,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ProjectBrainState:
        d = dict(data)
        d["project_identity"] = ProjectIdentity.from_dict(d["project_identity"])
        return cls(**d)


class PPBStorageSchemaError(RuntimeError):
    """Raised when persistent storage schema is incompatible or corrupted."""
    pass
