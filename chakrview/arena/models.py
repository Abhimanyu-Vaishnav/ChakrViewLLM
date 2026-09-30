"""
ChakrView Project Arena: Data Models & Contracts.

Defines strongly typed specifications and records for isolated project execution:
- PathTraversalError, WorkspaceQuotaExceededError
- ProjectSpecification: Requirements, interfaces, entrypoints, and test specs.
- SourceFile: Path, contents, sha256 checksum, role.
- ProjectManifest: Comprehensive project file manifest with dependency metadata.
- TestResult: Outcome of test execution (passed, failed, duration, stdout, stderr).
- PatchDiff: Record of file modifications across iterations.
- IterationRecord: Detailed diagnostic and execution snapshot of a single iteration.
- ExecutionHistory: Sequence of iterations with convergence and regression tracking.
- ArenaExecutionResult: Full execution summary with failure classification.
- EvaluationMetrics: Aggregated score record (syntax validity, pass rate, latency).
"""

from __future__ import annotations

import difflib
import hashlib
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


class PathTraversalError(PermissionError):
    """Raised when a file path attempts to escape the isolated workspace boundary."""
    pass


class WorkspaceQuotaExceededError(RuntimeError):
    """Raised when workspace file count or disk byte limit is exceeded."""
    pass


class FailureCategory(str, Enum):
    """Classification of test / execution failure modes."""
    SUCCESS = "success"
    SYNTAX_ERROR = "syntax_error"
    IMPORT_ERROR = "import_error"
    ASSERTION_FAILURE = "assertion_failure"
    RUNTIME_ERROR = "runtime_error"
    TIMEOUT = "timeout"
    WORKSPACE_ERROR = "workspace_error"


class FileRole(str, Enum):
    """Role of a file in an Arena project."""
    SOURCE = "source"
    TEST = "test"
    INIT = "init"
    CONFIG = "config"
    DOCS = "docs"


@dataclass
class SourceFile:
    """Represents a single file inside an Arena project."""
    path: str
    content: str
    role: FileRole = FileRole.SOURCE
    sha256: str = field(default="")

    def __post_init__(self) -> None:
        if not self.sha256:
            self.sha256 = hashlib.sha256(self.content.encode("utf-8")).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "content": self.content,
            "role": self.role.value,
            "sha256": self.sha256,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SourceFile:
        return cls(
            path=data["path"],
            content=data["content"],
            role=FileRole(data.get("role", "source")),
            sha256=data.get("sha256", ""),
        )


@dataclass
class ProjectSpecification:
    """Specification describing a project to be constructed or tested."""
    project_id: str
    project_name: str
    description: str
    language: str = "python"
    version: str = "0.1.0"
    license: str = "MIT"
    dependencies: List[str] = field(default_factory=list)
    entrypoint: str = "core.py"
    test_framework: str = "pytest"
    timeout_seconds: float = 5.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "project_id": self.project_id,
            "project_name": self.project_name,
            "description": self.description,
            "language": self.language,
            "version": self.version,
            "license": self.license,
            "dependencies": self.dependencies,
            "entrypoint": self.entrypoint,
            "test_framework": self.test_framework,
            "timeout_seconds": self.timeout_seconds,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ProjectSpecification:
        return cls(
            project_id=data["project_id"],
            project_name=data["project_name"],
            description=data["description"],
            language=data.get("language", "python"),
            version=data.get("version", "0.1.0"),
            license=data.get("license", "MIT"),
            dependencies=data.get("dependencies", []),
            entrypoint=data.get("entrypoint", "core.py"),
            test_framework=data.get("test_framework", "pytest"),
            timeout_seconds=float(data.get("timeout_seconds", 5.0)),
        )


@dataclass
class ProjectManifest:
    """Manifest of an entire project repository."""
    specification: ProjectSpecification
    files: List[SourceFile] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_source_files(self) -> List[SourceFile]:
        return [f for f in self.files if f.role in (FileRole.SOURCE, FileRole.INIT)]

    def get_test_files(self) -> List[SourceFile]:
        return [f for f in self.files if f.role == FileRole.TEST]

    def get_file(self, path: str) -> Optional[SourceFile]:
        for f in self.files:
            if f.path == path:
                return f
        return None

    def upsert_file(self, sfile: SourceFile) -> None:
        for idx, f in enumerate(self.files):
            if f.path == sfile.path:
                self.files[idx] = sfile
                return
        self.files.append(sfile)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "specification": self.specification.to_dict(),
            "files": [f.to_dict() for f in self.files],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ProjectManifest:
        spec = ProjectSpecification.from_dict(data["specification"])
        files = [SourceFile.from_dict(f) for f in data.get("files", [])]
        return cls(specification=spec, files=files, metadata=data.get("metadata", {}))


@dataclass
class TestResult:
    """Outcome of test execution."""
    __test__ = False
    passed: int = 0
    failed: int = 0
    errors: int = 0
    skipped: int = 0
    duration_seconds: float = 0.0
    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0

    @property
    def total_tests(self) -> int:
        return self.passed + self.failed + self.errors

    @property
    def pass_rate(self) -> float:
        if self.total_tests == 0:
            return 0.0
        return self.passed / self.total_tests

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "failed": self.failed,
            "errors": self.errors,
            "skipped": self.skipped,
            "duration_seconds": self.duration_seconds,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "exit_code": self.exit_code,
            "total_tests": self.total_tests,
            "pass_rate": round(self.pass_rate, 4),
        }


@dataclass
class PatchDiff:
    """Record of a code modification applied to a file."""
    path: str
    old_content: str
    new_content: str
    diff_text: str = field(default="")
    timestamp_utc: str = field(default="")

    def __post_init__(self) -> None:
        if not self.diff_text:
            old_lines = self.old_content.splitlines(keepends=True)
            new_lines = self.new_content.splitlines(keepends=True)
            diff = difflib.unified_diff(old_lines, new_lines, fromfile=f"a/{self.path}", tofile=f"b/{self.path}")
            self.diff_text = "".join(diff)
        if not self.timestamp_utc:
            self.timestamp_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "diff_text": self.diff_text,
            "timestamp_utc": self.timestamp_utc,
        }


@dataclass
class IterationRecord:
    """Diagnostic and execution snapshot of a single iteration loop."""
    iteration: int
    files_modified: List[str]
    test_result: TestResult
    failure_category: FailureCategory
    diagnosis: Optional[str] = None
    patch_diffs: List[PatchDiff] = field(default_factory=list)
    duration_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "iteration": self.iteration,
            "files_modified": self.files_modified,
            "test_result": self.test_result.to_dict(),
            "failure_category": self.failure_category.value,
            "diagnosis": self.diagnosis,
            "patch_diffs": [p.to_dict() for p in self.patch_diffs],
            "duration_seconds": round(self.duration_seconds, 4),
        }


@dataclass
class ExecutionHistory:
    """Full execution trajectory of a project across multiple iterations."""
    project_id: str
    iterations: List[IterationRecord] = field(default_factory=list)
    converged: bool = False
    final_pass_rate: float = 0.0

    def add_iteration(self, record: IterationRecord) -> None:
        self.iterations.append(record)
        self.final_pass_rate = record.test_result.pass_rate
        if record.failure_category == FailureCategory.SUCCESS and record.test_result.failed == 0 and record.test_result.passed > 0:
            self.converged = True

    @property
    def total_iterations(self) -> int:
        return len(self.iterations)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "project_id": self.project_id,
            "total_iterations": self.total_iterations,
            "converged": self.converged,
            "final_pass_rate": round(self.final_pass_rate, 4),
            "iterations": [it.to_dict() for it in self.iterations],
        }


@dataclass
class ArenaExecutionResult:
    """Aggregate execution result from an isolated Arena run."""
    project_id: str
    failure_category: FailureCategory
    test_result: TestResult
    syntax_valid: bool = True
    syntax_error_message: Optional[str] = None
    execution_time_seconds: float = 0.0
    workspace_path: Optional[str] = None
    artifacts: Dict[str, Any] = field(default_factory=dict)
    history: Optional[ExecutionHistory] = None

    @property
    def is_success(self) -> bool:
        return (
            self.failure_category == FailureCategory.SUCCESS
            and self.syntax_valid
            and self.test_result.failed == 0
            and self.test_result.errors == 0
            and self.test_result.passed > 0
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "project_id": self.project_id,
            "failure_category": self.failure_category.value,
            "is_success": self.is_success,
            "syntax_valid": self.syntax_valid,
            "syntax_error_message": self.syntax_error_message,
            "test_result": self.test_result.to_dict(),
            "execution_time_seconds": round(self.execution_time_seconds, 4),
            "workspace_path": self.workspace_path,
            "artifacts": self.artifacts,
            "history": self.history.to_dict() if self.history else None,
        }


@dataclass
class EvaluationMetrics:
    """Aggregated evaluation metrics for Model, Engine, and Arena."""
    model_loss: Optional[float] = None
    model_perplexity: Optional[float] = None
    syntax_validity_rate: float = 0.0
    functional_pass_rate: float = 0.0
    ttft_ms: float = 0.0
    throughput_tokens_per_sec: float = 0.0
    repetition_ratio: float = 0.0
    isolation_maintained: bool = True
    timeout_enforced: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_loss": round(self.model_loss, 4) if self.model_loss is not None else None,
            "model_perplexity": round(self.model_perplexity, 2) if self.model_perplexity is not None else None,
            "syntax_validity_rate": round(self.syntax_validity_rate, 4),
            "functional_pass_rate": round(self.functional_pass_rate, 4),
            "ttft_ms": round(self.ttft_ms, 2),
            "throughput_tokens_per_sec": round(self.throughput_tokens_per_sec, 2),
            "repetition_ratio": round(self.repetition_ratio, 4),
            "isolation_maintained": self.isolation_maintained,
            "timeout_enforced": self.timeout_enforced,
        }
