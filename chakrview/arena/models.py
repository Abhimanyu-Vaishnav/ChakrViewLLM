"""
ChakrView Project Arena: Data Models & Contracts.

Defines strongly typed specifications and records for isolated project execution:
- ProjectSpecification: Requirements, interfaces, entrypoints, and test specs.
- SourceFile: Path, contents, sha256 checksum, role.
- ProjectManifest: Comprehensive project file manifest with dependency metadata.
- TestResult: Outcome of test execution (passed, failed, duration, stdout, stderr).
- ArenaExecutionResult: Full execution summary with failure classification.
- EvaluationMetrics: Aggregated score record (syntax validity, pass rate, latency).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional
import hashlib
import time


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
