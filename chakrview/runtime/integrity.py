"""
Software Integrity, Health Checks & Rollback System for ChakrView (Step 9).

Deterministic defense-in-depth:
- Cryptographic artifact verification (SHA-256)
- Pre-execution health and NaN/Inf sanity checks
- Automated quarantine of corrupt/suspicious candidates
- Atomic state rollback to last known-good configuration
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Dict, List, Optional, Any, Callable
import torch
import torch.nn as nn


class HealthStatus(str, Enum):
    """System health classification."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"


@dataclass
class HealthCheckResult:
    """Result of an individual integrity or sanity check."""
    name: str
    passed: bool
    status: HealthStatus
    message: str
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["status"] = self.status.value
        return data


@dataclass
class HealthReport:
    """Aggregated health report across multiple diagnostic checks."""
    overall_status: HealthStatus
    passed: bool
    checks: List[HealthCheckResult]
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_status": self.overall_status.value,
            "passed": self.passed,
            "checks": [c.to_dict() for c in self.checks],
            "timestamp": self.timestamp,
        }


class ArtifactVerifier:
    """
    Cryptographic verification utilities for files, configs, and model state dictionaries.
    """

    @staticmethod
    def compute_file_sha256(path: Path | str, chunk_size: int = 65536) -> str:
        """Compute SHA-256 hex digest of a file on disk."""
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"File not found for hash computation: {path}")
        h = hashlib.sha256()
        with open(path, "rb") as f:
            while chunk := f.read(chunk_size):
                h.update(chunk)
        return h.hexdigest()

    @staticmethod
    def verify_file_sha256(path: Path | str, expected_hash: str) -> bool:
        """Verify whether a file matches an expected SHA-256 digest."""
        computed = ArtifactVerifier.compute_file_sha256(path)
        return computed.lower() == expected_hash.lower()

    @staticmethod
    def compute_json_sha256(path: Path | str, indent: int = 2) -> str:
        """Compute platform-independent SHA-256 hex digest of JSON file with normalized LF newlines."""
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        normalized = json.dumps(data, indent=indent)
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    @staticmethod
    def verify_json_sha256(path: Path | str, expected_hash: str, indent: int = 2) -> bool:
        """Verify normalized JSON content matches expected SHA-256 digest."""
        computed = ArtifactVerifier.compute_json_sha256(path, indent=indent)
        return computed.lower() == expected_hash.lower()

    @staticmethod
    def verify_model_weights_health(model: nn.Module) -> HealthCheckResult:
        """
        Verify that all parameters in a PyTorch model are finite (0 NaNs, 0 Infs).
        """
        nan_count = 0
        inf_count = 0
        param_count = 0

        for name, param in model.named_parameters():
            param_count += 1
            if torch.isnan(param).any():
                nan_count += 1
            if torch.isinf(param).any():
                inf_count += 1

        if nan_count > 0 or inf_count > 0:
            return HealthCheckResult(
                name="model_weights_health",
                passed=False,
                status=HealthStatus.CRITICAL,
                message=f"Model weights unhealthy: {nan_count} tensors with NaN, {inf_count} with Inf.",
                details={"nan_tensors": nan_count, "inf_tensors": inf_count, "total_tensors": param_count},
            )

        return HealthCheckResult(
            name="model_weights_health",
            passed=True,
            status=HealthStatus.HEALTHY,
            message="All model weight parameters are finite and healthy.",
            details={"total_tensors": param_count},
        )


@dataclass
class QuarantineRecord:
    """Audit log entry for an isolated artifact."""
    artifact_id: str
    original_path: str
    quarantine_path: str
    reason: str
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class QuarantineManager:
    """
    Safely moves suspicious, corrupted, or failed update artifacts to an isolated quarantine area.
    """

    def __init__(self, quarantine_dir: Path | str) -> None:
        self.quarantine_dir = Path(quarantine_dir)
        self.quarantine_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_file = self.quarantine_dir / "quarantine_manifest.json"
        self._records: List[QuarantineRecord] = []
        self._load_manifest()

    def quarantine_artifact(
        self,
        artifact_path: Path | str,
        reason: str,
        artifact_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> QuarantineRecord:
        """Move an artifact to the quarantine directory and record metadata."""
        src = Path(artifact_path)
        if not src.exists():
            raise FileNotFoundError(f"Cannot quarantine non-existent file: {src}")

        aid = artifact_id or src.stem
        timestamp_slug = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        dest_name = f"{aid}_{timestamp_slug}{src.suffix}"
        dest_path = self.quarantine_dir / dest_name

        shutil.move(str(src), str(dest_path))

        record = QuarantineRecord(
            artifact_id=aid,
            original_path=str(src),
            quarantine_path=str(dest_path),
            reason=reason,
            metadata=metadata or {},
        )
        self._records.append(record)
        self._save_manifest()
        return record

    def list_records(self) -> List[QuarantineRecord]:
        return list(self._records)

    def _load_manifest(self) -> None:
        if self.manifest_file.is_file():
            try:
                with open(self.manifest_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self._records = [QuarantineRecord(**r) for r in data]
            except Exception:
                self._records = []

    def _save_manifest(self) -> None:
        with open(self.manifest_file, "w", encoding="utf-8") as f:
            json.dump([r.to_dict() for r in self._records], f, indent=2)


class RollbackManager:
    """
    Manages atomic pointer switching and automatic restoration to known-good state.
    """

    def __init__(self, state_dir: Path | str) -> None:
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.pointer_file = self.state_dir / "active_version.json"
        self.backup_pointer_file = self.state_dir / "known_good_version.json"

    def set_active_version(
        self, version_id: str, version_hash: str, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Atomically set the active version pointer."""
        payload = {
            "version_id": version_id,
            "version_hash": version_hash,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "metadata": metadata or {},
        }
        tmp_file = self.state_dir / "active_version.json.tmp"
        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        os.replace(str(tmp_file), str(self.pointer_file))

    def mark_as_known_good(self) -> None:
        """Snapshot current active version as the known-good rollback target."""
        if not self.pointer_file.is_file():
            raise RuntimeError("No active version pointer to mark as known-good.")
        shutil.copy(str(self.pointer_file), str(self.backup_pointer_file))

    def rollback(self) -> Dict[str, Any]:
        """Restore active version from known-good backup. Returns restored metadata."""
        if not self.backup_pointer_file.is_file():
            raise RuntimeError("No known-good backup exists to rollback to.")
        shutil.copy(str(self.backup_pointer_file), str(self.pointer_file))
        with open(self.pointer_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def get_active_version(self) -> Optional[Dict[str, Any]]:
        if not self.pointer_file.is_file():
            return None
        with open(self.pointer_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def get_known_good_version(self) -> Optional[Dict[str, Any]]:
        if not self.backup_pointer_file.is_file():
            return None
        with open(self.backup_pointer_file, "r", encoding="utf-8") as f:
            return json.load(f)
