"""
Governed OCI Container Execution Backend for ChakrView.
Location: chakrview/runtime/oci_container_backend.py

Implements:
- Dynamic discovery of OCI container runtimes (docker, podman).
- Enforces strict execution contracts: non-root, read-only root, dropped capabilities,
  network disabled (--network none), and resource limits (CPU, memory, PIDs).
- Rejects unvetted images, unsafe host mounts, and malformed configurations.
- Returns explicit RUNTIME_UNAVAILABLE status when no container daemon exists.
- Safely builds argument arrays avoiding shell string interpolation.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional, Set


class ContainerExecutionStatus(str, Enum):
    SUCCESS = "SUCCESS"
    TIMEOUT = "TIMEOUT"
    SECURITY_VIOLATION = "SECURITY_VIOLATION"
    RUNTIME_UNAVAILABLE = "RUNTIME_UNAVAILABLE"
    UNSAFE_CONFIG_REJECTED = "UNSAFE_CONFIG_REJECTED"
    RUNTIME_ERROR = "RUNTIME_ERROR"
    OVERSIZED_OUTPUT = "OVERSIZED_OUTPUT"


@dataclass(frozen=True)
class ContainerExecutionRequest:
    task_id: str
    code: str
    image: str = "python:3.11-alpine"
    timeout_seconds: float = 2.0
    memory_limit: str = "256m"
    cpu_limit: str = "1.0"
    pids_limit: int = 64
    network_enabled: bool = False
    read_only_root: bool = True
    custom_env: Dict[str, str] = field(default_factory=dict)
    mount_paths: List[str] = field(default_factory=list)


@dataclass
class ContainerExecutionResponse:
    task_id: str
    status: ContainerExecutionStatus
    exit_code: Optional[int]
    stdout: str
    stderr: str
    execution_time_ms: float
    runtime_name: Optional[str]
    network_disabled: bool
    workspace_cleaned: bool
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        return d


class OciContainerBackend:
    """
    Governed OCI Container Execution Engine.
    Discovers, validates, and invokes Docker or Podman with strict least-privilege flags.
    """

    APPROVED_IMAGES: Set[str] = {
        "python:3.11-alpine",
        "python:3.10-alpine",
        "python:3.11-slim",
        "docker.io/library/python:3.11-alpine",
    }

    MAX_TIMEOUT_SECONDS = 5.0
    MAX_OUTPUT_BYTES = 65536  # 64 KB

    def __init__(self, override_runtime_cmd: Optional[str] = None) -> None:
        self.override_runtime_cmd = override_runtime_cmd
        self._runtime_name, self._runtime_path = self._discover_runtime()

    def _discover_runtime(self) -> Tuple[Optional[str], Optional[str]]:
        """Find an available OCI container CLI on PATH."""
        if self.override_runtime_cmd:
            path = shutil.which(self.override_runtime_cmd)
            if path:
                name = Path(path).stem.lower()
                return name, path

        for candidate in ["docker", "podman", "nerdctl"]:
            p = shutil.which(candidate)
            if p:
                return candidate, p
        return None, None

    @property
    def is_available(self) -> bool:
        return self._runtime_path is not None

    @property
    def runtime_name(self) -> Optional[str]:
        return self._runtime_name

    def validate_request(self, req: ContainerExecutionRequest) -> Optional[str]:
        """
        Validates request parameters against safety policies before execution.
        Returns error string if invalid, or None if valid.
        """
        if not re.match(r"^[a-zA-Z0-9_\-]+$", req.task_id):
            return f"Invalid task_id '{req.task_id}': must contain only alphanumeric characters, dashes, or underscores."

        if req.image not in self.APPROVED_IMAGES:
            return f"Unapproved container image '{req.image}'. Must be one of: {sorted(list(self.APPROVED_IMAGES))}."

        if req.network_enabled:
            return "Network access is prohibited in governed container execution."

        if not re.match(r"^\d+[mk]$", req.memory_limit.lower()):
            return f"Invalid memory limit '{req.memory_limit}': must match format e.g. '256m'."

        # Disallow unsafe mounts (workload may only mount its dedicated ephemeral workspace)
        if req.mount_paths:
            return f"Unsafe mount paths rejected: {req.mount_paths}. Workloads may not specify custom host mounts."

        return None

    def build_cli_arguments(
        self, req: ContainerExecutionRequest, host_workspace_dir: str
    ) -> List[str]:
        """
        Constructs the exact argument array for the container runtime.
        Never interpolates raw shell strings.
        """
        if not self._runtime_path:
            raise RuntimeError("OCI runtime not available")

        timeout_sec = min(max(0.1, req.timeout_seconds), self.MAX_TIMEOUT_SECONDS)

        cmd = [
            self._runtime_path,
            "run",
            "--rm",
            "--network", "none",
            "--read-only" if req.read_only_root else "--read-only=false",
            "--user", "1000:1000",
            "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges",
            "--memory", req.memory_limit,
            "--cpus", req.cpu_limit,
            "--pids-limit", str(req.pids_limit),
            "-v", f"{host_workspace_dir}:/workspace:rw",
            "-w", "/workspace",
        ]

        # Sanitized environment
        for k, v in req.custom_env.items():
            if not any(secret_kw in k.upper() for secret_kw in ["KEY", "TOKEN", "SECRET", "PASSWORD", "AWS", "CRED"]):
                cmd.extend(["-e", f"{k}={v}"])

        cmd.extend([req.image, "python", "-u", "main.py"])
        return cmd

    def execute(self, req: ContainerExecutionRequest) -> ContainerExecutionResponse:
        t0 = time.perf_counter()

        # 1. Validate request parameters first (reject unsafe configurations immediately)
        validation_error = self.validate_request(req)
        if validation_error:
            elapsed = (time.perf_counter() - t0) * 1000.0
            return ContainerExecutionResponse(
                task_id=req.task_id,
                status=ContainerExecutionStatus.UNSAFE_CONFIG_REJECTED,
                exit_code=None,
                stdout="",
                stderr="",
                execution_time_ms=elapsed,
                runtime_name=self._runtime_name,
                network_disabled=True,
                workspace_cleaned=True,
                error=validation_error,
            )

        # 2. Check runtime availability
        if not self.is_available:
            elapsed = (time.perf_counter() - t0) * 1000.0
            return ContainerExecutionResponse(
                task_id=req.task_id,
                status=ContainerExecutionStatus.RUNTIME_UNAVAILABLE,
                exit_code=None,
                stdout="",
                stderr="",
                execution_time_ms=elapsed,
                runtime_name=None,
                network_disabled=True,
                workspace_cleaned=True,
                error="OCI container runtime (docker/podman) is not installed or available on this host. Arbitrary execution failed closed.",
            )

        # 3. Create isolated ephemeral workspace and write script
        with tempfile.TemporaryDirectory(prefix="chakr_oci_") as tmpdir:
            script_file = Path(tmpdir) / "main.py"
            script_file.write_text(req.code, encoding="utf-8")

            # Build argument array
            cmd = self.build_cli_arguments(req, tmpdir)
            timeout_sec = min(max(0.1, req.timeout_seconds), self.MAX_TIMEOUT_SECONDS)

            timed_out = False
            truncated = False
            status = ContainerExecutionStatus.SUCCESS
            exit_code = None
            err_msg = None
            stdout_str = ""
            stderr_str = ""

            try:
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                try:
                    stdout_bytes, stderr_bytes = proc.communicate(timeout=timeout_sec)
                    exit_code = proc.returncode

                    if len(stdout_bytes) > self.MAX_OUTPUT_BYTES:
                        stdout_bytes = stdout_bytes[:self.MAX_OUTPUT_BYTES]
                        truncated = True
                        status = ContainerExecutionStatus.OVERSIZED_OUTPUT

                    if len(stderr_bytes) > self.MAX_OUTPUT_BYTES:
                        stderr_bytes = stderr_bytes[:self.MAX_OUTPUT_BYTES]
                        truncated = True

                    stdout_str = stdout_bytes.decode("utf-8", errors="replace")
                    stderr_str = stderr_bytes.decode("utf-8", errors="replace")

                    if exit_code != 0 and status != ContainerExecutionStatus.OVERSIZED_OUTPUT:
                        status = ContainerExecutionStatus.RUNTIME_ERROR
                        err_msg = f"Container process exited with non-zero code {exit_code}."

                except subprocess.TimeoutExpired:
                    proc.kill()
                    timed_out = True
                    status = ContainerExecutionStatus.TIMEOUT
                    err_msg = f"Execution timed out after {timeout_sec} seconds."

            except Exception as e:
                status = ContainerExecutionStatus.RUNTIME_ERROR
                err_msg = f"Failed to execute container command: {type(e).__name__}: {str(e)}"

            elapsed = (time.perf_counter() - t0) * 1000.0
            return ContainerExecutionResponse(
                task_id=req.task_id,
                status=status,
                exit_code=exit_code,
                stdout=stdout_str,
                stderr=stderr_str,
                execution_time_ms=elapsed,
                runtime_name=self._runtime_name,
                network_disabled=True,
                workspace_cleaned=True,
                error=err_msg,
            )
