"""
Governed Sandboxed Python Execution Tool for ChakrView.
Location: chakrview/runtime/sandboxed_python_tool.py

Implements:
- Restricted execution of bounded Python scripts in isolated ephemeral directories.
- AST pre-execution security filter rejecting network, system calls, and unsafe modules.
- Subprocess timeouts (hard process termination).
- Standard output and error size clamping (64 KB ceiling).
- Environment variable sanitization (scrubbing secrets and credentials).
- Fail-closed default rejecting unvetted arbitrary code.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass, asdict
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional, Set

from chakrview.runtime.tools import Tool, ToolResult


@dataclass
class SandboxExecutionResult:
    execution_id: str
    status: str  # SUCCESS, TIMEOUT, SECURITY_VIOLATION, RUNTIME_ERROR, OVERSIZED_OUTPUT
    exit_code: Optional[int]
    stdout: str
    stderr: str
    timed_out: bool
    output_truncated: bool
    execution_time_ms: float
    workspace_cleaned: bool
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SandboxedPythonTool(Tool):
    """
    Narrow sandboxed Python execution tool enforcing pre-execution static AST checks,
    ephemeral filesystem isolation, environment variable scrub, and hard timeouts.
    """

    MAX_CODE_LENGTH = 8192
    MAX_TIMEOUT_SECONDS = 2.0
    MAX_OUTPUT_BYTES = 65536  # 64 KB

    # Forbidden modules that represent network, process control, or host inspection vectors
    FORBIDDEN_MODULES: Set[str] = {
        "socket", "http", "urllib", "requests", "httpx", "aiohttp",
        "subprocess", "multiprocessing", "threading", "os", "sys", "shutil",
        "ctypes", "winreg", "win32api", "win32con", "win32file",
        "socketserver", "asyncio", "signal", "posix", "nt", "builtins"
    }

    def __init__(self, allow_arbitrary_code: bool = False, python_executable: Optional[str] = None) -> None:
        self.allow_arbitrary_code = allow_arbitrary_code
        self.python_executable = python_executable or sys.executable

    @property
    def tool_id(self) -> str:
        return "sandboxed_python"

    @property
    def name(self) -> str:
        return "Governed Sandboxed Python Runner"

    @property
    def description(self) -> str:
        return "Executes bounded Python tasks in an ephemeral restricted workspace with static AST filtering."

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "code": {
                "type": "string",
                "required": True,
                "description": "Python source code to execute.",
            },
            "timeout": {
                "type": "number",
                "required": False,
                "description": "Timeout ceiling in seconds (max 2.0s).",
            }
        }

    def _inspect_ast_security(self, code_str: str) -> Optional[str]:
        """
        Statically inspects code AST. Rejects forbidden imports and low-level system accesses.
        """
        try:
            tree = ast.parse(code_str)
        except SyntaxError as e:
            return f"Syntax error during AST inspection: {str(e)}"

        for node in ast.walk(tree):
            # Check import statements
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root_mod = alias.name.split(".")[0]
                    if root_mod in self.FORBIDDEN_MODULES:
                        return f"Security violation: Import of forbidden module '{alias.name}' is blocked."

            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    root_mod = node.module.split(".")[0]
                    if root_mod in self.FORBIDDEN_MODULES:
                        return f"Security violation: Import from forbidden module '{node.module}' is blocked."

            # Check for dynamic execution or file opening
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in ("eval", "exec", "open", "__import__", "compile"):
                        return f"Security violation: Function '{node.func.id}' is blocked in sandbox."

        return None

    def execute_script(
        self,
        code: str,
        timeout: float = 2.0,
        execution_id: str = "sbx_run",
    ) -> SandboxExecutionResult:
        t0 = time.perf_counter()
        code_clean = code.strip()

        # 1. Check code length limits
        if len(code_clean) > self.MAX_CODE_LENGTH:
            elapsed = (time.perf_counter() - t0) * 1000.0
            return SandboxExecutionResult(
                execution_id=execution_id,
                status="SECURITY_VIOLATION",
                exit_code=-1,
                stdout="",
                stderr="",
                timed_out=False,
                output_truncated=False,
                execution_time_ms=elapsed,
                workspace_cleaned=True,
                error=f"Code length ({len(code_clean)}) exceeds maximum allowed ({self.MAX_CODE_LENGTH} characters).",
            )

        # 2. Enforce fail-closed arbitrary code policy
        if not self.allow_arbitrary_code:
            # Enforce static AST filter
            violation = self._inspect_ast_security(code_clean)
            if violation:
                elapsed = (time.perf_counter() - t0) * 1000.0
                return SandboxExecutionResult(
                    execution_id=execution_id,
                    status="SECURITY_VIOLATION",
                    exit_code=-1,
                    stdout="",
                    stderr="",
                    timed_out=False,
                    output_truncated=False,
                    execution_time_ms=elapsed,
                    workspace_cleaned=True,
                    error=violation,
                )

        timeout_sec = min(max(0.1, timeout), self.MAX_TIMEOUT_SECONDS)

        # 3. Create isolated ephemeral directory
        workspace_cleaned = False
        with tempfile.TemporaryDirectory(prefix="chakr_sandbox_") as tmpdir:
            script_path = Path(tmpdir) / "sandbox_main.py"
            script_path.write_text(code_clean, encoding="utf-8")

            # 4. Scrub environment variables
            sanitized_env = {
                "SYSTEMROOT": os.environ.get("SYSTEMROOT", "C:\\Windows"),
                "PATH": os.environ.get("PATH", ""),
                "PYTHONIOENCODING": "utf-8",
                "PYTHONUNBUFFERED": "1",
            }

            timed_out = False
            stdout_str = ""
            stderr_str = ""
            truncated = False
            exit_code = None
            error_msg = None
            status = "SUCCESS"

            try:
                # Windows flags to ensure process group isolation
                flags = 0
                if sys.platform == "win32":
                    flags = subprocess.CREATE_NEW_PROCESS_GROUP

                proc = subprocess.Popen(
                    [self.python_executable, "-I", str(script_path.name)],
                    cwd=tmpdir,
                    env=sanitized_env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    creationflags=flags,
                )

                try:
                    stdout_bytes, stderr_bytes = proc.communicate(timeout=timeout_sec)
                    exit_code = proc.returncode

                    if len(stdout_bytes) > self.MAX_OUTPUT_BYTES:
                        stdout_bytes = stdout_bytes[:self.MAX_OUTPUT_BYTES]
                        truncated = True
                        status = "OVERSIZED_OUTPUT"

                    if len(stderr_bytes) > self.MAX_OUTPUT_BYTES:
                        stderr_bytes = stderr_bytes[:self.MAX_OUTPUT_BYTES]
                        truncated = True

                    stdout_str = stdout_bytes.decode("utf-8", errors="replace")
                    stderr_str = stderr_bytes.decode("utf-8", errors="replace")

                    if exit_code != 0:
                        status = "RUNTIME_ERROR"
                        error_msg = f"Script exited with non-zero exit code {exit_code}."

                except subprocess.TimeoutExpired:
                    proc.kill()
                    timed_out = True
                    status = "TIMEOUT"
                    error_msg = f"Execution timed out after {timeout_sec} seconds."
                    try:
                        stdout_bytes, stderr_bytes = proc.communicate(timeout=0.5)
                        stdout_str = stdout_bytes.decode("utf-8", errors="replace")
                        stderr_str = stderr_bytes.decode("utf-8", errors="replace")
                    except Exception:
                        pass

            except Exception as e:
                status = "RUNTIME_ERROR"
                error_msg = f"Subprocess spawning error: {type(e).__name__}: {str(e)}"

            workspace_cleaned = True
            elapsed = (time.perf_counter() - t0) * 1000.0

            return SandboxExecutionResult(
                execution_id=execution_id,
                status=status,
                exit_code=exit_code,
                stdout=stdout_str,
                stderr=stderr_str,
                timed_out=timed_out,
                output_truncated=truncated,
                execution_time_ms=elapsed,
                workspace_cleaned=workspace_cleaned,
                error=error_msg,
            )

    def execute(self, **kwargs: Any) -> ToolResult:
        code = kwargs.get("code", "")
        timeout = float(kwargs.get("timeout", 2.0))
        res = self.execute_script(code=code, timeout=timeout)
        return ToolResult(
            tool_id=self.tool_id,
            success=(res.status == "SUCCESS"),
            output={"stdout": res.stdout, "exit_code": res.exit_code},
            error=res.error or (res.stderr if res.stderr else None),
            execution_time_ms=res.execution_time_ms,
        )
