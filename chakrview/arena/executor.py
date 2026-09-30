"""
ChakrView Project Arena: Sandboxed Test Executor.

Executes tests inside isolated subprocesses with:
- Strict wall-clock timeouts (fail-closed)
- Sanitized environment variables
- Controlled PYTHONPATH (restricted to workspace source dir)
- Capturing stdout, stderr, exit codes, and durations
- Structured parsing of pytest / unittest outputs
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

from chakrview.arena.models import TestResult, FailureCategory
from chakrview.arena.workspace import IsolatedWorkspace


class SandboxedExecutor:
    """
    Executes project tests in an isolated, restricted subprocess.
    """
    def __init__(self, python_executable: Optional[str] = None) -> None:
        self.python_executable = python_executable or sys.executable

    def _build_sanitized_env(self, workspace: IsolatedWorkspace) -> dict[str, str]:
        """
        Build an environment dictionary containing only essential system paths,
        with PYTHONPATH pointing strictly to the workspace source directory.
        """
        env = {}
        # Preserve essential OS variables
        for key in ("PATH", "SYSTEMROOT", "WINDIR", "TEMP", "TMP", "COMSPEC", "HOME", "USERPROFILE"):
            if key in os.environ:
                env[key] = os.environ[key]

        # Restrict PYTHONPATH strictly to workspace source directory
        env["PYTHONPATH"] = str(workspace.source_dir.resolve())
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["PYTHONUNBUFFERED"] = "1"

        return env

    def run_tests(
        self,
        workspace: IsolatedWorkspace,
        timeout_seconds: Optional[float] = None,
    ) -> tuple[TestResult, FailureCategory]:
        """
        Run test files in workspace.test_dir.
        """
        timeout = timeout_seconds or workspace.manifest.specification.timeout_seconds
        test_files = workspace.list_test_files()

        if not test_files:
            return (
                TestResult(
                    passed=0,
                    failed=0,
                    errors=1,
                    duration_seconds=0.0,
                    stdout="",
                    stderr="No test files found in workspace",
                    exit_code=1,
                ),
                FailureCategory.WORKSPACE_ERROR,
            )

        cmd = [
            self.python_executable,
            "-m",
            "pytest",
            str(workspace.test_dir.resolve()),
            "-v",
            "--tb=short",
            "--no-header",
        ]

        env = self._build_sanitized_env(workspace)
        start_time = time.perf_counter()

        try:
            proc = subprocess.run(
                cmd,
                cwd=str(workspace.workspace_dir.resolve()),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout,
            )
            duration = time.perf_counter() - start_time
            stdout = proc.stdout
            stderr = proc.stderr
            exit_code = proc.returncode

            # Save execution log to workspace
            log_path = workspace.logs_dir / "execution.log"
            with open(log_path, "w", encoding="utf-8") as f:
                f.write(f"EXIT CODE: {exit_code}\n")
                f.write(f"DURATION: {duration:.4f}s\n\n--- STDOUT ---\n{stdout}\n\n--- STDERR ---\n{stderr}\n")

            test_res = self._parse_pytest_output(stdout, stderr, exit_code, duration)
            failure_cat = self._classify_failure(stdout, stderr, exit_code)

            return test_res, failure_cat

        except subprocess.TimeoutExpired as exc:
            duration = time.perf_counter() - start_time
            stdout = exc.stdout if isinstance(exc.stdout, str) else (exc.stdout.decode("utf-8", "replace") if exc.stdout else "")
            stderr = exc.stderr if isinstance(exc.stderr, str) else (exc.stderr.decode("utf-8", "replace") if exc.stderr else "")

            log_path = workspace.logs_dir / "execution.log"
            with open(log_path, "w", encoding="utf-8") as f:
                f.write(f"TIMEOUT EXPIRED after {timeout:.2f}s\n\n--- STDOUT ---\n{stdout}\n\n--- STDERR ---\n{stderr}\n")

            return (
                TestResult(
                    passed=0,
                    failed=0,
                    errors=1,
                    duration_seconds=duration,
                    stdout=stdout,
                    stderr=stderr + f"\nExecution timed out after {timeout:.2f} seconds",
                    exit_code=-1,
                ),
                FailureCategory.TIMEOUT,
            )

        except Exception as exc:
            duration = time.perf_counter() - start_time
            return (
                TestResult(
                    passed=0,
                    failed=0,
                    errors=1,
                    duration_seconds=duration,
                    stdout="",
                    stderr=str(exc),
                    exit_code=-2,
                ),
                FailureCategory.RUNTIME_ERROR,
            )

    def _parse_pytest_output(
        self,
        stdout: str,
        stderr: str,
        exit_code: int,
        duration: float,
    ) -> TestResult:
        """Parse pytest stdout to extract passed, failed, and error counts."""
        passed = 0
        failed = 0
        errors = 0
        skipped = 0

        # Match pattern like: "2 passed, 1 failed, 1 error in 0.05s"
        match = re.search(r"(=+ ([\d\w ,]+) in [\d\.]+s? =+)", stdout)
        if match:
            summary_str = match.group(2)
            passed_m = re.search(r"(\d+) passed", summary_str)
            failed_m = re.search(r"(\d+) failed", summary_str)
            errors_m = re.search(r"(\d+) error", summary_str)
            skipped_m = re.search(r"(\d+) skipped", summary_str)

            if passed_m:
                passed = int(passed_m.group(1))
            if failed_m:
                failed = int(failed_m.group(1))
            if errors_m:
                errors = int(errors_m.group(1))
            if skipped_m:
                skipped = int(skipped_m.group(1))
        elif exit_code == 0:
            # Fallback if pytest returned 0 but format differed
            passed = max(1, len(re.findall(r"PASSED", stdout)))
        else:
            # If nonzero and unparsed, count as failure or error
            if "ModuleNotFoundError" in stderr or "ImportError" in stderr or "ModuleNotFoundError" in stdout or "ImportError" in stdout:
                errors = 1
            elif "SyntaxError" in stderr or "SyntaxError" in stdout:
                errors = 1
            else:
                failed = 1

        return TestResult(
            passed=passed,
            failed=failed,
            errors=errors,
            skipped=skipped,
            duration_seconds=duration,
            stdout=stdout,
            stderr=stderr,
            exit_code=exit_code,
        )

    def _classify_failure(
        self,
        stdout: str,
        stderr: str,
        exit_code: int,
    ) -> FailureCategory:
        """Classify exit condition into a FailureCategory enum."""
        if exit_code == 0:
            return FailureCategory.SUCCESS

        combined = f"{stdout}\n{stderr}"
        if "SyntaxError" in combined:
            return FailureCategory.SYNTAX_ERROR
        if "ModuleNotFoundError" in combined or "ImportError" in combined:
            return FailureCategory.IMPORT_ERROR
        if "AssertionError" in combined or "FAILED" in combined:
            return FailureCategory.ASSERTION_FAILURE

        return FailureCategory.RUNTIME_ERROR
