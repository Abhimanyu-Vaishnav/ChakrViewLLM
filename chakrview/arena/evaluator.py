"""
ChakrView Project Arena: Evaluator & Quality Assessment.

Evaluates code syntax, project test execution, and computes objective metrics:
- validate_syntax: AST syntax parsing
- evaluate_project: Orchestrates workspace creation, execution, and artifact saving
- compute_metrics: Combines model, engine, and arena metrics
"""

from __future__ import annotations

import ast
import json
import time
from typing import Optional, Dict, Any, Tuple

from chakrview.arena.models import (
    ProjectManifest,
    ArenaExecutionResult,
    TestResult,
    FailureCategory,
    EvaluationMetrics,
)
from chakrview.arena.workspace import IsolatedWorkspace
from chakrview.arena.executor import SandboxedExecutor


class ArenaEvaluator:
    """
    Evaluates project source code and orchestrates test execution in the Arena.
    """
    def __init__(self, executor: Optional[SandboxedExecutor] = None) -> None:
        self.executor = executor or SandboxedExecutor()

    @staticmethod
    def validate_syntax(code: str, filename: str = "<code_string>") -> Tuple[bool, Optional[str]]:
        """
        Validate whether Python code parses cleanly into an AST.
        Returns:
            (is_valid, error_message_or_none)
        """
        try:
            ast.parse(code, filename=filename)
            return True, None
        except SyntaxError as e:
            return False, f"SyntaxError at line {e.lineno}, col {e.offset}: {e.msg}"
        except Exception as e:
            return False, f"Parse error: {str(e)}"

    def evaluate_manifest(
        self,
        manifest: ProjectManifest,
        base_dir: Optional[str] = None,
        preserve_workspace: bool = False,
        timeout_seconds: Optional[float] = None,
    ) -> ArenaExecutionResult:
        """
        Run full evaluation on a ProjectManifest:
        1. Validate syntax of all source files.
        2. Spin up an isolated workspace.
        3. Execute test suite.
        4. Save reports to artifacts.
        5. Return ArenaExecutionResult.
        """
        start_time = time.perf_counter()
        project_id = manifest.specification.project_id

        # 1. Syntax check across all source files
        all_syntax_valid = True
        first_syntax_err = None
        for sfile in manifest.get_source_files():
            valid, err = self.validate_syntax(sfile.content, filename=sfile.path)
            if not valid:
                all_syntax_valid = False
                first_syntax_err = f"{sfile.path}: {err}"
                break

        if not all_syntax_valid:
            duration = time.perf_counter() - start_time
            return ArenaExecutionResult(
                project_id=project_id,
                failure_category=FailureCategory.SYNTAX_ERROR,
                test_result=TestResult(
                    passed=0,
                    failed=0,
                    errors=1,
                    duration_seconds=duration,
                    stdout="",
                    stderr=first_syntax_err or "Syntax error in source",
                    exit_code=1,
                ),
                syntax_valid=False,
                syntax_error_message=first_syntax_err,
                execution_time_seconds=duration,
                workspace_path=None,
                artifacts={"syntax_error": first_syntax_err},
            )

        # 2. Workspace execution
        with IsolatedWorkspace(
            manifest,
            base_dir=base_dir,
            preserve=preserve_workspace,
        ) as ws:
            test_res, failure_cat = self.executor.run_tests(
                ws,
                timeout_seconds=timeout_seconds,
            )
            duration = time.perf_counter() - start_time

            # Save artifacts
            ast_report = {
                "project_id": project_id,
                "syntax_valid": True,
                "source_files_count": len(manifest.get_source_files()),
                "test_files_count": len(manifest.get_test_files()),
            }
            ast_report_path = ws.artifacts_dir / "ast_report.json"
            with open(ast_report_path, "w", encoding="utf-8") as f:
                json.dump(ast_report, f, indent=2)

            exec_result = ArenaExecutionResult(
                project_id=project_id,
                failure_category=failure_cat,
                test_result=test_res,
                syntax_valid=True,
                syntax_error_message=None,
                execution_time_seconds=duration,
                workspace_path=str(ws.workspace_dir) if preserve_workspace else None,
                artifacts={"ast_report": ast_report},
            )

            # Write score to workspace
            score_path = ws.artifacts_dir / "score.json"
            with open(score_path, "w", encoding="utf-8") as f:
                json.dump(exec_result.to_dict(), f, indent=2)

            return exec_result
