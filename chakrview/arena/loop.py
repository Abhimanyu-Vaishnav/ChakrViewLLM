"""
ChakrView Project Arena: Closed-Loop Feedback & Repair Controller.

Implements the executable real-world software engineering feedback loop:
    PROJECT SPECIFICATION
          ↓
    TASK DECOMPOSITION & PLANNING
          ↓
    INITIAL CODE GENERATION
          ↓
    WORKSPACE MATERIALIZATION
          ↓
    SANDBOXED EXECUTION & TESTS
          ↓
    FAILURE CLASSIFICATION & TRACEBACK EXTRACTION
          ↓
    DIAGNOSTIC REPAIR PROMPT FORMATTING
          ↓
    PATCH GENERATION & APPLICATION
          ↓
    RE-TEST & REGRESSION CHECK
          ↓
    CONVERGENCE EVALUATION
          ↓
    EXECUTION HISTORY & RIL EXPERIENCE CAPTURE
"""

from __future__ import annotations

import time
from typing import Optional, Callable, Dict, Any, List

from chakrview.arena.models import (
    ProjectSpecification,
    ProjectManifest,
    SourceFile,
    FileRole,
    FailureCategory,
    IterationRecord,
    ExecutionHistory,
    ArenaExecutionResult,
)
from chakrview.arena.workspace import IsolatedWorkspace
from chakrview.arena.executor import SandboxedExecutor
from chakrview.arena.evaluator import ArenaEvaluator


class ArenaClosedLoopController:
    """
    Manages the multi-round iterative feedback and repair loop for a project.
    """
    def __init__(
        self,
        evaluator: Optional[ArenaEvaluator] = None,
        max_iterations: int = 3,
    ) -> None:
        self.evaluator = evaluator or ArenaEvaluator()
        self.max_iterations = max_iterations

    def format_initial_prompt(self, spec: ProjectSpecification) -> str:
        """Format an initial project synthesis prompt from a specification."""
        return (
            f"# PROJECT SPECIFICATION: {spec.project_name}\n"
            f"# ID: {spec.project_id}\n"
            f"# DESCRIPTION: {spec.description}\n"
            f"# LANGUAGE: {spec.language}\n"
            f"# DEPENDENCIES: {', '.join(spec.dependencies)}\n\n"
            f"# Task: Implement {spec.entrypoint} satisfying all requirements.\n"
        )

    def format_repair_prompt(
        self,
        spec: ProjectSpecification,
        current_code: str,
        diagnostic: Dict[str, Any],
    ) -> str:
        """Format an iterative repair prompt containing the error diagnostic."""
        return (
            f"# REPAIR TASK FOR PROJECT: {spec.project_name}\n"
            f"# FAILING TEST: {diagnostic.get('failing_test')}\n"
            f"# EXCEPTION TYPE: {diagnostic.get('exception_type')}\n"
            f"# ERROR MESSAGE: {diagnostic.get('error_message')}\n\n"
            f"# CURRENT CODE:\n{current_code}\n\n"
            f"# TRACEBACK SNIPPET:\n{diagnostic.get('raw_snippet')}\n\n"
            f"# Task: Fix the defect and return corrected code.\n"
        )

    def run_closed_loop(
        self,
        manifest: ProjectManifest,
        repair_generator: Optional[Callable[[str], str]] = None,
        base_dir: Optional[str] = None,
        preserve_workspace: bool = False,
        timeout_seconds: Optional[float] = None,
    ) -> ExecutionHistory:
        """
        Execute the closed loop:
        1. Materialize workspace with initial manifest.
        2. Run tests.
        3. If failure and repair_generator provided, loop up to max_iterations.
        4. Return complete ExecutionHistory with patch diffs.
        """
        history = ExecutionHistory(project_id=manifest.specification.project_id)
        spec = manifest.specification
        effective_timeout = timeout_seconds or spec.timeout_seconds

        with IsolatedWorkspace(manifest, base_dir=base_dir, preserve=preserve_workspace) as ws:
            for iteration in range(1, self.max_iterations + 1):
                iter_start = time.perf_counter()

                # 1. Run tests in sandbox
                test_res, failure_cat = self.evaluator.executor.run_tests(ws, timeout_seconds=effective_timeout)
                iter_duration = time.perf_counter() - iter_start

                # 2. Extract diagnostic if failure occurred
                diagnostic = self.evaluator.executor.extract_traceback_diagnostic(test_res)
                diagnosis_msg = diagnostic.get("error_message") if diagnostic else None

                # 3. Create iteration record
                record = IterationRecord(
                    iteration=iteration,
                    files_modified=[],
                    test_result=test_res,
                    failure_category=failure_cat,
                    diagnosis=diagnosis_msg,
                    patch_diffs=list(ws.patch_history),
                    duration_seconds=iter_duration,
                )
                history.add_iteration(record)

                # 4. If all tests pass, break early (converged)
                if history.converged or failure_cat == FailureCategory.SUCCESS:
                    break

                # 5. If no repair generator or last iteration, finish
                if not repair_generator or iteration >= self.max_iterations:
                    break

                # 6. Apply repair - determine target source file to patch
                target_file = diagnostic.get("target_file", spec.entrypoint) if diagnostic else spec.entrypoint
                if target_file:
                    target_file = target_file.replace("\\", "/").split("/")[-1]

                # Identify eligible source files (role == SOURCE or INIT)
                source_files = {f.path for f in ws.manifest.files if f.role in (FileRole.SOURCE, FileRole.INIT)}

                # If target_file is a test file or not in source_files, redirect to actual source file
                if target_file not in source_files or target_file.startswith("test_") or target_file.endswith("_test.py"):
                    if spec.entrypoint in source_files:
                        target_file = spec.entrypoint
                    else:
                        # Check if any source file is mentioned in failure output
                        found = None
                        combined_out = f"{test_res.stdout}\n{test_res.stderr}"
                        for sf in sorted(list(source_files)):
                            if sf in combined_out:
                                found = sf
                                break
                        target_file = found if found else (sorted(list(source_files))[0] if source_files else spec.entrypoint)

                current_sfile = ws.manifest.get_file(target_file)
                current_code = current_sfile.content if current_sfile else ""

                repair_prompt = self.format_repair_prompt(spec, current_code, diagnostic or {})
                new_code = repair_generator(repair_prompt)

                if new_code and new_code != current_code:
                    diff = ws.apply_patch(target_file, new_code)
                    record.files_modified.append(target_file)
                    record.patch_diffs.append(diff)
                else:
                    # Generator produced no change; break to prevent looping
                    break

        return history
