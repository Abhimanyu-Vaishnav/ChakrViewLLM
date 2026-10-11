"""
ChakrView Step 59: 4-Level Repository Verifier.

Executes hierarchical verification:
Level 1: Targeted Verification (targeted test file passes)
Level 2: Regression Verification (all other test files pass)
Level 3: Repository-Wide Verification (entire suite passes with zero errors)
Level 4: Diff Integrity (only allowed target files were modified)
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Set

from chakrview.arena.executor import SandboxedExecutor
from chakrview.arena.models import TestResult, FailureCategory
from chakrview.arena.workspace import IsolatedWorkspace


@dataclass
class RepositoryVerificationResult:
    level1_targeted_pass: bool = False
    level2_regression_pass: bool = False
    level3_repo_state_pass: bool = False
    level4_diff_integrity: bool = False
    overall_verified: bool = False
    unexpected_modified_files: List[str] = field(default_factory=list)
    test_result_summary: Dict[str, Any] = field(default_factory=dict)
    diagnostics: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class RepositoryVerifier:
    """
    4-Tier Repository Verifier coordinating with SandboxedExecutor.
    """

    def __init__(self, executor: Optional[SandboxedExecutor] = None) -> None:
        self.executor = executor or SandboxedExecutor()

    def verify_repository(
        self,
        workspace: IsolatedWorkspace,
        targeted_test_rel_path: Optional[str] = None,
        allowed_modified_files: Optional[Set[str]] = None,
        currently_modified_files: Optional[List[str]] = None,
    ) -> RepositoryVerificationResult:
        """
        Execute full 4-level verification.
        """
        # Run repository-wide tests
        test_res, failure_cat = self.executor.run_tests(workspace)
        all_passed = (failure_cat == FailureCategory.SUCCESS and test_res.passed > 0 and test_res.failed == 0 and test_res.errors == 0)

        # Level 1: Targeted test pass
        level1 = False
        output_text = (test_res.stdout or "") + (test_res.stderr or "")
        if all_passed:
            level1 = True
        elif targeted_test_rel_path:
            # Targeted test is considered failed if it appears under failures or FAILED lines
            target_failed = (
                f"FAILED tests/{targeted_test_rel_path}" in output_text
                or f"FAILED {targeted_test_rel_path}" in output_text
                or f"::{targeted_test_rel_path}" in output_text and "FAILED" in output_text
            )
            level1 = not target_failed and test_res.passed > 0

        # Level 2: Regression pass
        level2 = (test_res.failed == 0 and test_res.errors == 0) if all_passed else False

        # Level 3: Full repo pass
        level3 = all_passed

        # Level 4: Diff integrity
        level4 = True
        unexpected: List[str] = []
        if allowed_modified_files is not None and currently_modified_files is not None:
            for f in currently_modified_files:
                if f not in allowed_modified_files:
                    level4 = False
                    unexpected.append(f)

        overall = level1 and level2 and level3 and level4

        diag = None
        if not overall:
            diag = (test_res.stderr or test_res.stdout or "Verification failed")[:200]

        return RepositoryVerificationResult(
            level1_targeted_pass=level1,
            level2_regression_pass=level2,
            level3_repo_state_pass=level3,
            level4_diff_integrity=level4,
            overall_verified=overall,
            unexpected_modified_files=unexpected,
            test_result_summary={
                "passed": test_res.passed,
                "failed": test_res.failed,
                "errors": test_res.errors,
                "duration": test_res.duration_seconds,
            },
            diagnostics=diag,
        )
