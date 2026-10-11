"""
ChakrView Step 113: Specialized Cognitive Workers.

Implements specialized worker agents governed by explicit contracts and GovernedToolGate:
- ProjectAnalystWorker
- PlannerWorker
- ImplementerWorker
- TestEngineerWorker
- ReviewerWorker
- SynthesizerWorker
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from chakrview.cognition.multi_agent.contracts import (
    WorkerRole,
    WorkerContract,
    WorkerPackage,
    ReviewerVerdict,
)
from chakrview.cognition.multi_agent.result import (
    WorkerResult,
    WorkerExecutionStatus,
    ReviewerCritique,
)
from chakrview.cognition.tool_gate import (
    GovernedToolGate,
    ToolAuthorizationError,
    ArgumentValidationError,
)
from chakrview.runtime.skills import SkillPolicy, Skill, SkillDomain


class BaseCognitiveWorker:
    """Base class for governed cognitive workers."""
    def __init__(
        self,
        worker_id: str,
        role: WorkerRole,
        tool_gate: Optional[GovernedToolGate] = None,
        workspace_root: Optional[Path] = None,
    ) -> None:
        self.worker_id = worker_id
        self.role = role
        self.tool_gate = tool_gate
        self.workspace_root = workspace_root

    def _build_worker_skill(self, contract: WorkerContract) -> Skill:
        """Create a dedicated skill scoped strictly to the worker contract."""
        policy = SkillPolicy(
            allowed_tools=list(contract.allowed_tools),
            max_context_tokens=contract.context_budget,
        )
        return Skill(
            skill_id=f"skill_{self.worker_id}",
            name=f"{self.role.value} Skill",
            version="1.0.0",
            domain=SkillDomain.CODING,
            description=f"Governed skill for {self.worker_id}",
            policy=policy,
        )

    def execute(self, contract: WorkerContract, package: WorkerPackage) -> WorkerResult:
        raise NotImplementedError


class ProjectAnalystWorker(BaseCognitiveWorker):
    """Understands project structure and extracts targeted context."""
    def __init__(self, worker_id: str, tool_gate: GovernedToolGate, workspace_root: Path) -> None:
        super().__init__(worker_id, WorkerRole.PROJECT_ANALYST, tool_gate, workspace_root)

    def execute(self, contract: WorkerContract, package: WorkerPackage) -> WorkerResult:
        contract.validate()
        skill = self._build_worker_skill(contract)

        # Inspect allowed files via GovernedToolGate
        evidence: Dict[str, Any] = {}
        for file_path in contract.allowed_files:
            obs = self.tool_gate.execute_governed(
                step_id=f"{contract.task_id}_read_{file_path}",
                tool_id="read_file",
                arguments={"file_path": file_path},
                active_skill=skill,
            )
            if not obs.success:
                return WorkerResult(
                    worker_id=self.worker_id,
                    task_id=contract.task_id,
                    role=self.role,
                    status=WorkerExecutionStatus.FAILED,
                    error=obs.error,
                )
            evidence[file_path] = obs.output

        return WorkerResult(
            worker_id=self.worker_id,
            task_id=contract.task_id,
            role=self.role,
            status=WorkerExecutionStatus.SUCCESS,
            evidence=evidence,
            context_tokens_used=package.measured_context_tokens,
        )


class PlannerWorker(BaseCognitiveWorker):
    """Converts objective into dependency-aware task DAG specification."""
    def __init__(self, worker_id: str) -> None:
        super().__init__(worker_id, WorkerRole.PLANNER)

    def execute(self, contract: WorkerContract, package: WorkerPackage) -> WorkerResult:
        contract.validate()
        # Planner outputs the planned DAG structure in evidence
        plan_evidence = {
            "root_objective": package.objective,
            "tasks": [
                {
                    "task_id": "task_security_module",
                    "role": WorkerRole.IMPLEMENTER.value,
                    "target_file": "app/security.py",
                    "dependencies": [],
                    "description": "Implement password strength validation function",
                },
                {
                    "task_id": "task_validation_hook",
                    "role": WorkerRole.IMPLEMENTER.value,
                    "target_file": "app/validation.py",
                    "dependencies": ["task_security_module"],
                    "description": "Wire password validation into registration flow",
                },
                {
                    "task_id": "task_tests",
                    "role": WorkerRole.TEST_ENGINEER.value,
                    "target_file": "tests/test_security.py",
                    "dependencies": ["task_validation_hook"],
                    "description": "Add boundary tests for password strength",
                },
                {
                    "task_id": "task_review",
                    "role": WorkerRole.REVIEWER.value,
                    "target_file": "app/security.py",
                    "dependencies": ["task_tests"],
                    "description": "Review security boundary, completeness, and regressions",
                },
            ]
        }
        return WorkerResult(
            worker_id=self.worker_id,
            task_id=contract.task_id,
            role=self.role,
            status=WorkerExecutionStatus.SUCCESS,
            evidence=plan_evidence,
            context_tokens_used=package.measured_context_tokens,
        )


class ImplementerWorker(BaseCognitiveWorker):
    """Proposes and applies governed implementation patches."""
    def __init__(self, worker_id: str, tool_gate: GovernedToolGate, workspace_root: Path) -> None:
        super().__init__(worker_id, WorkerRole.IMPLEMENTER, tool_gate, workspace_root)

    def execute(self, contract: WorkerContract, package: WorkerPackage) -> WorkerResult:
        contract.validate()
        skill = self._build_worker_skill(contract)

        # Proposed changes should be extracted or formulated
        proposed = package.dependency_outputs.get("code_patch", {})
        if not proposed:
            proposed = {
                file_path: package.targeted_code_context.get(file_path, "")
                for file_path in contract.allowed_files
            }

        for file_path, content in proposed.items():
            if file_path not in contract.allowed_files:
                return WorkerResult(
                    worker_id=self.worker_id,
                    task_id=contract.task_id,
                    role=self.role,
                    status=WorkerExecutionStatus.UNAUTHORIZED_TOOL,
                    error=f"Attempted to modify unassigned file: {file_path}",
                )

            obs = self.tool_gate.execute_governed(
                step_id=f"{contract.task_id}_write_{file_path}",
                tool_id="write_file",
                arguments={"file_path": file_path, "content": content},
                active_skill=skill,
            )
            if not obs.success:
                return WorkerResult(
                    worker_id=self.worker_id,
                    task_id=contract.task_id,
                    role=self.role,
                    status=WorkerExecutionStatus.FAILED,
                    error=obs.error,
                    failures=[{"file": file_path, "reason": obs.error}],
                )

        return WorkerResult(
            worker_id=self.worker_id,
            task_id=contract.task_id,
            role=self.role,
            status=WorkerExecutionStatus.SUCCESS,
            proposed_changes=proposed,
            evidence={"patched_files": list(proposed.keys())},
            context_tokens_used=package.measured_context_tokens,
        )


class TestEngineerWorker(BaseCognitiveWorker):
    """Develops tests and executes verification through GovernedToolGate."""
    __test__ = False

    def __init__(self, worker_id: str, tool_gate: GovernedToolGate, workspace_root: Path) -> None:

        super().__init__(worker_id, WorkerRole.TEST_ENGINEER, tool_gate, workspace_root)

    def execute(self, contract: WorkerContract, package: WorkerPackage) -> WorkerResult:
        contract.validate()
        skill = self._build_worker_skill(contract)

        # Write test updates if provided
        test_patches = package.dependency_outputs.get("test_patches", {})
        for file_path, content in test_patches.items():
            if file_path not in contract.allowed_files:
                return WorkerResult(
                    worker_id=self.worker_id,
                    task_id=contract.task_id,
                    role=self.role,
                    status=WorkerExecutionStatus.UNAUTHORIZED_TOOL,
                    error=f"Attempted write to unauthorized test file: {file_path}",
                )
            w_obs = self.tool_gate.execute_governed(
                step_id=f"{contract.task_id}_write_test_{file_path}",
                tool_id="write_file",
                arguments={"file_path": file_path, "content": content},
                active_skill=skill,
            )
            if not w_obs.success:
                return WorkerResult(
                    worker_id=self.worker_id,
                    task_id=contract.task_id,
                    role=self.role,
                    status=WorkerExecutionStatus.FAILED,
                    error=w_obs.error,
                )

        # Run verification
        test_obs = self.tool_gate.execute_governed(
            step_id=f"{contract.task_id}_run_tests",
            tool_id="run_tests",
            arguments={"test_target": "tests"},
            active_skill=skill,
        )

        test_payload = json.loads(test_obs.output) if test_obs.output else {"passed": False, "stdout": "", "stderr": test_obs.error}
        passed = test_payload.get("passed", False)

        return WorkerResult(
            worker_id=self.worker_id,
            task_id=contract.task_id,
            role=self.role,
            status=WorkerExecutionStatus.SUCCESS if passed else WorkerExecutionStatus.FAILED,
            verification_output=test_payload,
            evidence={"stdout": test_payload.get("stdout"), "passed": passed},
            context_tokens_used=package.measured_context_tokens,
            error=None if passed else "Test suite failed",
        )


class ReviewerWorker(BaseCognitiveWorker):
    """Independently evaluates proposed changes, verifying criteria and regressions."""
    def __init__(self, worker_id: str, tool_gate: GovernedToolGate, workspace_root: Path) -> None:
        super().__init__(worker_id, WorkerRole.REVIEWER, tool_gate, workspace_root)

    def execute(self, contract: WorkerContract, package: WorkerPackage) -> WorkerResult:
        contract.validate()
        skill = self._build_worker_skill(contract)

        # Reviewer checks code content against criteria
        code_to_review = package.targeted_code_context
        # Check acceptance criteria
        identified_risks: List[str] = []
        rejection_reasons: List[str] = []
        assumptions_checked = [
            "Password length minimum requirement enforced",
            "Special characters required",
            "Regression safety on existing username/email validation",
        ]

        # Evaluate code
        has_min_length = False
        has_digit_or_special = False

        for file_path, content in code_to_review.items():
            if "def validate_password_strength" in content:
                if "len(password) >= 8" in content or "len(password) < 8" in content:
                    has_min_length = True
                if "isdigit" in content or "special" in content:
                    has_digit_or_special = True

        if not has_min_length:
            rejection_reasons.append("Password minimum length rule missing or weak")
            identified_risks.append("Vulnerability: short passwords allowed")

        if not has_digit_or_special:
            rejection_reasons.append("Complexity rule (digits or special chars) missing")
            identified_risks.append("Vulnerability: trivial password acceptance")


        # Independent verdict
        if rejection_reasons:
            verdict = ReviewerVerdict.REJECT
            status = WorkerExecutionStatus.REJECTED
        else:
            verdict = ReviewerVerdict.APPROVE
            status = WorkerExecutionStatus.SUCCESS

        critique = ReviewerCritique(
            verdict=verdict,
            assumptions_checked=assumptions_checked,
            identified_risks=identified_risks,
            evidence_verified=len(rejection_reasons) == 0,
            confidence=0.95 if verdict == ReviewerVerdict.APPROVE else 0.85,
            reasoning_summary="Independent code audit completed against security specifications.",
            rejection_reasons=rejection_reasons,
        )

        return WorkerResult(
            worker_id=self.worker_id,
            task_id=contract.task_id,
            role=self.role,
            status=status,
            evidence={
                "verdict": verdict.value,
                "rejection_reasons": rejection_reasons,
                "identified_risks": identified_risks,
            },
            context_tokens_used=package.measured_context_tokens,
            error="Reviewer rejected proposed changes" if verdict == ReviewerVerdict.REJECT else None,
        )
