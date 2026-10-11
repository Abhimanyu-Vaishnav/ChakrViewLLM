"""
ChakrView Step 113: Governed Multi-Agent Cognitive Contracts.

Defines roles, contracts, policies, and schemas for specialized cognitive workers.
Adheres strictly to:
- NEURAL BRAIN != KNOWLEDGE != MEMORY != TOOLS != COGNITION
- Strict context limits (<= 512 tokens)
- Explicit resource boundaries and allowed files
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set


class WorkerRole(str, Enum):
    """Specialized cognitive roles with distinct responsibilities."""
    PROJECT_ANALYST = "PROJECT_ANALYST"
    PLANNER = "PLANNER"
    IMPLEMENTER = "IMPLEMENTER"
    TEST_ENGINEER = "TEST_ENGINEER"
    REVIEWER = "REVIEWER"
    SYNTHESIZER = "SYNTHESIZER"


class ReviewerVerdict(str, Enum):
    """Independent reviewer evaluation outcomes."""
    APPROVE = "APPROVE"
    REJECT = "REJECT"
    REQUEST_REVISION = "REQUEST_REVISION"


@dataclass
class WorkerContract:
    """
    Explicit, auditable specification of a worker's authorized scope.
    Any attempt to execute outside this contract is blocked before execution.
    """
    worker_id: str
    role: WorkerRole
    task_id: str
    allowed_tools: List[str]
    allowed_files: List[str]
    context_budget: int = 512
    expected_output_schema: Dict[str, Any] = field(default_factory=dict)
    dependency_requirements: List[str] = field(default_factory=list)
    success_criteria: str = ""
    failure_reporting_rules: str = ""

    def validate(self) -> None:
        """Validate structural integrity of the contract."""
        if not self.worker_id or not self.worker_id.strip():
            raise ValueError("WorkerContract error: worker_id must not be empty")
        if not self.task_id or not self.task_id.strip():
            raise ValueError("WorkerContract error: task_id must not be empty")
        if self.context_budget <= 0 or self.context_budget > 512:
            raise ValueError(f"WorkerContract error: context_budget {self.context_budget} exceeds maximum 512 tokens")
        if not isinstance(self.role, WorkerRole):
            raise ValueError(f"WorkerContract error: invalid WorkerRole {self.role}")
        # Role-based tool permissions check
        if self.role in (WorkerRole.PROJECT_ANALYST, WorkerRole.PLANNER, WorkerRole.REVIEWER, WorkerRole.SYNTHESIZER):
            if "write_file" in self.allowed_tools:
                raise ValueError(f"WorkerContract error: role {self.role} is prohibited from write_file tool authority")


@dataclass
class WorkerPackage:
    """
    Isolated context packet supplied to a cognitive worker.
    Zero full repository dumping allowed.
    """
    task_id: str
    objective: str
    targeted_code_context: Dict[str, str] = field(default_factory=dict)
    dependency_outputs: Dict[str, Any] = field(default_factory=dict)
    relevant_ppb_records: List[Dict[str, Any]] = field(default_factory=list)
    acceptance_criteria: str = ""
    measured_context_tokens: int = 0
