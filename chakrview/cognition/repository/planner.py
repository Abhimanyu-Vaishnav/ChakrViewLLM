"""
ChakrView Step 59: Repository Planning & Action Contracts.

Defines:
- RepositoryActionType: Explicit enum of valid repository actions.
- RepositoryAction: Strongly typed action envelope.
- RepositoryObservation: Structured observation returned from ChakrKshetra execution.
- RepositoryDiagnosis: Structured diagnosis separating symptom from root cause and dependency path.
- RepositoryPlan: Structured repository plan.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
import time
from typing import Any, Dict, List, Optional


class RepositoryActionType(str, Enum):
    INSPECT_FILE = "INSPECT_FILE"
    INSPECT_DEPENDENCIES = "INSPECT_DEPENDENCIES"
    RUN_TEST = "RUN_TEST"
    RUN_TARGETED_TEST = "RUN_TARGETED_TEST"
    APPLY_PATCH = "APPLY_PATCH"
    REVERT_PATCH = "REVERT_PATCH"
    VERIFY_REPOSITORY = "VERIFY_REPOSITORY"


@dataclass
class RepositoryAction:
    action_id: str
    action_type: RepositoryActionType
    target_file: Optional[str] = None
    payload: Dict[str, Any] = field(default_factory=dict)
    rationale: str = ""
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["action_type"] = self.action_type.value
        return d


@dataclass
class RepositoryObservation:
    action_id: str
    action_type: RepositoryActionType
    target_file: Optional[str]
    success: bool
    stdout: str = ""
    stderr: str = ""
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    affected_files: List[str] = field(default_factory=list)
    test_passed: int = 0
    test_failed: int = 0
    duration_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["action_type"] = self.action_type.value
        return d


@dataclass
class RepositoryDiagnosis:
    symptom_module: str
    symptom_description: str
    suspected_root_cause_module: str
    dependency_chain: List[str]
    root_cause_description: str
    proposed_correction: str
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RepositoryPlan:
    task_id: str
    objective: str
    inspected_files: List[str] = field(default_factory=list)
    dependency_hypotheses: List[str] = field(default_factory=list)
    suspected_root_causes: List[str] = field(default_factory=list)
    planned_actions: List[RepositoryAction] = field(default_factory=list)
    verification_strategy: str = ""
    rollback_strategy: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["planned_actions"] = [a.to_dict() for a in self.planned_actions]
        return d
