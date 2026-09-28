"""
Problem Decomposition Engine for ChakrView Reasoning (Step 19).

Provides deterministic hierarchical subproblem decomposition with strict recursion
and breadth bounding to prevent uncontrolled expansion.
"""

from dataclasses import dataclass, field, asdict
from enum import Enum
import re
import time
from typing import Dict, List, Optional, Any
import uuid

from chakrview.reasoning.task import ReasoningTask, ReasoningTaskType


class SubproblemStatus(str, Enum):
    """Lifecycle status of an individual subproblem."""
    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    RESOLVED = "RESOLVED"
    BLOCKED = "BLOCKED"
    FAILED = "FAILED"


class DecompositionLimitError(Exception):
    """Raised when problem decomposition exceeds configured recursive or breadth limits."""
    pass


@dataclass
class Subproblem:
    """
    Strongly typed discrete subproblem within a decomposition tree.
    """
    subproblem_id: str = field(default_factory=lambda: f"sub_{uuid.uuid4().hex[:8]}")
    parent_id: Optional[str] = None
    objective: str = ""
    dependencies: List[str] = field(default_factory=list)
    required_evidence: List[str] = field(default_factory=list)
    completion_criteria: str = ""
    priority: int = 1
    status: SubproblemStatus = SubproblemStatus.PENDING
    result: Optional[Any] = None
    depth: int = 0
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "subproblem_id": self.subproblem_id,
            "parent_id": self.parent_id,
            "objective": self.objective,
            "dependencies": list(self.dependencies),
            "required_evidence": list(self.required_evidence),
            "completion_criteria": self.completion_criteria,
            "priority": self.priority,
            "status": self.status.value,
            "result": self.result,
            "depth": self.depth,
            "created_at": self.created_at,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Subproblem":
        d = dict(data)
        d["status"] = SubproblemStatus(d["status"])
        return cls(**d)


@dataclass
class DecompositionTree:
    """
    Hierarchical graph of subproblems derived from a root reasoning task.
    """
    root_task_id: str
    subproblems: Dict[str, Subproblem] = field(default_factory=dict)
    max_depth: int = 4
    max_subproblems: int = 10

    def add_subproblem(self, subproblem: Subproblem) -> None:
        """Add subproblem subject to strict depth and capacity constraints."""
        if subproblem.depth > self.max_depth:
            raise DecompositionLimitError(
                f"Subproblem depth {subproblem.depth} exceeds max_depth {self.max_depth}."
            )
        if len(self.subproblems) >= self.max_subproblems:
            raise DecompositionLimitError(
                f"Subproblem count {len(self.subproblems)} reached limit {self.max_subproblems}."
            )
        self.subproblems[subproblem.subproblem_id] = subproblem

    def get_execution_order(self) -> List[Subproblem]:
        """
        Topological / dependency-ordered list of subproblems.
        Resolves dependencies first, then sorts by priority.
        """
        ordered: List[Subproblem] = []
        visited = set()
        temp_mark = set()

        def visit(n_id: str):
            if n_id in temp_mark:
                # Cycle fallback: ignore dependency to prevent hang
                return
            if n_id not in visited and n_id in self.subproblems:
                temp_mark.add(n_id)
                sub = self.subproblems[n_id]
                for dep_id in sub.dependencies:
                    visit(dep_id)
                temp_mark.remove(n_id)
                visited.add(n_id)
                ordered.append(sub)

        # Sort remaining by priority (descending)
        all_subs = sorted(self.subproblems.values(), key=lambda s: s.priority, reverse=True)
        for s in all_subs:
            if s.subproblem_id not in visited:
                visit(s.subproblem_id)

        return ordered

    def to_dict(self) -> Dict[str, Any]:
        return {
            "root_task_id": self.root_task_id,
            "subproblems": {k: v.to_dict() for k, v in self.subproblems.items()},
            "max_depth": self.max_depth,
            "max_subproblems": self.max_subproblems,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DecompositionTree":
        tree = cls(
            root_task_id=data["root_task_id"],
            max_depth=data.get("max_depth", 4),
            max_subproblems=data.get("max_subproblems", 10),
        )
        for k, v in data.get("subproblems", {}).items():
            tree.subproblems[k] = Subproblem.from_dict(v)
        return tree


class ProblemDecomposer:
    """
    Deterministic rule-based decomposition layer for cognitive tasks.
    """

    def __init__(self, max_depth: int = 4, max_subproblems: int = 10) -> None:
        self.max_depth = max_depth
        self.max_subproblems = max_subproblems

    def decompose(self, task: ReasoningTask) -> DecompositionTree:
        """
        Decomposes task into structured subproblems based on task type and structure.
        """
        tree = DecompositionTree(
            root_task_id=task.task_id,
            max_depth=self.max_depth,
            max_subproblems=self.max_subproblems,
        )

        objective = task.normalized_objective or task.original_objective

        if task.task_type == ReasoningTaskType.MATHEMATICAL:
            self._decompose_mathematical(tree, task, objective)
        elif task.task_type == ReasoningTaskType.DECISION_MAKING:
            self._decompose_decision(tree, task, objective)
        elif task.task_type == ReasoningTaskType.HYPOTHESIS_TESTING:
            self._decompose_hypothesis_testing(tree, task, objective)
        else:
            self._decompose_general(tree, task, objective)

        return tree

    def _decompose_mathematical(
        self, tree: DecompositionTree, task: ReasoningTask, objective: str
    ) -> None:
        # Step 1: Parse expression and identify operations
        sub1 = Subproblem(
            subproblem_id=f"{task.task_id}_parse",
            objective="Extract mathematical terms, operators, and grouping from expression",
            completion_criteria="Extracted valid mathematical syntax",
            priority=3,
            depth=1,
        )
        tree.add_subproblem(sub1)

        # Step 2: Compute intermediate sub-expressions
        sub2 = Subproblem(
            subproblem_id=f"{task.task_id}_compute",
            objective="Execute arithmetic / algebraic computation in order of precedence",
            dependencies=[sub1.subproblem_id],
            completion_criteria="Obtained numerical or symbolic intermediate value",
            priority=2,
            depth=1,
        )
        tree.add_subproblem(sub2)

        # Step 3: Verification
        sub3 = Subproblem(
            subproblem_id=f"{task.task_id}_verify",
            objective="Verify calculation results using invariant or alternative computation",
            dependencies=[sub2.subproblem_id],
            completion_criteria="Confirmed computation is mathematically sound",
            priority=1,
            depth=1,
        )
        tree.add_subproblem(sub3)

    def _decompose_decision(
        self, tree: DecompositionTree, task: ReasoningTask, objective: str
    ) -> None:
        sub1 = Subproblem(
            subproblem_id=f"{task.task_id}_gather_options",
            objective="Identify decision candidate actions and relevant constraints",
            completion_criteria="List of non-empty candidates and constraints compiled",
            priority=4,
            depth=1,
        )
        tree.add_subproblem(sub1)

        sub2 = Subproblem(
            subproblem_id=f"{task.task_id}_evaluate_tradeoffs",
            objective="Evaluate evidence, risks, and utility for each candidate",
            dependencies=[sub1.subproblem_id],
            completion_criteria="Ranked candidates by expected utility",
            priority=3,
            depth=1,
        )
        tree.add_subproblem(sub2)

        sub3 = Subproblem(
            subproblem_id=f"{task.task_id}_select_action",
            objective="Select optimal action and verify capability governance compatibility",
            dependencies=[sub2.subproblem_id],
            completion_criteria="Action selected with verified authorization path",
            priority=2,
            depth=1,
        )
        tree.add_subproblem(sub3)

        sub4 = Subproblem(
            subproblem_id=f"{task.task_id}_verify_decision",
            objective="Verify decision meets task constraints and safety interlocks",
            dependencies=[sub3.subproblem_id],
            completion_criteria="Decision passes verification criteria",
            priority=1,
            depth=1,
        )
        tree.add_subproblem(sub4)

    def _decompose_hypothesis_testing(
        self, tree: DecompositionTree, task: ReasoningTask, objective: str
    ) -> None:
        sub1 = Subproblem(
            subproblem_id=f"{task.task_id}_formulate_hypotheses",
            objective="Formulate competing candidate hypotheses",
            completion_criteria="Generated distinct testable hypotheses",
            priority=3,
            depth=1,
        )
        tree.add_subproblem(sub1)

        sub2 = Subproblem(
            subproblem_id=f"{task.task_id}_gather_evidence",
            objective="Gather supporting and refuting evidence from state and memory",
            dependencies=[sub1.subproblem_id],
            completion_criteria="Evidence linked to each candidate hypothesis",
            priority=2,
            depth=1,
        )
        tree.add_subproblem(sub2)

        sub3 = Subproblem(
            subproblem_id=f"{task.task_id}_evaluate_consistency",
            objective="Evaluate contradiction, epistemic validity, and confidence",
            dependencies=[sub2.subproblem_id],
            completion_criteria="Determined supported/contradicted status for all hypotheses",
            priority=1,
            depth=1,
        )
        tree.add_subproblem(sub3)

    def _decompose_general(
        self, tree: DecompositionTree, task: ReasoningTask, objective: str
    ) -> None:
        sub1 = Subproblem(
            subproblem_id=f"{task.task_id}_analyze",
            objective=f"Analyze objectives and extract core premises: {objective}",
            completion_criteria="Core premises identified",
            priority=3,
            depth=1,
        )
        tree.add_subproblem(sub1)

        sub2 = Subproblem(
            subproblem_id=f"{task.task_id}_reason",
            objective="Apply structured deduction or capability execution",
            dependencies=[sub1.subproblem_id],
            completion_criteria="Intermediate inference generated",
            priority=2,
            depth=1,
        )
        tree.add_subproblem(sub2)

        sub3 = Subproblem(
            subproblem_id=f"{task.task_id}_verify",
            objective="Verify final reasoning output satisfies initial criteria",
            dependencies=[sub2.subproblem_id],
            completion_criteria="Verified solution consistency",
            priority=1,
            depth=1,
        )
        tree.add_subproblem(sub3)
