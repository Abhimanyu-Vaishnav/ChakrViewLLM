"""
Lightweight Execution Graph Abstraction for ChakrView (Step 15).

Provides dependency-aware execution tracking, cycle detection, deterministic
topological sorting, and safe cascading failure propagation.
"""

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Any, Tuple

from chakrview.cognition.planner import PlanStep, PlanStepStatus, CognitivePlan


class CycleDetectedError(ValueError):
    """Raised when an execution graph contains a cyclic dependency."""
    pass


class GraphExecutionError(RuntimeError):
    """Raised on illegal graph operations or inconsistencies."""
    pass


class ExecutionGraph:
    """
    Lightweight dependency-aware execution graph (DAG).
    
    Attributes:
        plan: Underlying CognitivePlan.
        steps: Dictionary mapping step_id to PlanStep.
        adjacency: Mapping from dependency step_id to list of dependent step_ids (forward edges).
        reverse_adjacency: Mapping from dependent step_id to list of dependency step_ids (back edges).
    """

    def __init__(self, plan: CognitivePlan) -> None:
        self.plan = plan
        self.steps: Dict[str, PlanStep] = {s.step_id: s for s in plan.steps}
        self.adjacency: Dict[str, List[str]] = {s.step_id: [] for s in plan.steps}
        self.reverse_adjacency: Dict[str, List[str]] = {s.step_id: [] for s in plan.steps}

        self._build_edges()
        self._validate_acyclic()

    def _build_edges(self) -> None:
        """Construct forward and reverse adjacency lists."""
        for step in self.steps.values():
            for dep_id in step.dependencies:
                if dep_id not in self.steps:
                    raise GraphExecutionError(
                        f"Step '{step.step_id}' declares nonexistent dependency '{dep_id}'."
                    )
                self.adjacency[dep_id].append(step.step_id)
                self.reverse_adjacency[step.step_id].append(dep_id)

    def _validate_acyclic(self) -> None:
        """Detect any cycles in the dependency graph using Kahn's algorithm."""
        in_degree = {sid: len(deps) for sid, deps in self.reverse_adjacency.items()}
        queue = deque([sid for sid, deg in in_degree.items() if deg == 0])
        visited_count = 0

        while queue:
            node = queue.popleft()
            visited_count += 1
            for child in self.adjacency[node]:
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)

        if visited_count != len(self.steps):
            raise CycleDetectedError(
                f"Cyclic dependency detected in execution plan '{self.plan.plan_id}'. "
                f"Resolved {visited_count}/{len(self.steps)} nodes."
            )

    def get_step(self, step_id: str) -> Optional[PlanStep]:
        """Look up step by ID."""
        return self.steps.get(step_id)

    def get_ready_steps(self) -> List[PlanStep]:
        """
        Return all steps whose dependencies are all COMPLETED and whose status
        is currently PENDING or READY.
        """
        ready: List[PlanStep] = []
        for step in self.steps.values():
            if step.status in (PlanStepStatus.PENDING, PlanStepStatus.READY):
                deps_met = True
                for dep_id in step.dependencies:
                    dep_step = self.steps.get(dep_id)
                    if not dep_step or dep_step.status != PlanStepStatus.COMPLETED:
                        deps_met = False
                        break
                if deps_met:
                    ready.append(step)
        return ready

    def mark_step_running(self, step_id: str) -> None:
        """Mark a step as actively executing."""
        step = self._require_step(step_id)
        step.status = PlanStepStatus.RUNNING

    def mark_step_completed(self, step_id: str) -> None:
        """Mark a step as successfully completed."""
        step = self._require_step(step_id)
        step.status = PlanStepStatus.COMPLETED

    def mark_step_failed(self, step_id: str, error: str = "") -> List[str]:
        """
        Mark a step as FAILED and recursively mark all downstream dependent
        steps as BLOCKED to prevent executing broken chains.
        
        Returns:
            List of blocked step IDs.
        """
        step = self._require_step(step_id)
        step.status = PlanStepStatus.FAILED
        step.metadata["failure_reason"] = error

        blocked_steps: List[str] = []
        queue = deque(self.adjacency[step_id])

        while queue:
            child_id = queue.popleft()
            child_step = self.steps[child_id]
            if child_step.status not in (PlanStepStatus.FAILED, PlanStepStatus.BLOCKED, PlanStepStatus.COMPLETED):
                child_step.status = PlanStepStatus.BLOCKED
                child_step.metadata["blocked_by"] = step_id
                child_step.metadata["blocked_reason"] = f"Dependency '{step_id}' failed: {error}"
                blocked_steps.append(child_id)
                for grandchild in self.adjacency[child_id]:
                    queue.append(grandchild)

        return blocked_steps

    def mark_step_skipped(self, step_id: str, reason: str = "") -> None:
        """Mark a step as skipped."""
        step = self._require_step(step_id)
        step.status = PlanStepStatus.SKIPPED
        step.metadata["skip_reason"] = reason

    def is_finished(self) -> bool:
        """
        Check if all steps have reached a terminal state
        (COMPLETED, FAILED, SKIPPED, or BLOCKED).
        """
        terminal_statuses = {
            PlanStepStatus.COMPLETED,
            PlanStepStatus.FAILED,
            PlanStepStatus.SKIPPED,
            PlanStepStatus.BLOCKED,
        }
        return all(s.status in terminal_statuses for s in self.steps.values())

    def has_failed(self) -> bool:
        """Check if any step in the graph has failed."""
        return any(s.status == PlanStepStatus.FAILED for s in self.steps.values())

    def get_topological_order(self) -> List[str]:
        """
        Return a deterministic topological order of step IDs.
        Ties are broken deterministically by step ID string order.
        """
        in_degree = {sid: len(deps) for sid, deps in self.reverse_adjacency.items()}
        # Priority/deterministic queue using sorted step IDs
        queue = sorted([sid for sid, deg in in_degree.items() if deg == 0])
        order: List[str] = []

        while queue:
            curr = queue.pop(0)
            order.append(curr)
            for child in sorted(self.adjacency[curr]):
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)
            queue.sort()

        return order

    def _require_step(self, step_id: str) -> PlanStep:
        step = self.steps.get(step_id)
        if step is None:
            raise GraphExecutionError(f"Step '{step_id}' not found in execution graph.")
        return step

    def summary(self) -> Dict[str, Any]:
        """Return status summary of the execution graph."""
        status_counts: Dict[str, int] = {}
        for s in self.steps.values():
            status_counts[s.status.value] = status_counts.get(s.status.value, 0) + 1
        return {
            "total_steps": len(self.steps),
            "status_counts": status_counts,
            "is_finished": self.is_finished(),
            "has_failed": self.has_failed(),
        }
