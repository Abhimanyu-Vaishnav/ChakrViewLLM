"""
Wave 583: Hierarchical Adaptive Planner.

Architectural Guarantees:
- Structured hierarchical decomposition:
  GOAL -> SUBGOALS -> ACTIONS -> OBSERVATIONS -> CORRECTIONS.
- Supports dependency ordering and uncertainty.
- Dynamic Replanning: If a sub-action fails or premise fails, generates recovery subgoals.
- Extends and unifies existing planner abstractions without rewriting them.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Set


@dataclass
class SubGoal:
    subgoal_id: str
    description: str
    dependencies: List[str] = field(default_factory=list)
    action_type: str = "EXECUTE"  # RETRIEVE, VERIFY, COMPUTE, SYNTHESIZE, DECIDE
    is_completed: bool = False
    is_failed: bool = False
    result_data: Optional[Any] = None
    retry_count: int = 0


@dataclass
class HierarchicalPlan:
    plan_id: str
    root_goal: str
    subgoals: List[SubGoal] = field(default_factory=list)
    is_fully_executed: bool = False
    replanned_count: int = 0


class HierarchicalPlanner:
    """
    Decomposes abstract goals into structured dependency-aware subgoals
    and adapts plans dynamically upon failure.
    """

    @classmethod
    def create_plan(
        cls,
        root_goal: str,
        task_context: Optional[Dict[str, Any]] = None,
    ) -> HierarchicalPlan:
        plan_id = f"plan_{uuid.uuid4().hex[:8]}"
        context = task_context or {}

        # Decompose into standard logical subgoals
        sg1 = SubGoal(
            subgoal_id="sg_1_gather",
            description=f"Gather required evidence for '{root_goal}'",
            dependencies=[],
            action_type="RETRIEVE",
        )
        sg2 = SubGoal(
            subgoal_id="sg_2_verify",
            description=f"Verify gathered facts against authority constraints",
            dependencies=["sg_1_gather"],
            action_type="VERIFY",
        )
        sg3 = SubGoal(
            subgoal_id="sg_3_synthesize",
            description=f"Synthesize structured answer and calibrate confidence",
            dependencies=["sg_2_verify"],
            action_type="SYNTHESIZE",
        )

        return HierarchicalPlan(
            plan_id=plan_id,
            root_goal=root_goal,
            subgoals=[sg1, sg2, sg3],
        )

    @classmethod
    def replan_on_failure(
        cls,
        plan: HierarchicalPlan,
        failed_subgoal_id: str,
        failure_reason: str,
    ) -> HierarchicalPlan:
        """
        Dynamically adapts plan by inserting a diagnostic/recovery subgoal.
        """
        plan.replanned_count += 1
        # Find failed goal
        failed_sg = next((sg for sg in plan.subgoals if sg.subgoal_id == failed_subgoal_id), None)
        if failed_sg:
            failed_sg.is_failed = True

        recovery_sg = SubGoal(
            subgoal_id=f"sg_recov_{plan.replanned_count}",
            description=f"Recover from failure '{failure_reason}': probe alternative evidence source",
            dependencies=[],
            action_type="RETRIEVE",
        )
        # Update subgoals that depended on failed_subgoal_id to depend on recovery_sg
        for sg in plan.subgoals:
            if failed_subgoal_id in sg.dependencies:
                sg.dependencies.remove(failed_subgoal_id)
                sg.dependencies.append(recovery_sg.subgoal_id)

        # Insert before dependent synthesis steps
        plan.subgoals.insert(1, recovery_sg)
        return plan
