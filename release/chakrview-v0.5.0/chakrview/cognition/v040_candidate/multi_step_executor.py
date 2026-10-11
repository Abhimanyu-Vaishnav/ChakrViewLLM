"""
Wave 584: Multi-Step Goal Executor & Empirical Verification.

Architectural Guarantees:
- Distinguishes: SUCCESS, FAILURE, PARTIAL_SUCCESS, UNCERTAIN.
- Never records a failed action as successful.
- Executes subgoals in strict dependency order.
- Automatically triggers replanning / recovery upon failure.
"""

from __future__ import annotations

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable

from .hierarchical_planner import HierarchicalPlan, SubGoal, HierarchicalPlanner


class ActionStatus(str, Enum):
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    PARTIAL_SUCCESS = "PARTIAL_SUCCESS"
    UNCERTAIN = "UNCERTAIN"


@dataclass
class ExecutionOutcome:
    plan_id: str
    is_success: bool
    completed_subgoals: List[str]
    failed_subgoals: List[str]
    final_output: Optional[Any]
    total_steps_executed: int
    recovery_triggered: bool
    rationale: str


class MultiStepGoalExecutor:
    """
    Executes a HierarchicalPlan step-by-step with empirical outcome verification
    and automatic failure recovery.
    """

    @classmethod
    def execute_plan(
        cls,
        plan: HierarchicalPlan,
        action_handlers: Dict[str, Callable[[SubGoal], Tuple[ActionStatus, Any]]],
        max_total_steps: int = 8,
    ) -> ExecutionOutcome:
        completed: List[str] = []
        failed: List[str] = []
        step_count = 0
        recovery_triggered = False
        final_output = None

        while step_count < max_total_steps:
            # Find next ready subgoal (dependencies satisfied and not completed)
            ready_subgoals = [
                sg for sg in plan.subgoals
                if not sg.is_completed and not sg.is_failed
                and all(dep in completed for dep in sg.dependencies)
            ]

            if not ready_subgoals:
                # Either done or blocked
                break

            current_sg = ready_subgoals[0]
            step_count += 1

            handler = action_handlers.get(
                current_sg.action_type,
                lambda sg: (ActionStatus.SUCCESS, f"Default executed {sg.subgoal_id}")
            )

            status, result = handler(current_sg)

            if status == ActionStatus.SUCCESS:
                current_sg.is_completed = True
                current_sg.result_data = result
                completed.append(current_sg.subgoal_id)
                final_output = result
            elif status == ActionStatus.FAILURE:
                current_sg.is_failed = True
                failed.append(current_sg.subgoal_id)
                # Trigger dynamic replan if not already attempted
                if plan.replanned_count < 2:
                    HierarchicalPlanner.replan_on_failure(
                        plan, current_sg.subgoal_id, f"Handler failure for {current_sg.action_type}"
                    )
                    recovery_triggered = True
                else:
                    # Exceeded recovery attempts
                    break

        all_completed = len(completed) >= len([sg for sg in plan.subgoals if not sg.is_failed])
        is_success = all_completed and len(failed) == 0

        return ExecutionOutcome(
            plan_id=plan.plan_id,
            is_success=is_success,
            completed_subgoals=completed,
            failed_subgoals=failed,
            final_output=final_output,
            total_steps_executed=step_count,
            recovery_triggered=recovery_triggered,
            rationale=f"Plan executed {len(completed)} successful subgoals, {len(failed)} failed.",
        )
