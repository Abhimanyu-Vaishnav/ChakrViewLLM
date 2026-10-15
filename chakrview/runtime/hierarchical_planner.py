"""
Batches 681-685: Hierarchical Dialogue Planner, DAG Execution Engine & Replanner.
Location: chakrview/runtime/hierarchical_planner.py

Responsibilities:
1. Decomposes compound, multi-step user goals into a DAG of Subtasks.
2. Identifies required parameters, tools, and dependencies.
3. Detects missing parameters and switches plan status to AWAITING_CLARIFICATION.
4. Executes ready subtasks topologically, resolving dynamic input bindings.
5. Invalidation & Replanning:
   - When a user updates a constraint (e.g. "change multiplier from 4 to 5"), marks dependent subtasks
     INVALIDATED, creates plan v2 linked to v1, and re-executes only necessary subtasks.
6. Seamlessly integrates with EpisodicTaskMemory and ToolRegistry.
"""
from __future__ import annotations

import re
import time
from typing import Any, Dict, List, Optional, Tuple

from chakrview.runtime.cognitive_router import (
    ComponentOrigin,
    GroundedResponse,
    RouteDecision,
    RouteType,
)
from chakrview.runtime.episodic_task_memory import (
    EpisodicTaskMemory,
    Subtask,
    SubtaskStatus,
    TaskPlan,
    TaskPlanStatus,
)
from chakrview.runtime.tools import CalculatorTool, TextUtilityTool, ToolExecutor, ToolRegistry


class HierarchicalDialoguePlanner:
    """
    Typed, inspectable hierarchical task planner and execution coordinator.
    """

    def __init__(
        self,
        episodic_memory: EpisodicTaskMemory,
        tool_registry: Optional[ToolRegistry] = None,
    ) -> None:
        self.memory = episodic_memory
        self.tool_registry = tool_registry or ToolRegistry()
        if not self.tool_registry.has_tool("calculator"):
            self.tool_registry.register(CalculatorTool())
        if not self.tool_registry.has_tool("text_utility"):
            self.tool_registry.register(TextUtilityTool())
        self.tool_executor = ToolExecutor(self.tool_registry)

    def plan_user_goal(
        self,
        session_id: str,
        user_goal: str,
        active_constraints: Optional[Dict[str, Any]] = None,
    ) -> TaskPlan:
        """
        Decomposes user goal into subtasks with explicit dependencies and bindings.
        """
        constraints = dict(active_constraints or {})
        q_low = user_goal.lower().strip()
        plan_id = f"plan_{int(time.time() * 1000)}"

        # 1. Multi-step Math Workflow: "Calculate 15 * 12 and then add 20 to the result"
        m_compound_math = re.search(
            r"(?:calculate|evaluate|compute)\s+(\d+(?:\.\d+)?\s*[\+\-\*\/]\s*\d+(?:\.\d+)?)\s+(?:and\s+then|then|next)\s+(?:add|subtract|multiply|divide)\s+(\d+(?:\.\d+)?)",
            q_low,
        )
        if m_compound_math:
            expr1 = m_compound_math.group(1).replace(" ", "")
            step2_num = m_compound_math.group(2)
            op2 = "*" if "multiply" in q_low else ("/" if "divide" in q_low else ("-" if "subtract" in q_low else "+"))

            st1 = Subtask(
                subtask_id="st_01",
                goal_description=f"Calculate base expression '{expr1}'",
                action_type="tool_call",
                tool_name="calculator",
                tool_arguments={"expression": expr1},
                dependencies=[],
                status=SubtaskStatus.READY,
            )
            st2 = Subtask(
                subtask_id="st_02",
                goal_description=f"Apply secondary operator '{op2} {step2_num}' to st_01 output",
                action_type="tool_call",
                tool_name="calculator",
                tool_arguments={"operator": op2, "operand": step2_num},
                dependencies=["st_01"],
                input_bindings={"base": "st_01.result"},
                status=SubtaskStatus.PROPOSED,
            )
            plan = TaskPlan(
                plan_id=plan_id,
                session_id=session_id,
                user_goal=user_goal,
                subtasks=[st1, st2],
                active_constraints=constraints,
                plan_status=TaskPlanStatus.READY,
            )
            self.memory.save_plan(plan)
            return plan

        # 2. Text Drafting & Iterative Transformation: e.g. "Draft text 'ChakrView Core' and convert it to uppercase"
        m_quotes = re.search(r"['\"]([^'\"]+)['\"]", user_goal)
        m_op = re.search(r"\b(uppercase|lowercase|strip|spaces|word\s+count)\b", q_low)
        if m_quotes and m_op:
            raw_text = m_quotes.group(1)
            op_raw = m_op.group(1)
            op = "strip" if "spaces" in op_raw else ("word_count" if "word" in op_raw else op_raw)
            st1 = Subtask(
                subtask_id="st_01",
                goal_description=f"Record draft text '{raw_text}'",
                action_type="reasoning",
                result_artifact=raw_text,
                dependencies=[],
                status=SubtaskStatus.COMPLETED,
            )
            st2 = Subtask(
                subtask_id="st_02",
                goal_description=f"Transform text to {op}",
                action_type="tool_call",
                tool_name="text_utility",
                tool_arguments={"operation": op},
                dependencies=["st_01"],
                input_bindings={"text": "st_01.result"},
                status=SubtaskStatus.READY,
            )
            plan = TaskPlan(
                plan_id=plan_id,
                session_id=session_id,
                user_goal=user_goal,
                subtasks=[st1, st2],
                active_constraints=constraints,
                plan_status=TaskPlanStatus.READY,
            )
            self.memory.save_plan(plan)
            return plan

        # 3. Missing Information / Ambiguous Constraints: "Calculate the total price but discount is missing"
        if "calculate" in q_low and "discount" in q_low and not re.search(r"\b\d+%\b|\b\d+\s+percent\b", q_low):
            st1 = Subtask(
                subtask_id="st_01",
                goal_description="Clarify discount percentage from user",
                action_type="clarification",
                dependencies=[],
                status=SubtaskStatus.AWAITING_CLARIFICATION,
                error_message="Discount percentage not specified in user query",
            )
            st2 = Subtask(
                subtask_id="st_02",
                goal_description="Compute discounted price",
                action_type="tool_call",
                tool_name="calculator",
                dependencies=["st_01"],
                status=SubtaskStatus.BLOCKED,
            )
            plan = TaskPlan(
                plan_id=plan_id,
                session_id=session_id,
                user_goal=user_goal,
                subtasks=[st1, st2],
                active_constraints=constraints,
                plan_status=TaskPlanStatus.AWAITING_CLARIFICATION,
            )
            self.memory.save_plan(plan)
            return plan

        # 4. Standard Single-Step / Fallback Task
        st1 = Subtask(
            subtask_id="st_01",
            goal_description=f"Process goal: {user_goal}",
            action_type="reasoning",
            status=SubtaskStatus.READY,
        )
        plan = TaskPlan(
            plan_id=plan_id,
            session_id=session_id,
            user_goal=user_goal,
            subtasks=[st1],
            active_constraints=constraints,
            plan_status=TaskPlanStatus.READY,
        )
        self.memory.save_plan(plan)
        return plan

    def execute_plan(self, session_id: str, plan_id: str) -> TaskPlan:
        """
        Executes subtasks in topological dependency order. Resolves dynamic bindings.
        """
        plan = self.memory.get_plan(session_id, plan_id)
        if not plan:
            raise ValueError(f"Plan '{plan_id}' not found in session '{session_id}'")

        if plan.plan_status == TaskPlanStatus.AWAITING_CLARIFICATION:
            return plan

        artifacts: Dict[str, Any] = {}
        # Pre-populate completed artifacts
        for st in plan.subtasks:
            if st.status == SubtaskStatus.COMPLETED and st.result_artifact is not None:
                artifacts[st.subtask_id] = st.result_artifact

        for st in plan.subtasks:
            # Check dependencies
            deps_met = all(
                dep_id in artifacts for dep_id in st.dependencies
            )
            if not deps_met and st.dependencies:
                st.status = SubtaskStatus.BLOCKED
                continue

            if st.status in (SubtaskStatus.PROPOSED, SubtaskStatus.READY, SubtaskStatus.BLOCKED):
                st.status = SubtaskStatus.RUNNING
                try:
                    # Resolve input bindings
                    resolved_args = dict(st.tool_arguments)
                    for arg_name, binding_expr in st.input_bindings.items():
                        # e.g., "st_01.result"
                        source_id = binding_expr.split(".")[0]
                        resolved_args[arg_name] = artifacts.get(source_id)

                    if st.action_type == "tool_call":
                        tool = self.tool_registry.get(st.tool_name)
                        if st.tool_name == "calculator" and "operator" in resolved_args:
                            base_val = resolved_args.get("base")
                            op = resolved_args.get("operator")
                            operand = resolved_args.get("operand")
                            full_expr = f"{base_val} {op} {operand}"
                            res = tool.execute(expression=full_expr)
                        elif st.tool_name == "calculator" and "expression" in resolved_args:
                            res = tool.execute(expression=resolved_args["expression"])
                        elif st.tool_name == "text_utility":
                            res = tool.execute(operation=resolved_args["operation"], text=resolved_args["text"])
                        else:
                            res = tool.execute(**resolved_args)

                        st.result_artifact = str(res.output)
                        st.evidence_provenance = f"Tool:{st.tool_name}"
                        st.status = SubtaskStatus.COMPLETED
                        st.completed_at = time.time()
                        artifacts[st.subtask_id] = st.result_artifact

                    elif st.action_type == "reasoning":
                        st.status = SubtaskStatus.COMPLETED
                        st.completed_at = time.time()
                        if st.result_artifact is not None:
                            artifacts[st.subtask_id] = st.result_artifact

                except Exception as e:
                    st.status = SubtaskStatus.FAILED
                    st.error_message = str(e)
                    plan.plan_status = TaskPlanStatus.FAILED
                    self.memory.save_plan(plan)
                    return plan

        all_completed = all(st.status == SubtaskStatus.COMPLETED for st in plan.subtasks)
        if all_completed:
            plan.plan_status = TaskPlanStatus.COMPLETED
        else:
            plan.plan_status = TaskPlanStatus.IN_PROGRESS

        self.memory.save_plan(plan)
        return plan

    def replan_on_constraint_change(
        self,
        session_id: str,
        active_plan_id: str,
        updated_constraints: Dict[str, Any],
        modified_subtask_id: Optional[str] = None,
        new_parameter: Optional[Tuple[str, Any]] = None,
    ) -> TaskPlan:
        """
        Invalidates dependent subtasks and creates a new version of the plan.
        Preserves un-invalidated upstream completed subtasks.
        """
        old_plan = self.memory.get_plan(session_id, active_plan_id)
        if not old_plan:
            raise ValueError(f"Active plan '{active_plan_id}' not found")

        old_plan.plan_status = TaskPlanStatus.SUPERSEDED
        new_plan_id = f"plan_{int(time.time() * 1000)}"
        old_plan.superseded_by = new_plan_id
        self.memory.save_plan(old_plan)

        new_subtasks: List[Subtask] = []
        # Find which subtasks need invalidation
        invalidated_ids = set()
        if modified_subtask_id:
            invalidated_ids.add(modified_subtask_id)

        # Propagate invalidation downstream
        for st in old_plan.subtasks:
            if any(dep in invalidated_ids for dep in st.dependencies):
                invalidated_ids.add(st.subtask_id)

        for st in old_plan.subtasks:
            if st.subtask_id in invalidated_ids:
                new_args = dict(st.tool_arguments)
                if new_parameter and st.subtask_id == modified_subtask_id:
                    new_args[new_parameter[0]] = new_parameter[1]

                cloned_st = Subtask(
                    subtask_id=st.subtask_id,
                    goal_description=st.goal_description,
                    action_type=st.action_type,
                    tool_name=st.tool_name,
                    tool_arguments=new_args,
                    dependencies=list(st.dependencies),
                    status=SubtaskStatus.READY if not st.dependencies else SubtaskStatus.PROPOSED,
                    input_bindings=dict(st.input_bindings),
                )
                new_subtasks.append(cloned_st)
            else:
                # Keep completed upstream subtask intact
                new_subtasks.append(
                    Subtask(
                        subtask_id=st.subtask_id,
                        goal_description=st.goal_description,
                        action_type=st.action_type,
                        tool_name=st.tool_name,
                        tool_arguments=dict(st.tool_arguments),
                        dependencies=list(st.dependencies),
                        status=st.status,
                        input_bindings=dict(st.input_bindings),
                        result_artifact=st.result_artifact,
                        evidence_provenance=st.evidence_provenance,
                        completed_at=st.completed_at,
                    )
                )

        merged_constraints = dict(old_plan.active_constraints)
        merged_constraints.update(updated_constraints)

        new_plan = TaskPlan(
            plan_id=new_plan_id,
            session_id=session_id,
            user_goal=old_plan.user_goal,
            subtasks=new_subtasks,
            active_constraints=merged_constraints,
            plan_status=TaskPlanStatus.READY,
            version=old_plan.version + 1,
        )
        self.memory.save_plan(new_plan)
        return new_plan
