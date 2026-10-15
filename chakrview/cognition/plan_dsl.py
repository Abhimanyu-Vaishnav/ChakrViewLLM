"""
Batches 746-750: Plan DSL Parser, Serializer, and Task Graph Validator.
Location: chakrview/cognition/plan_dsl.py

Features:
1. PlanDSLSerializer: Formats TaskPlan into compact, canonical DSL text.
2. PlanDSLParser: Parses raw generated DSL text into structured dicts/TaskPlan.
3. PlanDSLValidator: Enforces DAG acyclicity, tool authorization, argument syntax, and EOS markers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class ParsedStep:
    step_id: int
    action: str
    tool: str
    args: Dict[str, Any]
    deps: List[int]


@dataclass
class ParsedPlan:
    goal: str
    steps: List[ParsedStep]
    clarification_needed: bool
    is_valid: bool
    validation_error: Optional[str] = None
    has_eos: bool = False


class PlanDSLParser:
    """
    Parser for Plan DSL text emitted by the neural model.
    """

    @staticmethod
    def parse(dsl_text: str) -> ParsedPlan:
        text = dsl_text.strip()
        has_eos = "<END_PLAN>" in text

        # Extract Goal
        m_goal = re.search(r"^GOAL:\s*(\w+)", text, re.MULTILINE)
        goal = m_goal.group(1) if m_goal else "unknown"

        # Extract Clarification
        m_clar = re.search(r"^CLARIFICATION:\s*(NEEDED|NONE)", text, re.MULTILINE)
        clar_needed = (m_clar.group(1) == "NEEDED") if m_clar else False

        # Extract Steps
        # Pattern: STEP 0: tool_call | calculator | args={expr=12*4} | deps=[]
        step_pattern = re.compile(
            r"^STEP\s*(\d+)\s*:\s*(\w+)\s*\|\s*(\w+)\s*\|\s*args=\{([^}]*)\}\s*\|\s*deps=\[([^\]]*)\]",
            re.MULTILINE,
        )

        steps: List[ParsedStep] = []
        for match in step_pattern.finditer(text):
            step_id = int(match.group(1))
            action = match.group(2).strip()
            tool = match.group(3).strip()
            raw_args = match.group(4).strip()
            raw_deps = match.group(5).strip()

            # Parse args
            args_dict: Dict[str, Any] = {}
            if raw_args:
                for pair in raw_args.split(","):
                    if "=" in pair:
                        k, v = pair.split("=", 1)
                        args_dict[k.strip()] = v.strip().strip("'\"")

            # Parse deps
            deps_list: List[int] = []
            if raw_deps:
                for d in raw_deps.split(","):
                    d_clean = d.strip()
                    if d_clean.isdigit():
                        deps_list.append(int(d_clean))

            steps.append(
                ParsedStep(
                    step_id=step_id,
                    action=action,
                    tool=tool,
                    args=args_dict,
                    deps=deps_list,
                )
            )

        # Validation
        is_valid, err = PlanDSLValidator.validate(goal, steps, clar_needed, has_eos)

        return ParsedPlan(
            goal=goal,
            steps=steps,
            clarification_needed=clar_needed,
            is_valid=is_valid,
            validation_error=err,
            has_eos=has_eos,
        )


class PlanDSLValidator:
    """
    Validates structural rules of a parsed DSL plan.
    """

    ALLOWED_TOOLS = {"calculator", "text_utility", "none"}
    ALLOWED_ACTIONS = {"tool_call", "text_transform", "clarification", "reasoning"}

    @classmethod
    def validate(
        cls,
        goal: str,
        steps: List[ParsedStep],
        clarification_needed: bool,
        has_eos: bool,
    ) -> Tuple[bool, Optional[str]]:
        if not has_eos:
            return False, "Plan is missing terminal <END_PLAN> marker (truncated generation)"

        if not steps and not clarification_needed:
            return False, "Plan contains 0 steps and clarification is not flagged"

        if clarification_needed:
            return True, None

        for idx, st in enumerate(steps):
            if st.action not in cls.ALLOWED_ACTIONS:
                return False, f"Illegal action type '{st.action}' in step {st.step_id}"

            if st.tool not in cls.ALLOWED_TOOLS:
                return False, f"Unauthorized or unknown tool '{st.tool}' in step {st.step_id}"

            # Acyclicity check
            for dep in st.deps:
                if dep >= st.step_id:
                    return False, f"Acyclicity violation: step {st.step_id} depends on forward/self step {dep}"

        return True, None


class PlanDSLSerializer:
    """
    Serializes structured plan fixtures to canonical Plan DSL text.
    """

    @staticmethod
    def serialize(goal: str, steps: List[Dict[str, Any]], clarification: bool = False) -> str:
        lines = [f"GOAL: {goal}"]
        for idx, st in enumerate(steps):
            args_str = ",".join(f"{k}={v}" for k, v in st.get("args", {}).items())
            deps_str = ",".join(str(d) for d in st.get("deps", []))
            lines.append(
                f"STEP {idx}: {st['action']} | {st['tool']} | args={{{args_str}}} | deps=[{deps_str}]"
            )
        lines.append(f"CLARIFICATION: {'NEEDED' if clarification else 'NONE'}")
        lines.append("<END_PLAN>")
        return "\n".join(lines)
