"""
Governed Tool Execution Gate for ChakrView Cognitive Subsystem (Step 15).

Enforces strict authority boundaries:
Agent -> Plan -> Skill -> Tool Request -> Policy Gate -> Safe Tool Execution -> Result -> Observation

Security Principle:
- DATA != INSTRUCTION
- MEMORY != AUTHORITY
- KNOWLEDGE != AUTHORITY
- SKILL != UNRESTRICTED AUTHORITY

Retrieved documents and conversational memory can NEVER grant or escalate tool permissions.
Only explicit runtime SkillPolicy / GovernedToolGate policies authorize tool calls.
"""

from dataclasses import dataclass, field
import time
from typing import Dict, List, Optional, Any, Set

from chakrview.runtime.skills import SkillPolicy, Skill
from chakrview.runtime.tools import Tool, ToolRegistry, ToolExecutor, ToolResult, get_standard_tool_registry
from chakrview.cognition.observation import StepObservation


class ToolAuthorizationError(PermissionError):
    """Raised when an unauthorized or unpermitted tool invocation is attempted."""
    pass


class ArgumentValidationError(ValueError):
    """Raised when tool arguments fail security or type sanitization."""
    pass


@dataclass
class ToolInvocationRequest:
    """
    Explicit separation of requested tool call and its provenance.
    """
    tool_id: str
    arguments: Dict[str, Any]
    caller_step_id: str
    authorized_by_skill_id: Optional[str] = None
    is_authorized: bool = False
    authorization_reason: str = ""


class GovernedToolGate:
    """
    Policy gate sitting between cognitive execution and underlying ToolExecutor.
    """

    # Dangerous parameter substrings / injection markers rejected outright
    DANGEROUS_PATTERNS = ["__import__", "eval(", "exec(", "subprocess", "os.system", "shutil", "socket"]

    def __init__(
        self,
        tool_executor: Optional[ToolExecutor] = None,
        tool_registry: Optional[ToolRegistry] = None,
    ) -> None:
        self.registry = tool_registry or get_standard_tool_registry()
        self.executor = tool_executor or ToolExecutor(self.registry)

    def validate_arguments(self, tool_id: str, arguments: Dict[str, Any]) -> None:
        """
        Sanitize arguments to ensure no code injection or dangerous payloads.
        """
        for k, v in arguments.items():
            if isinstance(v, str):
                for pat in self.DANGEROUS_PATTERNS:
                    if pat in v:
                        raise ArgumentValidationError(
                            f"Argument '{k}' contains prohibited dangerous pattern '{pat}'."
                        )

    def authorize(
        self,
        tool_id: str,
        active_skill_policy: Optional[SkillPolicy] = None,
        provenance_source: Optional[str] = None,
    ) -> bool:
        """
        Check whether tool_id is explicitly authorized by the active SkillPolicy.
        
        Strict rule: If the request claims authority from retrieved data or memory,
        that authority is explicitly REJECTED.
        """
        if provenance_source in ("retrieved_knowledge", "working_memory", "document"):
            raise ToolAuthorizationError(
                f"Authority denial: Retrieved content or memory ('{provenance_source}') "
                f"cannot grant tool execution authority for '{tool_id}'."
            )

        if active_skill_policy is None:
            raise ToolAuthorizationError(
                f"No active SkillPolicy provided; tool '{tool_id}' is unauthorized by default."
            )

        if tool_id not in active_skill_policy.allowed_tools:
            raise ToolAuthorizationError(
                f"Tool '{tool_id}' is not in the allowed_tools whitelist of active SkillPolicy. "
                f"Allowed: {active_skill_policy.allowed_tools}"
            )

        return True

    def execute_governed(
        self,
        step_id: str,
        tool_id: str,
        arguments: Dict[str, Any],
        active_skill: Optional[Skill] = None,
        provenance_source: Optional[str] = None,
    ) -> StepObservation:
        """
        Execute a tool through the full policy gate, returning a StepObservation.
        """
        t0 = time.perf_counter()
        policy = active_skill.policy if active_skill else None

        # 1. Authorize
        try:
            self.authorize(tool_id, policy, provenance_source=provenance_source)
        except ToolAuthorizationError as err:
            return StepObservation(
                step_id=step_id,
                success=False,
                error=str(err),
                execution_time_ms=(time.perf_counter() - t0) * 1000.0,
                provenance={"tool_id": tool_id, "gate_status": "DENIED"},
                verification_status=False,
                verification_notes="Tool execution denied by policy gate.",
            )

        # 2. Argument validation
        try:
            self.validate_arguments(tool_id, arguments)
        except ArgumentValidationError as err:
            return StepObservation(
                step_id=step_id,
                success=False,
                error=str(err),
                execution_time_ms=(time.perf_counter() - t0) * 1000.0,
                provenance={"tool_id": tool_id, "gate_status": "INVALID_ARGUMENTS"},
                verification_status=False,
                verification_notes="Tool arguments failed security sanitization.",
            )

        # 3. Execution via ToolExecutor
        tool_result: ToolResult = self.executor.execute(tool_id, arguments, policy)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        return StepObservation(
            step_id=step_id,
            success=tool_result.success,
            output=tool_result.output,
            error=tool_result.error,
            execution_time_ms=elapsed_ms,
            provenance={
                "tool_id": tool_id,
                "skill_id": active_skill.skill_id if active_skill else None,
                "gate_status": "ALLOWED",
            },
            verification_status=tool_result.success,
            verification_notes="Tool executed successfully." if tool_result.success else f"Tool error: {tool_result.error}",
        )
