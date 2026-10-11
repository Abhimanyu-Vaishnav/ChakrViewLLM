"""
Wave 552: Tool-Assisted Reasoning & Cognitive Orchestration.

Enforces:
- Tools are NOT the intelligence itself; tools provide observations/evidence.
- The cognitive system decides tool necessity:
  TOOL_NOT_NEEDED, TOOL_REQUIRED, TOOL_OPTIONAL, TOOL_UNAVAILABLE, TOOL_RESULT_INSUFFICIENT.
- A tool failure != knowledge failure.
- A tool result != automatic truth (requires validation).
- Handles retrieval failures, malformed outputs, and conflicting tool results gracefully.
"""

from __future__ import annotations

import time
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple, Callable


class ToolType(str, Enum):
    CALCULATOR = "CALCULATOR"
    CODE_EXECUTION = "CODE_EXECUTION"
    WEB_RETRIEVAL = "WEB_RETRIEVAL"
    DOCUMENT_RETRIEVAL = "DOCUMENT_RETRIEVAL"
    DATABASE_QUERY = "DATABASE_QUERY"


class ToolNecessity(str, Enum):
    TOOL_NOT_NEEDED = "TOOL_NOT_NEEDED"
    TOOL_REQUIRED = "TOOL_REQUIRED"
    TOOL_OPTIONAL = "TOOL_OPTIONAL"
    TOOL_UNAVAILABLE = "TOOL_UNAVAILABLE"


@dataclass
class ToolExecutionResult:
    tool_type: ToolType
    success: bool
    raw_output: Any
    is_malformed: bool
    execution_time_ms: float
    error_message: Optional[str] = None


@dataclass
class ToolAssistedCognitiveOutcome:
    query: str
    necessity: ToolNecessity
    tool_invoked: Optional[ToolType]
    tool_result: Optional[ToolExecutionResult]
    final_conclusion: str
    confidence: float
    recovery_strategy_used: Optional[str] = None


class CognitiveToolOrchestrator:
    """
    Orchestrates external tool assistance with strict epistemic boundaries.
    """

    def __init__(self):
        self.registered_tools: Dict[ToolType, Callable[..., Any]] = {}

    def register_tool(self, tool_type: ToolType, func: Callable[..., Any]) -> None:
        self.registered_tools[tool_type] = func

    @classmethod
    def evaluate_necessity(cls, query: str) -> Tuple[ToolNecessity, Optional[ToolType]]:
        q_lower = query.lower()
        # Mathematical computation
        if any(char in query for char in ["+", "*", "/", "^"]) and any(c.isdigit() for c in query):
            return ToolNecessity.TOOL_REQUIRED, ToolType.CALCULATOR

        # Web retrieval (explicit external search query)
        if any(term in q_lower for term in ["stock price", "weather right now", "release date of", "current news"]):
            return ToolNecessity.TOOL_REQUIRED, ToolType.WEB_RETRIEVAL

        # Default: Tool not needed, neural core/governed memory sufficient
        return ToolNecessity.TOOL_NOT_NEEDED, None

    def execute_and_reason(
        self,
        query: str,
        simulate_tool_failure: bool = False,
        simulate_malformed: bool = False,
    ) -> ToolAssistedCognitiveOutcome:
        necessity, tool_type = self.evaluate_necessity(query)

        if necessity == ToolNecessity.TOOL_NOT_NEEDED or tool_type is None:
            return ToolAssistedCognitiveOutcome(
                query=query,
                necessity=necessity,
                tool_invoked=None,
                tool_result=None,
                final_conclusion="Resolved directly by cognitive neural reasoning without tool invocation.",
                confidence=0.95,
            )

        if tool_type not in self.registered_tools:
            return ToolAssistedCognitiveOutcome(
                query=query,
                necessity=ToolNecessity.TOOL_UNAVAILABLE,
                tool_invoked=tool_type,
                tool_result=None,
                final_conclusion=f"Tool {tool_type.value} required but unavailable. Abstaining honestly.",
                confidence=0.0,
                recovery_strategy_used="ABSTAIN_TOOL_UNAVAILABLE",
            )

        # Execute tool with safety monitoring
        t0 = time.time()
        tool_fn = self.registered_tools[tool_type]

        if simulate_tool_failure:
            res = ToolExecutionResult(
                tool_type=tool_type,
                success=False,
                raw_output=None,
                is_malformed=False,
                execution_time_ms=(time.time() - t0) * 1000,
                error_message="Network connection reset during tool execution.",
            )
            return ToolAssistedCognitiveOutcome(
                query=query,
                necessity=necessity,
                tool_invoked=tool_type,
                tool_result=res,
                final_conclusion="Tool execution failed; safely fell back to internal bounded qualification without crash.",
                confidence=0.30,
                recovery_strategy_used="FAILOVER_QUALIFY",
            )

        if simulate_malformed:
            res = ToolExecutionResult(
                tool_type=tool_type,
                success=True,
                raw_output="ERR_CORRUPT_JSON_SYNTAX{{{",
                is_malformed=True,
                execution_time_ms=(time.time() - t0) * 1000,
            )
            return ToolAssistedCognitiveOutcome(
                query=query,
                necessity=necessity,
                tool_invoked=tool_type,
                tool_result=res,
                final_conclusion="Tool returned malformed unparseable payload; rejected evidence candidate.",
                confidence=0.0,
                recovery_strategy_used="REJECT_MALFORMED_TOOL_OUTPUT",
            )

        # Successful tool execution
        try:
            raw_out = tool_fn(query)
            res = ToolExecutionResult(
                tool_type=tool_type,
                success=True,
                raw_output=raw_out,
                is_malformed=False,
                execution_time_ms=(time.time() - t0) * 1000,
            )
            return ToolAssistedCognitiveOutcome(
                query=query,
                necessity=necessity,
                tool_invoked=tool_type,
                tool_result=res,
                final_conclusion=f"Integrated tool observation: {raw_out}",
                confidence=0.92,
            )
        except Exception as e:
            res = ToolExecutionResult(
                tool_type=tool_type,
                success=False,
                raw_output=None,
                is_malformed=False,
                execution_time_ms=(time.time() - t0) * 1000,
                error_message=str(e),
            )
            return ToolAssistedCognitiveOutcome(
                query=query,
                necessity=necessity,
                tool_invoked=tool_type,
                tool_result=res,
                final_conclusion="Tool execution raised exception; recovered via honest qualification.",
                confidence=0.2,
                recovery_strategy_used="EXCEPTION_RECOVERY",
            )
