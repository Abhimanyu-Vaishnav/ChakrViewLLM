"""
Governed Tool Subsystem for ChakrView (Step 11).

Provides safe, deterministic, zero-OS-authority tool abstractions.
Enforces strict policy boundaries:
- Pure mathematical computation and text utility
- Zero filesystem write, shell execution, process spawning, or network access
- Governed execution lifecycle:
    SkillPolicy Permission Check -> Input Validation -> Safe Execution -> Structured Result
"""

from abc import ABC, abstractmethod
import ast
from dataclasses import dataclass, field, asdict
import math
import operator
import time
from typing import Dict, List, Optional, Any, Union, Set

from chakrview.runtime.skills import SkillPolicy


@dataclass
class ToolResult:
    """
    Structured outcome of a tool execution.
    
    Attributes:
        tool_id: Identifier of the executed tool.
        success: Whether execution completed without violation or error.
        output: Resulting computed output (if successful).
        error: Diagnostic error message (if failed).
        execution_time_ms: Wall-clock execution time in milliseconds.
    """
    tool_id: str
    success: bool
    output: Any = None
    error: Optional[str] = None
    execution_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToolResult":
        return cls(**data)


class Tool(ABC):
    """
    Abstract interface for safe governed tools.
    """

    @property
    @abstractmethod
    def tool_id(self) -> str:
        """Unique tool identifier string."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable tool name."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Functional description of the tool."""
        pass

    @property
    @abstractmethod
    def parameters_schema(self) -> Dict[str, Any]:
        """Schema describing expected parameter names, types, and descriptions."""
        pass

    @abstractmethod
    def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the tool with validated arguments."""
        pass


class CalculatorTool(Tool):
    """
    Deterministic mathematical evaluator using safe AST parsing.
    
    Security Contract:
    - Pure in-memory arithmetic
    - AST node whitelist: BinOp (+, -, *, /, //, %, **), UnaryOp (+, -), Constant/Num
    - Strictly rejects: Call, Name, Attribute, Import, Lambda, statements, builtins
    - Prevents exponentiation DoS and zero division
    """

    _ALLOWED_OPERATORS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    @property
    def tool_id(self) -> str:
        return "calculator"

    @property
    def name(self) -> str:
        return "Safe Mathematical Calculator"

    @property
    def description(self) -> str:
        return "Evaluates safe deterministic arithmetic expressions (+, -, *, /, //, %, **)."

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "expression": {
                "type": "string",
                "required": True,
                "description": "Mathematical expression string (e.g. '12 * (4 + 3) / 2').",
            }
        }

    def _eval_node(self, node: ast.AST) -> Union[int, float]:
        if isinstance(node, ast.Expression):
            return self._eval_node(node.body)

        elif isinstance(node, ast.Constant):  # Python 3.8+
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError(f"Disallowed constant type in expression: {type(node.value).__name__}")

        elif isinstance(node, ast.UnaryOp):
            op_type = type(node.op)
            if op_type not in self._ALLOWED_OPERATORS:
                raise ValueError(f"Disallowed unary operator: {op_type.__name__}")
            operand = self._eval_node(node.operand)
            return self._ALLOWED_OPERATORS[op_type](operand)

        elif isinstance(node, ast.BinOp):
            op_type = type(node.op)
            if op_type not in self._ALLOWED_OPERATORS:
                raise ValueError(f"Disallowed binary operator: {op_type.__name__}")

            left = self._eval_node(node.left)
            right = self._eval_node(node.right)

            # DoS Prevention: Limit exponentiation magnitude
            if op_type is ast.Pow:
                if abs(right) > 100:
                    raise ValueError(f"Exponent magnitude {right} exceeds safety limit (100).")
                if abs(left) > 1e6 and right > 10:
                    raise ValueError("Base and exponent magnitude exceed safety limits.")

            if op_type in (ast.Div, ast.FloorDiv, ast.Mod) and right == 0:
                raise ZeroDivisionError("Division or modulo by zero.")

            return self._ALLOWED_OPERATORS[op_type](left, right)

        else:
            raise ValueError(f"Security violation: Disallowed AST node type '{type(node).__name__}'.")

    def execute(self, **kwargs: Any) -> ToolResult:
        t0 = time.perf_counter()
        expr = kwargs.get("expression")
        if not expr or not isinstance(expr, str):
            return ToolResult(
                tool_id=self.tool_id,
                success=False,
                error="Parameter 'expression' must be a non-empty string.",
                execution_time_ms=(time.perf_counter() - t0) * 1000.0,
            )

        expr_clean = expr.strip()
        try:
            parsed = ast.parse(expr_clean, mode="eval")
            result = self._eval_node(parsed)
            # Format float nicely if it's an exact integer
            if isinstance(result, float) and result.is_integer():
                result = int(result)
            return ToolResult(
                tool_id=self.tool_id,
                success=True,
                output=result,
                execution_time_ms=(time.perf_counter() - t0) * 1000.0,
            )
        except Exception as e:
            return ToolResult(
                tool_id=self.tool_id,
                success=False,
                error=f"Calculation error: {type(e).__name__}: {str(e)}",
                execution_time_ms=(time.perf_counter() - t0) * 1000.0,
            )


class TextUtilityTool(Tool):
    """
    Safe string operations tool for text inspection and formatting.
    """

    @property
    def tool_id(self) -> str:
        return "text_utility"

    @property
    def name(self) -> str:
        return "Safe Text Utility"

    @property
    def description(self) -> str:
        return "Provides text metrics and basic safe string transformations."

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "operation": {
                "type": "string",
                "required": True,
                "description": "Operation name: 'word_count', 'char_count', 'line_count', 'uppercase', 'lowercase', 'strip'.",
            },
            "text": {
                "type": "string",
                "required": True,
                "description": "Input text string to process.",
            },
        }

    def execute(self, **kwargs: Any) -> ToolResult:
        t0 = time.perf_counter()
        op = kwargs.get("operation")
        text = kwargs.get("text")

        if text is None or not isinstance(text, str):
            return ToolResult(
                tool_id=self.tool_id,
                success=False,
                error="Parameter 'text' must be a string.",
                execution_time_ms=(time.perf_counter() - t0) * 1000.0,
            )

        if not op or not isinstance(op, str):
            return ToolResult(
                tool_id=self.tool_id,
                success=False,
                error="Parameter 'operation' must be a string.",
                execution_time_ms=(time.perf_counter() - t0) * 1000.0,
            )

        op = op.lower().strip()
        elapsed = lambda: (time.perf_counter() - t0) * 1000.0

        if op == "word_count":
            return ToolResult(self.tool_id, True, output=len(text.split()), execution_time_ms=elapsed())
        elif op == "char_count":
            return ToolResult(self.tool_id, True, output=len(text), execution_time_ms=elapsed())
        elif op == "line_count":
            return ToolResult(self.tool_id, True, output=len(text.splitlines()), execution_time_ms=elapsed())
        elif op == "uppercase":
            return ToolResult(self.tool_id, True, output=text.upper(), execution_time_ms=elapsed())
        elif op == "lowercase":
            return ToolResult(self.tool_id, True, output=text.lower(), execution_time_ms=elapsed())
        elif op == "strip":
            return ToolResult(self.tool_id, True, output=text.strip(), execution_time_ms=elapsed())
        else:
            return ToolResult(
                self.tool_id,
                False,
                error=f"Unsupported operation '{op}'. Supported: word_count, char_count, line_count, uppercase, lowercase, strip.",
                execution_time_ms=elapsed(),
            )


class ToolRegistry:
    """
    Registry for managing available governed tools.
    """

    def __init__(self) -> None:
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool, overwrite: bool = False) -> None:
        """Register a tool instance."""
        if tool.tool_id in self._tools and not overwrite:
            raise ValueError(f"Tool '{tool.tool_id}' already registered. Use overwrite=True to replace.")
        self._tools[tool.tool_id] = tool

    def get(self, tool_id: str) -> Optional[Tool]:
        """Retrieve tool by identifier."""
        return self._tools.get(tool_id)

    def has_tool(self, tool_id: str) -> bool:
        """Check if tool is registered."""
        return tool_id in self._tools

    def unregister(self, tool_id: str) -> bool:
        """Unregister a tool."""
        if tool_id in self._tools:
            del self._tools[tool_id]
            return True
        return False

    def list_tools(self) -> List[Tool]:
        """List all registered tools."""
        return list(self._tools.values())

    def count(self) -> int:
        """Total count of registered tools."""
        return len(self._tools)

    def clear(self) -> None:
        """Clear all registered tools."""
        self._tools.clear()


class ToolExecutor:
    """
    Governed tool execution coordinator.
    
    Enforces the security and policy boundary:
    1. Permission check: Tool must be present in active SkillPolicy.allowed_tools
    2. Registry lookup: Tool must exist in ToolRegistry
    3. Input validation: Required parameters must be present
    4. Execution: Isolated execution with structured ToolResult return
    """

    def __init__(self, registry: Optional[ToolRegistry] = None) -> None:
        self.registry = registry or get_standard_tool_registry()

    def execute(
        self,
        tool_id: str,
        arguments: Dict[str, Any],
        skill_policy: Optional[SkillPolicy] = None,
    ) -> ToolResult:
        """
        Execute tool under strict policy governance.
        """
        t0 = time.perf_counter()

        # 1. Permission check via SkillPolicy
        if skill_policy is not None:
            if tool_id not in skill_policy.allowed_tools:
                return ToolResult(
                    tool_id=tool_id,
                    success=False,
                    error=f"Permission Denied: Tool '{tool_id}' is not authorized by active SkillPolicy. "
                          f"Allowed tools: {skill_policy.allowed_tools}",
                    execution_time_ms=(time.perf_counter() - t0) * 1000.0,
                )

        # 2. Registry lookup
        tool = self.registry.get(tool_id)
        if tool is None:
            return ToolResult(
                tool_id=tool_id,
                success=False,
                error=f"Tool '{tool_id}' not found in registry.",
                execution_time_ms=(time.perf_counter() - t0) * 1000.0,
            )

        # 3. Input validation
        schema = tool.parameters_schema
        for param_name, param_info in schema.items():
            if param_info.get("required", False) and param_name not in arguments:
                return ToolResult(
                    tool_id=tool_id,
                    success=False,
                    error=f"Missing required parameter '{param_name}' for tool '{tool_id}'.",
                    execution_time_ms=(time.perf_counter() - t0) * 1000.0,
                )

        # 4. Safe execution
        try:
            return tool.execute(**arguments)
        except Exception as e:
            return ToolResult(
                tool_id=tool_id,
                success=False,
                error=f"Unhandled tool execution exception: {type(e).__name__}: {str(e)}",
                execution_time_ms=(time.perf_counter() - t0) * 1000.0,
            )


def get_standard_tool_registry() -> ToolRegistry:
    """Factory returning standard pre-populated tool registry."""
    reg = ToolRegistry()
    reg.register(CalculatorTool())
    reg.register(TextUtilityTool())
    return reg
