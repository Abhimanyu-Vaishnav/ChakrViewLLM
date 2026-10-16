"""
Safe Arithmetic Tool and AST Evaluator for ChakrView.
Location: chakrview/runtime/safe_calculator.py

Implements:
- SafeCalculatorTool conforming to Tool interface.
- Pure in-memory AST evaluator.
- Strict node allowlist (Expression, Constant, BinOp, UnaryOp).
- Zero eval(), exec(), imports, attribute access, or process execution.
- Bounded expression lengths, finite float checks, and exponent magnitude limits.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass, asdict
import math
import operator
import re
import time
from typing import Any, Dict, Optional, Union

from chakrview.runtime.tools import Tool, ToolResult


@dataclass
class CalculationResult:
    expression: str
    normalized_expression: str
    success: bool
    result: Optional[Union[int, float]] = None
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    execution_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SafeCalculator(Tool):
    """
    Restricted mathematical evaluator using Python's standard `ast` module.
    """

    MAX_EXPR_LENGTH = 256
    MAX_EXPONENT = 100
    MAX_BASE_FOR_LARGE_EXP = 1e6

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
        return "safe_calculator"

    @property
    def name(self) -> str:
        return "Safe Restricted Arithmetic Calculator"

    @property
    def description(self) -> str:
        return "Evaluates pure deterministic arithmetic expressions with strict AST security limits."

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "expression": {
                "type": "string",
                "required": True,
                "description": "Mathematical expression (e.g., '147 + 258', '1024 / 8').",
            }
        }

    def _eval_ast(self, node: ast.AST) -> Union[int, float]:
        if isinstance(node, ast.Expression):
            return self._eval_ast(node.body)

        elif isinstance(node, ast.Constant):
            if isinstance(node.value, bool):
                raise ValueError("Security violation: Boolean constant is not allowed in arithmetic expression.")
            if isinstance(node.value, (int, float)):
                if not math.isfinite(node.value):
                    raise ValueError("Non-finite numeric constant is not allowed.")
                return node.value
            raise ValueError(f"Security violation: Disallowed constant type '{type(node.value).__name__}'.")

        elif isinstance(node, ast.UnaryOp):
            op_type = type(node.op)
            if op_type not in self._ALLOWED_OPERATORS:
                raise ValueError(f"Security violation: Disallowed unary operator '{op_type.__name__}'.")
            val = self._eval_ast(node.operand)
            return self._ALLOWED_OPERATORS[op_type](val)

        elif isinstance(node, ast.BinOp):
            op_type = type(node.op)
            if op_type not in self._ALLOWED_OPERATORS:
                raise ValueError(f"Security violation: Disallowed binary operator '{op_type.__name__}'.")

            left = self._eval_ast(node.left)
            right = self._eval_ast(node.right)

            if op_type is ast.Pow:
                if abs(right) > self.MAX_EXPONENT:
                    raise ValueError(f"Exponent magnitude {right} exceeds safety limit ({self.MAX_EXPONENT}).")
                if abs(left) > self.MAX_BASE_FOR_LARGE_EXP and right > 10:
                    raise ValueError("Base and exponent magnitude exceed computational limits.")

            if op_type in (ast.Div, ast.FloorDiv, ast.Mod):
                if right == 0:
                    raise ZeroDivisionError("Division or modulo by zero.")

            res = self._ALLOWED_OPERATORS[op_type](left, right)
            if isinstance(res, (int, float)) and not math.isfinite(res):
                raise ValueError("Calculation resulted in non-finite value (inf or nan).")
            return res

        else:
            raise ValueError(f"Security violation: Disallowed syntax node '{type(node).__name__}'.")

    def evaluate(self, expression: str) -> CalculationResult:
        t0 = time.perf_counter()
        if not expression or not isinstance(expression, str):
            return CalculationResult(
                expression=str(expression),
                normalized_expression="",
                success=False,
                error_code="INVALID_INPUT",
                error_message="Expression must be a non-empty string.",
                execution_time_ms=(time.perf_counter() - t0) * 1000.0,
            )

        expr_clean = expression.strip()
        if len(expr_clean) > self.MAX_EXPR_LENGTH:
            return CalculationResult(
                expression=expr_clean,
                normalized_expression="",
                success=False,
                error_code="EXPRESSION_TOO_LONG",
                error_message=f"Expression length {len(expr_clean)} exceeds maximum {self.MAX_EXPR_LENGTH}.",
                execution_time_ms=(time.perf_counter() - t0) * 1000.0,
            )

        # Normalize common natural-language math tokens (e.g. 'x' -> '*', '^' -> '**')
        # Only replace standalone 'x' or 'X' surrounded by numbers/whitespace
        normalized = re.sub(r"(?<=\d|\))\s*[xX]\s*(?=\d|\()", " * ", expr_clean)
        # Normalize '^' to '**' if used as power
        normalized = re.sub(r"\^", "**", normalized)

        try:
            parsed = ast.parse(normalized, mode="eval")
            val = self._eval_ast(parsed)

            # Format float nicely if it represents an exact integer (e.g. 25.0 -> 25)
            if isinstance(val, float) and val.is_integer():
                val = int(val)

            elapsed = (time.perf_counter() - t0) * 1000.0
            return CalculationResult(
                expression=expr_clean,
                normalized_expression=normalized,
                success=True,
                result=val,
                execution_time_ms=elapsed,
            )
        except ZeroDivisionError as e:
            elapsed = (time.perf_counter() - t0) * 1000.0
            return CalculationResult(
                expression=expr_clean,
                normalized_expression=normalized,
                success=False,
                error_code="DIVISION_BY_ZERO",
                error_message="Division or modulo by zero is undefined.",
                execution_time_ms=elapsed,
            )
        except SyntaxError as e:
            elapsed = (time.perf_counter() - t0) * 1000.0
            return CalculationResult(
                expression=expr_clean,
                normalized_expression=normalized,
                success=False,
                error_code="SYNTAX_ERROR",
                error_message=f"Malformed arithmetic syntax: {str(e)}",
                execution_time_ms=elapsed,
            )
        except Exception as e:
            elapsed = (time.perf_counter() - t0) * 1000.0
            return CalculationResult(
                expression=expr_clean,
                normalized_expression=normalized,
                success=False,
                error_code="SECURITY_OR_DOMAIN_VIOLATION",
                error_message=str(e),
                execution_time_ms=elapsed,
            )

    def execute(self, **kwargs: Any) -> ToolResult:
        expr = kwargs.get("expression", "")
        res = self.evaluate(expr)
        return ToolResult(
            tool_id=self.tool_id,
            success=res.success,
            output=res.result if res.success else None,
            error=res.error_message if not res.success else None,
            execution_time_ms=res.execution_time_ms,
        )
