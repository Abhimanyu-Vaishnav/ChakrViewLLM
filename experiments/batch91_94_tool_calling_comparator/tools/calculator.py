"""
Batch 91: Explicit Calculator Tool Protocol, Runtime & Security Boundary.

Location: experiments/batch91_94_tool_calling_comparator/tools/calculator.py

Features:
- Dedicated safe deterministic calculator (zero eval/exec, zero arbitrary code execution).
- Strict JSON tool-call schema validation:
  Schema:
  {
      "tool": "calculator",
      "op": "add" | "sub" | "mul",
      "a": int (-10000 <= a <= 10000),
      "b": int (-10000 <= b <= 10000)
  }
- Explicit failure classification:
  * MALFORMED_JSON
  * MISSING_FIELD
  * UNSUPPORTED_TOOL
  * UNSUPPORTED_OPERATION
  * OPERAND_TYPE_ERROR
  * OPERAND_BOUNDS_ERROR
- Multi-turn tool execution harness:
  1. Parses raw model generation for tool call pattern:
     ```
     <|im_start|>assistant
     TOOL_CALL: {"tool": "calculator", "op": "add", "a": 3, "b": 4}<|im_end|>
     ```
  2. Validates schema and executes calculator deterministically.
  3. Formulates tool response message:
     ```
     <|im_start|>tool
     {"result": 7, "status": "SUCCESS"}<|im_end|>
     ```
  4. Returns structured audit trace.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional, Tuple, Union


SUPPORTED_TOOLS = {"calculator"}
SUPPORTED_OPERATIONS = {"add", "sub", "mul"}
MAX_OPERAND_BOUND = 10000
MIN_OPERAND_BOUND = -10000


class ToolExecutionError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def execute_safe_calculator(op: str, a: int, b: int) -> int:
    """Pure safe deterministic arithmetic (zero eval, zero subprocess)."""
    if op == "add":
        return a + b
    elif op == "sub":
        return a - b
    elif op == "mul":
        return a * b
    else:
        raise ToolExecutionError("UNSUPPORTED_OPERATION", f"Operation '{op}' is not supported.")


def validate_tool_call_payload(payload: Any) -> Dict[str, Any]:
    """Validates tool call dictionary against schema without executing."""
    if not isinstance(payload, dict):
        return {"valid": False, "error_code": "MALFORMED_PAYLOAD", "error": "Payload must be a dictionary."}
    tool_name = payload.get("tool")
    if not tool_name:
        return {"valid": False, "error_code": "MISSING_FIELD", "error": "Missing 'tool'"}
    if tool_name not in SUPPORTED_TOOLS:
        return {"valid": False, "error_code": "UNSUPPORTED_TOOL", "error": f"Tool '{tool_name}' unsupported"}
    op = payload.get("op")
    if not op:
        return {"valid": False, "error_code": "MISSING_FIELD", "error": "Missing 'op'"}
    if op not in SUPPORTED_OPERATIONS:
        return {"valid": False, "error_code": "UNSUPPORTED_OPERATION", "error": f"Operation '{op}' unsupported"}
    if "a" not in payload or "b" not in payload:
        return {"valid": False, "error_code": "MISSING_FIELD", "error": "Missing operands"}
    a, b = payload.get("a"), payload.get("b")
    if not isinstance(a, int) or isinstance(a, bool) or not isinstance(b, int) or isinstance(b, bool):
        return {"valid": False, "error_code": "OPERAND_TYPE_ERROR", "error": "Operands must be integers"}
    if not (MIN_OPERAND_BOUND <= a <= MAX_OPERAND_BOUND and MIN_OPERAND_BOUND <= b <= MAX_OPERAND_BOUND):
        return {"valid": False, "error_code": "OPERAND_BOUNDS_ERROR", "error": "Operands out of bounds"}
    return {"valid": True, "tool": tool_name, "op": op, "a": a, "b": b}


def validate_and_execute_tool_call(payload: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Parses, validates, and executes a candidate tool call payload.
    Returns structured execution record with explicit status codes.
    """
    if isinstance(payload, str):
        # Extract JSON substring if wrapped with prefix
        clean = payload.strip()
        m = re.search(r"TOOL_CALL:\s*(\{.*\})", clean, re.DOTALL)
        if m:
            clean = m.group(1).strip()
        else:
            # Try finding direct JSON object
            m2 = re.search(r"(\{.*\})", clean, re.DOTALL)
            if m2:
                clean = m2.group(1).strip()
            else:
                return {
                    "status": "MALFORMED_JSON",
                    "error": "No JSON payload found in candidate text.",
                    "result": None,
                    "executed": False,
                }
        try:
            data = json.loads(clean)
        except Exception as e:
            return {
                "status": "MALFORMED_JSON",
                "error": f"JSON parsing failed: {e}",
                "result": None,
                "executed": False,
            }
    else:
        data = payload

    if not isinstance(data, dict):
        return {
            "status": "MALFORMED_PAYLOAD",
            "error": "Tool call payload must be a JSON object.",
            "result": None,
            "executed": False,
        }

    # 1. Validate tool name
    tool_name = data.get("tool")
    if not tool_name:
        return {
            "status": "MISSING_FIELD",
            "error": "Missing required field 'tool'.",
            "result": None,
            "executed": False,
        }
    if tool_name not in SUPPORTED_TOOLS:
        return {
            "status": "UNSUPPORTED_TOOL",
            "error": f"Tool '{tool_name}' is not in allowlist {list(SUPPORTED_TOOLS)}.",
            "result": None,
            "executed": False,
        }

    # 2. Validate operation
    op = data.get("op")
    if not op:
        return {
            "status": "MISSING_FIELD",
            "error": "Missing required field 'op'.",
            "result": None,
            "executed": False,
        }
    if op not in SUPPORTED_OPERATIONS:
        return {
            "status": "UNSUPPORTED_OPERATION",
            "error": f"Operation '{op}' is not supported.",
            "result": None,
            "executed": False,
        }

    # 3. Validate operands
    if "a" not in data or "b" not in data:
        return {
            "status": "MISSING_FIELD",
            "error": "Missing operands 'a' or 'b'.",
            "result": None,
            "executed": False,
        }

    a = data.get("a")
    b = data.get("b")
    if not isinstance(a, int) or isinstance(a, bool) or not isinstance(b, int) or isinstance(b, bool):
        return {
            "status": "OPERAND_TYPE_ERROR",
            "error": f"Operands 'a' and 'b' must be integers, got a={type(a)}, b={type(b)}.",
            "result": None,
            "executed": False,
        }

    if not (MIN_OPERAND_BOUND <= a <= MAX_OPERAND_BOUND and MIN_OPERAND_BOUND <= b <= MAX_OPERAND_BOUND):
        return {
            "status": "OPERAND_BOUNDS_ERROR",
            "error": f"Operands out of allowed range [{MIN_OPERAND_BOUND}, {MAX_OPERAND_BOUND}].",
            "result": None,
            "executed": False,
        }

    # 4. Safe Execution
    try:
        result = execute_safe_calculator(op, a, b)
        return {
            "status": "SUCCESS",
            "tool": tool_name,
            "op": op,
            "a": a,
            "b": b,
            "result": result,
            "executed": True,
            "provenance": "DETERMINISTIC_SAFE_CALCULATOR",
        }
    except Exception as e:
        return {
            "status": "EXECUTION_ERROR",
            "error": str(e),
            "result": None,
            "executed": False,
        }
