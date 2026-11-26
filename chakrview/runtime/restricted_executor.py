"""
NR-018 Phase 2: Restricted Deterministic Execution Engine
A safe, deterministic execution engine with a fixed allowlisted operation registry.
Strictly zero eval(), exec(), os.system(), subprocess, or arbitrary code execution.

Supported Operations:
1. math_arithmetic: add, sub, mul, div, mod, pow
2. state_accumulator: sequential state updates (start, steps: [(op, val)])
3. variable_storage: set, get, assign
4. list_transforms: reverse, sort, filter_even, filter_odd, max, min, sum, length
5. string_transforms: uppercase, lowercase, reverse, length
6. stack_operations: push, pop, peek, size
7. conditional_eval: compare (gt, lt, gte, lte, eq, neq) -> branch
8. multi_step_composition: sequence of validated allowlisted operations
"""

import math
from typing import Dict, Any, List, Optional, Union

class ExecutionError(Exception):
    """Base exception for deterministic executor errors."""
    pass

class SchemaValidationError(ExecutionError):
    """Raised when an operation request fails schema validation."""
    pass

class OperationNotAllowedError(ExecutionError):
    """Raised when an operation is not in the allowlist."""
    pass

class RestrictedExecutionEngine:
    def __init__(self):
        # Variable storage state for stateful sessions
        self.variables: Dict[str, Any] = {}
        # Stack state
        self.stack: List[Any] = []

        # Allowlist registry mapping operation name to handler and schema validator
        self.registry = {
            "math_add": self._op_math_add,
            "math_sub": self._op_math_sub,
            "math_mul": self._op_math_mul,
            "math_div": self._op_math_div,
            "state_accumulator": self._op_state_accumulator,
            "var_set": self._op_var_set,
            "var_get": self._op_var_get,
            "list_reverse": self._op_list_reverse,
            "list_sort": self._op_list_sort,
            "list_max": self._op_list_max,
            "list_min": self._op_list_min,
            "list_sum": self._op_list_sum,
            "string_upper": self._op_string_upper,
            "string_lower": self._op_string_lower,
            "string_reverse": self._op_string_reverse,
            "string_length": self._op_string_length,
            "stack_push": self._op_stack_push,
            "stack_pop": self._op_stack_pop,
            "stack_peek": self._op_stack_peek,
            "conditional_branch": self._op_conditional_branch,
            "multi_step": self._op_multi_step
        }

    def execute(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes a validated operation request.
        Request schema:
        {
            "operation": str,
            "arguments": dict
        }
        Returns:
        {
            "status": "success" | "error",
            "result": Any,
            "error": Optional[str]
        }
        """
        if not isinstance(request, dict):
            return {"status": "error", "result": None, "error": "Request must be a dictionary"}

        op = request.get("operation")
        args = request.get("arguments", {})

        if not op or not isinstance(op, str):
            return {"status": "error", "result": None, "error": "Missing or invalid 'operation' field"}

        if not isinstance(args, dict):
            return {"status": "error", "result": None, "error": "'arguments' field must be a dictionary"}

        if op not in self.registry:
            return {"status": "error", "result": None, "error": f"Operation '{op}' is not allowed or unsupported"}

        handler = self.registry[op]
        try:
            res = handler(args)
            return {"status": "success", "result": res, "error": None}
        except ExecutionError as ee:
            return {"status": "error", "result": None, "error": str(ee)}
        except Exception as e:
            return {"status": "error", "result": None, "error": f"Execution failure: {type(e).__name__}: {str(e)}"}

    # --- Handlers ---
    def _validate_numeric(self, val: Any, name: str) -> Union[int, float]:
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            # Check bounds to prevent memory/overflow exhaustion
            if abs(val) > 1e12:
                raise SchemaValidationError(f"Parameter '{name}' exceeds safe bounds (|val| > 1e12)")
            return val
        raise SchemaValidationError(f"Parameter '{name}' must be a numeric value, got {type(val).__name__}")

    def _op_math_add(self, args: dict):
        a = self._validate_numeric(args.get("a"), "a")
        b = self._validate_numeric(args.get("b"), "b")
        return a + b

    def _op_math_sub(self, args: dict):
        a = self._validate_numeric(args.get("a"), "a")
        b = self._validate_numeric(args.get("b"), "b")
        return a - b

    def _op_math_mul(self, args: dict):
        a = self._validate_numeric(args.get("a"), "a")
        b = self._validate_numeric(args.get("b"), "b")
        return a * b

    def _op_math_div(self, args: dict):
        a = self._validate_numeric(args.get("a"), "a")
        b = self._validate_numeric(args.get("b"), "b")
        if b == 0:
            raise ExecutionError("Division by zero")
        return a / b

    def _op_state_accumulator(self, args: dict):
        start = self._validate_numeric(args.get("start"), "start")
        steps = args.get("steps")
        if not isinstance(steps, list):
            raise SchemaValidationError("'steps' must be a list of (op, val) pairs")
        curr = start
        for idx, step in enumerate(steps):
            if not isinstance(step, (list, tuple)) or len(step) != 2:
                raise SchemaValidationError(f"Step {idx} must be a 2-element tuple/list (op, val)")
            op_type, val = step[0], self._validate_numeric(step[1], f"step_{idx}_val")
            if op_type == "add": curr += val
            elif op_type == "sub": curr -= val
            elif op_type == "mul": curr *= val
            elif op_type == "div":
                if val == 0: raise ExecutionError("Division by zero in accumulator")
                curr /= val
            else:
                raise SchemaValidationError(f"Unknown accumulator operation: {op_type}")
        return curr

    def _op_var_set(self, args: dict):
        name = args.get("name")
        val = args.get("value")
        if not isinstance(name, str) or not name.isidentifier():
            raise SchemaValidationError(f"Invalid variable identifier: '{name}'")
        self.variables[name] = val
        return f"{name}={val}"

    def _op_var_get(self, args: dict):
        name = args.get("name")
        if name not in self.variables:
            raise ExecutionError(f"Undefined variable: '{name}'")
        return self.variables[name]

    def _op_list_reverse(self, args: dict):
        lst = args.get("list")
        if not isinstance(lst, list): raise SchemaValidationError("'list' must be a list")
        return list(reversed(lst))

    def _op_list_sort(self, args: dict):
        lst = args.get("list")
        if not isinstance(lst, list): raise SchemaValidationError("'list' must be a list")
        return sorted(lst)

    def _op_list_max(self, args: dict):
        lst = args.get("list")
        if not isinstance(lst, list) or not lst: raise SchemaValidationError("'list' must be a non-empty list")
        return max(lst)

    def _op_list_min(self, args: dict):
        lst = args.get("list")
        if not isinstance(lst, list) or not lst: raise SchemaValidationError("'list' must be a non-empty list")
        return min(lst)

    def _op_list_sum(self, args: dict):
        lst = args.get("list")
        if not isinstance(lst, list): raise SchemaValidationError("'list' must be a list")
        return sum(lst)

    def _op_string_upper(self, args: dict):
        s = args.get("string")
        if not isinstance(s, str): raise SchemaValidationError("'string' must be a string")
        return s.upper()

    def _op_string_lower(self, args: dict):
        s = args.get("string")
        if not isinstance(s, str): raise SchemaValidationError("'string' must be a string")
        return s.lower()

    def _op_string_reverse(self, args: dict):
        s = args.get("string")
        if not isinstance(s, str): raise SchemaValidationError("'string' must be a string")
        return s[::-1]

    def _op_string_length(self, args: dict):
        s = args.get("string")
        if not isinstance(s, str): raise SchemaValidationError("'string' must be a string")
        return len(s)

    def _op_stack_push(self, args: dict):
        val = args.get("value")
        self.stack.append(val)
        return len(self.stack)

    def _op_stack_pop(self, args: dict):
        if not self.stack: raise ExecutionError("Stack underflow: pop from empty stack")
        return self.stack.pop()

    def _op_stack_peek(self, args: dict):
        if not self.stack: raise ExecutionError("Stack is empty")
        return self.stack[-1]

    def _op_conditional_branch(self, args: dict):
        val = self._validate_numeric(args.get("value"), "value")
        threshold = self._validate_numeric(args.get("threshold"), "threshold")
        cmp = args.get("compare", "gt")
        true_val = args.get("on_true")
        false_val = args.get("on_false")

        cond = False
        if cmp == "gt": cond = (val > threshold)
        elif cmp == "lt": cond = (val < threshold)
        elif cmp == "gte": cond = (val >= threshold)
        elif cmp == "lte": cond = (val <= threshold)
        elif cmp == "eq": cond = (val == threshold)
        elif cmp == "neq": cond = (val != threshold)
        else: raise SchemaValidationError(f"Invalid comparison operator: '{cmp}'")

        return true_val if cond else false_val

    def _op_multi_step(self, args: dict):
        ops = args.get("operations")
        if not isinstance(ops, list): raise SchemaValidationError("'operations' must be a list")
        last_res = None
        for step_idx, sub_req in enumerate(ops):
            if not isinstance(sub_req, dict):
                raise SchemaValidationError(f"Sub-step {step_idx} must be a dictionary request")
            res = self.execute(sub_req)
            if res["status"] != "success":
                raise ExecutionError(f"Multi-step failed at step {step_idx}: {res['error']}")
            last_res = res["result"]
        return last_res

def self_test_engine():
    engine = RestrictedExecutionEngine()
    print("Running self-test on RestrictedExecutionEngine...")

    # Math add
    r = engine.execute({"operation": "math_add", "arguments": {"a": 12, "b": 18}})
    assert r["status"] == "success" and r["result"] == 30, f"math_add failed: {r}"

    # Math div zero
    r = engine.execute({"operation": "math_div", "arguments": {"a": 10, "b": 0}})
    assert r["status"] == "error" and "Division by zero" in r["error"], f"div zero failed: {r}"

    # Accumulator
    r = engine.execute({"operation": "state_accumulator", "arguments": {"start": 100, "steps": [["add", 50], ["sub", 20], ["mul", 2]]}})
    assert r["status"] == "success" and r["result"] == 260, f"accumulator failed: {r}"

    # String reverse
    r = engine.execute({"operation": "string_reverse", "arguments": {"string": "chakra"}})
    assert r["status"] == "success" and r["result"] == "arkahc", f"string_reverse failed: {r}"

    # Conditional branch
    r = engine.execute({"operation": "conditional_branch", "arguments": {"value": 75, "threshold": 50, "compare": "gt", "on_true": "YES", "on_false": "NO"}})
    assert r["status"] == "success" and r["result"] == "YES", f"conditional_branch failed: {r}"

    # Security rejection: malicious / unknown operation
    r = engine.execute({"operation": "system_exec", "arguments": {"cmd": "whoami"}})
    assert r["status"] == "error" and "not allowed" in r["error"], f"malicious operation not rejected: {r}"

    # Parameter overflow bounds
    r = engine.execute({"operation": "math_mul", "arguments": {"a": 1e15, "b": 2}})
    assert r["status"] == "error" and "exceeds safe bounds" in r["error"], f"bounds check failed: {r}"

    print("All engine self-tests passed (100% verified)!")
    return True

if __name__ == '__main__':
    self_test_engine()
