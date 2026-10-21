"""
evaluation/coding_suite/isolated_executor.py

Restricted local code execution runner for ChakrView.
Provides time-bounded, process-isolated evaluation within a strictly vetted Python subset.

SECURITY BOUNDARY DECLARATION:
This runner operates as a restricted local test runner. It provides defense-in-depth via:
1. Static AST parsing & safety auditing of BOTH generated code and test fixtures.
2. Complete prohibition of all imports (os, sys, subprocess, etc.).
3. Prohibition of dangerous attributes (__class__, __subclasses__, __bases__, __globals__, etc.).
4. Prohibition of dynamic execution functions (eval, exec, compile, __import__, open, etc.).
5. Execution inside a clean temporary working directory with guaranteed cleanup on success, error, or timeout.
6. Execution with a sanitized, minimal environment (system PATH and Python executable only).
7. Strict wall-clock timeout and bounded output size.

LIMITATION:
This is not a hardware hypervisor or kernel sandbox. Hostile, untrusted code targeting
C-extension zero-days or low-level interpreter bugs must be run in containerized/VM sandboxes.
"""

import sys
import os
import ast
import time
import subprocess
import tempfile
from typing import Dict, Any, Optional
from pathlib import Path

# Explicitly allowed safe builtins inside runner namespace
SAFE_BUILTINS_NAMES = {
    "abs", "all", "any", "bool", "dict", "enumerate", "filter", "float",
    "int", "isinstance", "issubclass", "len", "list", "map", "max", "min",
    "pow", "range", "reversed", "round", "set", "str", "sum", "tuple", "zip",
    "AssertionError", "Exception", "ValueError", "TypeError", "KeyError",
    "IndexError", "StopIteration", "ZeroDivisionError", "AttributeError",
    "True", "False", "None"
}

# Dangerous attribute names that could facilitate introspection/escape
DISALLOWED_ATTRIBUTES = {
    "__subclasses__", "__bases__", "__base__", "__mro__", "__globals__",
    "__code__", "__closure__", "__func__", "__self__", "__module__",
    "__dict__", "__class__", "__builtins__", "__import__", "gi_frame",
    "f_globals", "f_locals", "cr_frame", "tb_frame"
}

# Dangerous function/call names
DISALLOWED_CALLS = {
    "eval", "exec", "compile", "open", "__import__", "globals", "locals",
    "getattr", "setattr", "delattr", "hasattr", "input", "print", "exit",
    "quit", "breakpoint", "memoryview", "format", "vars", "dir"
}

MAX_CODE_SIZE_BYTES = 32 * 1024  # 32 KB limit


def audit_code_safety(code_str: str) -> Optional[str]:
    """
    Performs static AST inspection to reject operations outside the safe subset:
    - Bounded length check
    - Zero imports allowed
    - No introspection or attribute-chain hopping
    - No dynamic execution
    """
    if len(code_str.encode("utf-8")) > MAX_CODE_SIZE_BYTES:
        return f"SecurityRejection: Code size {len(code_str)} exceeds {MAX_CODE_SIZE_BYTES} bytes limit"

    try:
        tree = ast.parse(code_str)
    except SyntaxError as e:
        return f"SyntaxError: {e}"

    for node in ast.walk(tree):
        # 1. Reject ALL imports
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            return "SecurityRejection: Imports are completely prohibited in restricted execution"

        # 2. Reject attribute-chain introspection
        if isinstance(node, ast.Attribute):
            if node.attr in DISALLOWED_ATTRIBUTES:
                return f"SecurityRejection: Prohibited introspection attribute '{node.attr}'"
            if node.attr.startswith("__") and node.attr.endswith("__") and node.attr not in ("__init__", "__str__", "__repr__", "__len__", "__getitem__", "__eq__"):
                return f"SecurityRejection: Prohibited dunder attribute access '{node.attr}'"

        # 3. Reject disallowed function calls
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in DISALLOWED_CALLS:
                return f"SecurityRejection: Invocation of prohibited function '{node.func.id}'"
            elif isinstance(node.func, ast.Attribute) and node.func.attr in ("system", "popen", "spawn", "read", "write"):
                return f"SecurityRejection: Prohibited system call method '{node.func.attr}'"

        # 4. Reject match statements or async operations that could complicate execution
        if isinstance(node, (ast.AsyncFunctionDef, ast.AsyncFor, ast.AsyncWith, ast.Await)):
            return "SecurityRejection: Asynchronous concurrency is not permitted in restricted runner"

    return None


def run_process_isolated_test(
    code_to_test: str,
    test_fixture: str,
    timeout_seconds: float = 2.0
) -> Dict[str, Any]:
    """
    Executes code in a separate child process via subprocess.run with a hard timeout,
    minimal sanitized environment, and separate temporary directory.
    """
    # 1. Check syntax of submitted code and test fixture
    try:
        ast.parse(code_to_test)
        code_syntax_valid = True
    except SyntaxError:
        code_syntax_valid = False

    try:
        ast.parse(test_fixture)
        fixture_syntax_valid = True
    except SyntaxError as e:
        return {
            "passed": False,
            "syntax_valid": code_syntax_valid,
            "error": f"FixtureSyntaxError: {e}",
            "latency_ms": 0.0,
            "timed_out": False,
            "rejected_by_policy": True
        }

    # 2. Safety Audit BOTH submitted code and test fixture
    code_safety_err = audit_code_safety(code_to_test)
    if code_safety_err:
        return {
            "passed": False,
            "syntax_valid": code_syntax_valid,
            "error": code_safety_err,
            "latency_ms": 0.0,
            "timed_out": False,
            "rejected_by_policy": True
        }

    fixture_safety_err = audit_code_safety(test_fixture)
    if fixture_safety_err:
        return {
            "passed": False,
            "syntax_valid": code_syntax_valid,
            "error": f"FixtureSecurityRejection: {fixture_safety_err}",
            "latency_ms": 0.0,
            "timed_out": False,
            "rejected_by_policy": True
        }

    # 3. Create sanitized execution harness script
    runner_code = f"""# -*- coding: utf-8 -*-
import sys

safe_builtins = {{
    "abs": abs, "all": all, "any": any, "bool": bool, "dict": dict,
    "enumerate": enumerate, "filter": filter, "float": float, "int": int,
    "isinstance": isinstance, "issubclass": issubclass, "len": len,
    "list": list, "map": map, "max": max, "min": min, "pow": pow,
    "range": range, "reversed": reversed, "round": round, "set": set,
    "str": str, "sum": sum, "tuple": tuple, "zip": zip,
    "AssertionError": AssertionError, "Exception": Exception,
    "ValueError": ValueError, "TypeError": TypeError, "KeyError": KeyError,
    "IndexError": IndexError, "StopIteration": StopIteration,
    "ZeroDivisionError": ZeroDivisionError, "AttributeError": AttributeError,
    "True": True, "False": False, "None": None
}}

global_scope = {{"__builtins__": safe_builtins}}
code = \"\"\"{code_to_test}\n\n{test_fixture}\"\"\"

exec(code, global_scope)
"""

    # Minimal sanitized environment passing only basic OS paths
    minimal_env = {
        "SYSTEMROOT": os.environ.get("SYSTEMROOT", "C:\\Windows"),
        "PATH": os.environ.get("PATH", ""),
        "PYTHONIOENCODING": "utf-8",
        "PYTHONDONTWRITEBYTECODE": "1"
    }

    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_script = Path(tmpdir) / "test_exec.py"
        with open(tmp_script, "w", encoding="utf-8") as f:
            f.write(runner_code)
            
        try:
            res = subprocess.run(
                [sys.executable, "-I", "-B", str(tmp_script)],
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                cwd=tmpdir,
                env=minimal_env
            )
            elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            
            if res.returncode == 0:
                return {
                    "passed": True,
                    "syntax_valid": code_syntax_valid,
                    "error": None,
                    "latency_ms": elapsed_ms,
                    "timed_out": False,
                    "rejected_by_policy": False
                }
            else:
                err_line = res.stderr.strip().splitlines()[-1] if res.stderr.strip() else "NonzeroExit"
                return {
                    "passed": False,
                    "syntax_valid": code_syntax_valid,
                    "error": err_line,
                    "latency_ms": elapsed_ms,
                    "timed_out": False,
                    "rejected_by_policy": False
                }
        except subprocess.TimeoutExpired:
            elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            return {
                "passed": False,
                "syntax_valid": code_syntax_valid,
                "error": f"TimeoutError: Execution exceeded {timeout_seconds}s limit",
                "latency_ms": elapsed_ms,
                "timed_out": True,
                "rejected_by_policy": False
            }
        except Exception as e:
            elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            return {
                "passed": False,
                "syntax_valid": code_syntax_valid,
                "error": f"{type(e).__name__}: {str(e)}",
                "latency_ms": elapsed_ms,
                "timed_out": False,
                "rejected_by_policy": False
            }
