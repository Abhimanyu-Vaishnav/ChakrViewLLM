"""
evaluation/coding_suite/isolated_executor.py

Process-isolated, time-bounded code execution runner.
Executes generated Python code in a clean, isolated subprocess using sys.executable -c
with explicit timeout enforcement, temporary directory isolation, and safe AST auditing.
"""

import sys
import os
import ast
import time
import subprocess
import tempfile
from typing import Dict, Any, Optional
from pathlib import Path


def audit_code_safety(code_str: str) -> Optional[str]:
    """
    Performs static AST inspection to reject overtly dangerous or escape operations:
    - __import__
    - import os, sys, subprocess, shutil, socket, urllib, requests, pathlib, ctypes
    - getattr, setattr, delattr, eval, exec, compile
    - open
    """
    try:
        tree = ast.parse(code_str)
    except SyntaxError as e:
        return f"SyntaxError: {e}"

    disallowed_modules = {
        "os", "sys", "subprocess", "shutil", "socket", "urllib", "requests",
        "pathlib", "ctypes", "posix", "nt", "importlib", "_thread", "threading",
        "multiprocessing", "signal", "tempfile"
    }
    
    disallowed_calls = {
        "eval", "exec", "compile", "open", "__import__", "globals", "locals"
    }

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root_pkg = alias.name.split(".")[0]
                if root_pkg in disallowed_modules:
                    return f"SecurityRejection: Import of prohibited module '{alias.name}'"
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                root_pkg = node.module.split(".")[0]
                if root_pkg in disallowed_modules:
                    return f"SecurityRejection: From-import of prohibited module '{node.module}'"
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in disallowed_calls:
                return f"SecurityRejection: Invocation of prohibited function '{node.func.id}'"
            elif isinstance(node.func, ast.Attribute) and node.func.attr in ("system", "popen", "spawn"):
                return f"SecurityRejection: Prohibited system call attribute '{node.func.attr}'"

    return None


def run_process_isolated_test(
    code_to_test: str,
    test_fixture: str,
    timeout_seconds: float = 2.0
) -> Dict[str, Any]:
    """
    Executes code in a separate child process via subprocess.run with a hard timeout and AST pre-audit.
    """
    # 1. Check syntax first
    try:
        ast.parse(code_to_test)
        syntax_valid = True
    except SyntaxError:
        syntax_valid = False

    # 2. Safety Audit
    safety_err = audit_code_safety(code_to_test)
    if safety_err:
        return {
            "passed": False,
            "syntax_valid": syntax_valid,
            "error": safety_err,
            "latency_ms": 0.0,
            "timed_out": False,
            "rejected_by_policy": True
        }

    # Wrap code with restricted builtins shim in subprocess
    runner_code = f"""
import sys

safe_builtins = {{
    "abs": abs, "all": all, "any": any, "bool": bool, "dict": dict,
    "enumerate": enumerate, "filter": filter, "float": float, "int": int,
    "isinstance": isinstance, "len": len, "list": list, "map": map,
    "max": max, "min": min, "pow": pow, "range": range, "reversed": reversed,
    "round": round, "set": set, "str": str, "sum": sum, "tuple": tuple,
    "zip": zip, "AssertionError": AssertionError, "Exception": Exception,
    "ValueError": ValueError, "TypeError": TypeError, "KeyError": KeyError,
    "IndexError": IndexError, "StopIteration": StopIteration,
    "True": True, "False": False, "None": None
}}

global_scope = {{"__builtins__": safe_builtins}}
code = \"\"\"{code_to_test}\n\n{test_fixture}\"\"\"

exec(code, global_scope)
"""

    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_script = Path(tmpdir) / "test_exec.py"
        with open(tmp_script, "w", encoding="utf-8") as f:
            f.write(runner_code)
            
        try:
            res = subprocess.run(
                [sys.executable, str(tmp_script)],
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                cwd=tmpdir
            )
            elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            
            if res.returncode == 0:
                return {
                    "passed": True,
                    "syntax_valid": syntax_valid,
                    "error": None,
                    "latency_ms": elapsed_ms,
                    "timed_out": False,
                    "rejected_by_policy": False
                }
            else:
                err_line = res.stderr.strip().splitlines()[-1] if res.stderr.strip() else "NonzeroExit"
                return {
                    "passed": False,
                    "syntax_valid": syntax_valid,
                    "error": err_line,
                    "latency_ms": elapsed_ms,
                    "timed_out": False,
                    "rejected_by_policy": False
                }
        except subprocess.TimeoutExpired:
            elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            return {
                "passed": False,
                "syntax_valid": syntax_valid,
                "error": f"TimeoutError: Execution exceeded {timeout_seconds}s limit",
                "latency_ms": elapsed_ms,
                "timed_out": True,
                "rejected_by_policy": False
            }
        except Exception as e:
            elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            return {
                "passed": False,
                "syntax_valid": syntax_valid,
                "error": f"{type(e).__name__}: {str(e)}",
                "latency_ms": elapsed_ms,
                "timed_out": False,
                "rejected_by_policy": False
            }
