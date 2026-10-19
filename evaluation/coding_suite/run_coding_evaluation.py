"""
evaluation/coding_suite/run_coding_evaluation.py

Sandboxed, time-limited coding capability evaluation runner.
Loads candidate or canonical checkpoint, evaluates on the versioned coding tasks,
safely executes test fixtures with resource limits (no filesystem/network access),
and records syntax validity, test pass rate, and exact match metrics.
"""

import sys
import os
import json
import time
import ast
import traceback
from pathlib import Path
from typing import Dict, Any, List

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from chakrview.runtime.neural_runtime_adapter import (
    NeuralRuntimeAdapter,
    CANONICAL_CHECKPOINTS_ALLOWLIST,
    DEFAULT_CHECKPOINT_KEY,
)

TASKS_PATH = Path(__file__).resolve().parent / "coding_tasks.json"


def check_python_syntax(code_str: str) -> bool:
    try:
        ast.parse(code_str)
        return True
    except SyntaxError:
        return False


def run_sandboxed_test(code_to_test: str, test_fixture: str, timeout_seconds: float = 1.0) -> Dict[str, Any]:
    """
    Safely executes generated code alongside test fixtures in an isolated namespace
    with restricted builtins (disallowing os, subprocess, open, network).
    """
    safe_builtins = {
        "abs": abs, "all": all, "any": any, "bool": bool, "dict": dict,
        "enumerate": enumerate, "filter": filter, "float": float, "int": int,
        "isinstance": isinstance, "len": len, "list": list, "map": map,
        "max": max, "min": min, "pow": pow, "range": range, "reversed": reversed,
        "round": round, "set": set, "str": str, "sum": sum, "tuple": tuple,
        "zip": zip, "AssertionError": AssertionError, "Exception": Exception,
        "True": True, "False": False, "None": None
    }
    
    combined_code = f"{code_to_test}\n\n{test_fixture}"
    local_scope = {}
    global_scope = {"__builtins__": safe_builtins}
    
    t0 = time.perf_counter()
    try:
        # Syntax check first
        compiled = compile(combined_code, "<sandboxed_eval>", "exec")
        exec(compiled, global_scope, local_scope)
        elapsed = (time.perf_counter() - t0) * 1000.0
        return {
            "passed": True,
            "syntax_valid": True,
            "error": None,
            "latency_ms": round(elapsed, 2)
        }
    except Exception as e:
        elapsed = (time.perf_counter() - t0) * 1000.0
        return {
            "passed": False,
            "syntax_valid": check_python_syntax(code_to_test),
            "error": f"{type(e).__name__}: {str(e)}",
            "latency_ms": round(elapsed, 2)
        }


from evaluation.coding_suite.isolated_executor import run_process_isolated_test


def evaluate_coding_suite(checkpoint_key: str = DEFAULT_CHECKPOINT_KEY) -> Dict[str, Any]:
    with open(TASKS_PATH, "r", encoding="utf-8") as f:
        suite = json.load(f)
        
    tasks = suite["tasks"]
    ckpt_path, ckpt_hash, ckpt_label = CANONICAL_CHECKPOINTS_ALLOWLIST[checkpoint_key]
    
    adapter = NeuralRuntimeAdapter(checkpoint_path=ckpt_path, max_new_tokens=48, decoding="greedy")
    adapter.initialize()
    
    results = {
        "suite_name": suite["suite_name"],
        "checkpoint_label": ckpt_label,
        "checkpoint_sha256": ckpt_hash,
        "total_tasks": len(tasks),
        "syntax_valid_count": 0,
        "test_passed_count": 0,
        "timeout_count": 0,
        "policy_rejected_count": 0,
        "task_results": []
    }
    
    print("=" * 65)
    print(f"RUNNING CODING EVALUATION: {ckpt_label}")
    print("=" * 65)
    
    for idx, task in enumerate(tasks):
        task_id = task["id"]
        category = task["category"]
        prompt = task["prompt"]
        test_code = task["test_code"]
        
        # Generation
        out = adapter.generate(prompt)
        gen_text = out["generated_text"]
        full_code = prompt + gen_text
        
        # Evaluate syntax and execution with process isolation
        is_syntax = check_python_syntax(full_code)
        exec_res = run_process_isolated_test(full_code, test_code, timeout_seconds=2.0)
        
        if is_syntax:
            results["syntax_valid_count"] += 1
        if exec_res["passed"]:
            results["test_passed_count"] += 1
        if exec_res.get("timed_out", False):
            results["timeout_count"] += 1
        if exec_res.get("rejected_by_policy", False):
            results["policy_rejected_count"] += 1
            
        task_res = {
            "id": task_id,
            "category": category,
            "prompt": prompt,
            "generated_text": gen_text,
            "syntax_valid": is_syntax,
            "test_passed": exec_res["passed"],
            "timed_out": exec_res.get("timed_out", False),
            "rejected_by_policy": exec_res.get("rejected_by_policy", False),
            "error": exec_res["error"],
            "exec_latency_ms": exec_res["latency_ms"]
        }
        results["task_results"].append(task_res)
        print(f"  [{idx+1}/{len(tasks)}] {task_id:<14} | Syntax: {str(is_syntax):<5} | Passed: {str(exec_res['passed']):<5} | Err: {exec_res['error']}")
        
    results["syntax_valid_rate"] = f"{results['syntax_valid_count']}/{len(tasks)} ({round(results['syntax_valid_count']/len(tasks)*100, 1)}%)"
    results["test_pass_rate"] = f"{results['test_passed_count']}/{len(tasks)} ({round(results['test_passed_count']/len(tasks)*100, 1)}%)"
    results["timeout_rate"] = f"{results['timeout_count']}/{len(tasks)} ({round(results['timeout_count']/len(tasks)*100, 1)}%)"
    results["rejection_rate"] = f"{results['policy_rejected_count']}/{len(tasks)} ({round(results['policy_rejected_count']/len(tasks)*100, 1)}%)"
    
    out_file = Path(__file__).resolve().parent / "coding_baseline_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        
    print("=" * 65)
    print(f"SUMMARY: Syntax Valid: {results['syntax_valid_rate']} | Test Pass Rate: {results['test_pass_rate']}")
    print(f"Saved baseline report to: {out_file}")
    print("=" * 65)
    return results


if __name__ == "__main__":
    evaluate_coding_suite()
