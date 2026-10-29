"""
evaluation/coding_suite/run_coding_evaluation.py

Sandboxed, time-limited coding capability evaluation runner.
Loads candidate or canonical checkpoint, evaluates on the versioned coding tasks
across all 6 core software development categories, safely executes test fixtures
with resource limits, and records syntax validity, test pass rate, exact/substring match,
empty/malformed status, generation speed, and error diagnostics.
"""

import sys
import os
import json
import time
import ast
import platform
import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from chakrview.runtime.neural_runtime_adapter import (
    NeuralRuntimeAdapter,
    CANONICAL_CHECKPOINTS_ALLOWLIST,
    DEFAULT_CHECKPOINT_KEY,
)
from evaluation.coding_suite.isolated_executor import run_process_isolated_test

TASKS_PATH = Path(__file__).resolve().parent / "coding_tasks.json"
DEFAULT_OUTPUT_PATH = Path(__file__).resolve().parent / "coding_baseline_results.json"


def check_python_syntax(code_str: str) -> bool:
    try:
        ast.parse(code_str)
        return True
    except (SyntaxError, ValueError, TypeError):
        return False


def evaluate_coding_suite(
    checkpoint_key: str = DEFAULT_CHECKPOINT_KEY,
    max_new_tokens: int = 48,
    decoding: str = "greedy",
    output_path: Optional[str] = None,
    smoke_mode: bool = False,
    mock_generator: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Runs the standardized coding evaluation benchmark.
    
    Supports:
    - Real neural model evaluation (when canonical weights exist).
    - Deterministic smoke mode (verifies task pipeline without neural weights).
    - Custom output path so historical baseline files are never silently overwritten.
    """
    with open(TASKS_PATH, "r", encoding="utf-8") as f:
        suite = json.load(f)
        
    tasks = suite["tasks"]
    ckpt_path, ckpt_hash, ckpt_label = CANONICAL_CHECKPOINTS_ALLOWLIST.get(
        checkpoint_key, (None, "unknown", "Unknown Checkpoint")
    )
    # Check external/source fallback if target path is absent
    if (ckpt_path is None or not Path(ckpt_path).is_file()) and "source_step_10000" in CANONICAL_CHECKPOINTS_ALLOWLIST:
        src_path, src_hash, src_label = CANONICAL_CHECKPOINTS_ALLOWLIST["source_step_10000"]
        if src_path.is_file():
            ckpt_path, ckpt_hash, ckpt_label = src_path, src_hash, src_label
    
    run_timestamp = datetime.now(timezone.utc).isoformat()
    adapter = None
    weights_available = False

    if not smoke_mode and mock_generator is None:
        if ckpt_path is None or not Path(ckpt_path).is_file():
            print(f"[NOTICE] Checkpoint file unavailable at: {ckpt_path}")
            print("[NOTICE] Running benchmark in weight-independent verification / smoke mode.")
            smoke_mode = True
        else:
            adapter = NeuralRuntimeAdapter(
                checkpoint_path=ckpt_path,
                max_new_tokens=max_new_tokens,
                decoding=decoding
            )
            adapter.initialize()
            weights_available = True

    results = {
        "suite_name": suite["suite_name"],
        "suite_version": suite.get("version", "1.2.0"),
        "evaluation_timestamp": run_timestamp,
        "environment": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
        },
        "mode": "smoke_verification" if smoke_mode else "neural_evaluation",
        "checkpoint_label": ckpt_label,
        "checkpoint_sha256": ckpt_hash,
        "checkpoint_present": weights_available,
        "decoding": decoding,
        "max_new_tokens": max_new_tokens,
        "total_tasks": len(tasks),
        "syntax_valid_count": 0,
        "test_passed_count": 0,
        "match_count": 0,
        "timeout_count": 0,
        "empty_output_count": 0,
        "policy_rejected_count": 0,
        "categories_evaluated": sorted(list({t["category"] for t in tasks})),
        "task_results": []
    }
    
    print("=" * 70)
    print(f"RUNNING CODING EVALUATION: {ckpt_label} ({decoding}) [Smoke: {smoke_mode}]")
    print("=" * 70)
    
    total_tokens_generated = 0
    total_gen_time_ms = 0.0
    
    for idx, task in enumerate(tasks):
        task_id = task["id"]
        category = task["category"]
        prompt = task["prompt"]
        target = task.get("target_snippet", "")
        val_type = task.get("validation_type", "unit_test")
        test_code = task.get("test_code", "")
        
        # Generation
        t0 = time.perf_counter()
        if smoke_mode:
            # Deterministic smoke generation: tests prompt parsing without weight dependency
            gen_text = ""
            gen_duration_ms = 0.1
            tok_count = 0
        elif mock_generator is not None:
            gen_text = mock_generator(task)
            gen_duration_ms = (time.perf_counter() - t0) * 1000.0
            tok_count = max(1, len(gen_text) // 3)
        else:
            out = adapter.generate(prompt)
            gen_duration_ms = (time.perf_counter() - t0) * 1000.0
            gen_text = out["generated_text"]
            gen_tokens = out.get("generated_tokens", [])
            tok_count = len(gen_tokens) if gen_tokens else max(1, len(gen_text) // 3)

        total_tokens_generated += tok_count
        total_gen_time_ms += gen_duration_ms
        
        is_empty = len(gen_text.strip()) == 0
        if is_empty:
            results["empty_output_count"] += 1
            
        full_code = prompt + gen_text
        is_syntax = check_python_syntax(full_code)
        if is_syntax:
            results["syntax_valid_count"] += 1
            
        test_passed = False
        substring_matched = False
        exact_matched = gen_text.strip() == target.strip()
        exec_err = None
        exec_latency = 0.0
        timed_out = False
        rejected_by_policy = False
        
        # Separate prompt scaffolding from model completion.
        # Functional completion requires non-empty, non-comment code.
        clean_gen = gen_text.strip()
        lines = [l.strip() for l in clean_gen.splitlines() if l.strip()]
        non_comment_lines = [l for l in lines if not l.startswith("#")]
        has_executable_code = len(non_comment_lines) > 0

        if val_type == "unit_test" and test_code:
            if not has_executable_code:
                test_passed = False
                exec_err = "NO_EXECUTABLE_CODE: completion is empty or comments only"
            else:
                exec_res = run_process_isolated_test(full_code, test_code, timeout_seconds=2.0)
                test_passed = exec_res["passed"]
                exec_err = exec_res["error"]
                exec_latency = exec_res.get("latency_ms", 0.0)
                timed_out = exec_res.get("timed_out", False)
                rejected_by_policy = exec_res.get("rejected_by_policy", False)
                if test_passed:
                    results["test_passed_count"] += 1
                if timed_out:
                    results["timeout_count"] += 1
                if rejected_by_policy:
                    results["policy_rejected_count"] += 1
        elif val_type == "substring_match":
            substring_matched = target.lower() in gen_text.lower() if gen_text else False
            if substring_matched:
                results["match_count"] += 1
                test_passed = True
        elif val_type == "exact_match":
            test_passed = exact_matched
            if exact_matched:
                results["match_count"] += 1
                
        task_res = {
            "id": task_id,
            "category": category,
            "title": task.get("title", task_id),
            "prompt": prompt,
            "generated_text": gen_text,
            "target_snippet": target,
            "syntax_valid": is_syntax,
            "test_passed": test_passed,
            "exact_match": exact_matched,
            "substring_match": substring_matched,
            "empty_output": is_empty,
            "timed_out": timed_out,
            "rejected_by_policy": rejected_by_policy,
            "error": exec_err,
            "gen_latency_ms": round(gen_duration_ms, 2),
            "exec_latency_ms": exec_latency
        }
        results["task_results"].append(task_res)
        status_str = "PASS" if test_passed else "FAIL"
        print(f"  [{idx+1}/{len(tasks)}] {task_id:<14} ({category:<26}) | {status_str} | Syntax: {str(is_syntax):<5} | Err: {exec_err}")
        
    overall_tok_per_sec = (total_tokens_generated / (total_gen_time_ms / 1000.0)) if total_gen_time_ms > 0 else 0.0
    results["tokens_per_second"] = round(overall_tok_per_sec, 2)
    results["syntax_valid_rate"] = f"{results['syntax_valid_count']}/{len(tasks)} ({round(results['syntax_valid_count']/len(tasks)*100, 1)}%)"
    results["overall_success_rate"] = f"{results['test_passed_count']}/{len(tasks)} ({round(results['test_passed_count']/len(tasks)*100, 1)}%)"
    results["timeout_rate"] = f"{results['timeout_count']}/{len(tasks)} ({round(results['timeout_count']/len(tasks)*100, 1)}%)"
    results["rejection_rate"] = f"{results['policy_rejected_count']}/{len(tasks)} ({round(results['policy_rejected_count']/len(tasks)*100, 1)}%)"
    
    out_target = Path(output_path) if output_path else DEFAULT_OUTPUT_PATH
    out_target.parent.mkdir(parents=True, exist_ok=True)
    with open(out_target, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        
    print("=" * 70)
    print(f"SUMMARY: Success: {results['overall_success_rate']} | Syntax: {results['syntax_valid_rate']} | Speed: {results['tokens_per_second']} tok/s")
    print(f"Saved results to: {out_target}")
    print("=" * 70)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ChakrView Coding Benchmark Runner")
    parser.add_argument("--checkpoint-key", type=str, default=DEFAULT_CHECKPOINT_KEY, help="Allowlist key for checkpoint")
    parser.add_argument("--max-new-tokens", type=int, default=48, help="Max tokens to generate")
    parser.add_argument("--decoding", choices=["greedy", "sampling"], default="greedy", help="Decoding strategy")
    parser.add_argument("--output-path", type=str, default=None, help="Path to write evaluation results")
    parser.add_argument("--smoke", action="store_true", help="Run deterministic smoke verification without model weights")
    args = parser.parse_args()

    evaluate_coding_suite(
        checkpoint_key=args.checkpoint_key,
        max_new_tokens=args.max_new_tokens,
        decoding=args.decoding,
        output_path=args.output_path,
        smoke_mode=args.smoke,
    )
