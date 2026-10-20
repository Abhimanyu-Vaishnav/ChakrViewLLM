"""
ChakrView Native Intelligence Evaluation Runner (v0.8).
Runs deterministic evaluation across the 32 versioned intelligence tasks.
Supports:
1. Native model evaluation with verified checkpoint allowlist.
2. Robust scoring using word boundary, number isolation, exact match, and isolated functional test execution.
3. Smoke verification mode (isolated test harness verification with Delta-W = 0).
4. Full result export into reproducible JSON artifacts without overwriting historical baselines.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import time
from typing import Any, Dict, List

from chakrview.runtime.neural_runtime_adapter import (
    NeuralRuntimeAdapter,
    CANONICAL_CHECKPOINTS_ALLOWLIST,
    DEFAULT_CHECKPOINT_KEY,
    compute_file_sha256,
)
from evaluation.intelligence_suite.scoring_evaluator import evaluate_response

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TASKS_FILE = REPO_ROOT / "evaluation" / "intelligence_suite" / "intelligence_tasks.json"
DEFAULT_OUTPUT_FILE = (
    REPO_ROOT
    / "evaluation"
    / "intelligence_suite"
    / f"intelligence_benchmark_run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
)


def run_benchmark(
    tasks_path: Path = DEFAULT_TASKS_FILE,
    output_path: Path | None = None,
    checkpoint_key: str = DEFAULT_CHECKPOINT_KEY,
    custom_checkpoint_path: str | None = None,
    smoke_mode: bool = False,
    decoding: str = "greedy",
    temperature: float = 0.0,
    repetition_penalty: float = 1.0,
    no_repeat_ngram_size: int = 0,
    max_new_tokens: int = 32,
) -> Dict[str, Any]:
    with open(tasks_path, "r", encoding="utf-8-sig") as f:
        data = json.load(f)

    tasks = data["tasks"]
    categories = data["categories"]

    ckpt_path = None
    ckpt_hash = None
    ckpt_label = "Unloaded / Smoke"

    # Resolve checkpoint
    if not smoke_mode:
        if custom_checkpoint_path:
            p = Path(custom_checkpoint_path).resolve()
            ckpt_path = p
            if p.is_file():
                ckpt_hash = compute_file_sha256(p)
                ckpt_label = f"Custom Checkpoint ({p.name})"
        else:
            allow_path, allow_hash, label = CANONICAL_CHECKPOINTS_ALLOWLIST[checkpoint_key]
            # Check source fallback if needed
            if not allow_path.is_file() and "source_step_10000" in CANONICAL_CHECKPOINTS_ALLOWLIST:
                allow_path, allow_hash, label = CANONICAL_CHECKPOINTS_ALLOWLIST["source_step_10000"]
            ckpt_path = allow_path
            ckpt_hash = allow_hash
            ckpt_label = label

    run_timestamp = datetime.now(timezone.utc).isoformat()
    adapter = None
    weights_available = False

    if not smoke_mode:
        if ckpt_path is None or not Path(ckpt_path).is_file():
            print(f"[NOTICE] Checkpoint file unavailable at: {ckpt_path}")
            print("[NOTICE] Operating in weight-independent verification / smoke mode.")
            smoke_mode = True
        else:
            adapter = NeuralRuntimeAdapter(
                checkpoint_path=ckpt_path,
                max_new_tokens=max_new_tokens,
                decoding=decoding,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                prompt_mode="raw_completion",
            )
            adapter.initialize()
            weights_available = True

    results = {
        "suite_name": data["suite_name"],
        "suite_version": data["version"],
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
        "generation_config": {
            "decoding": decoding,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "repetition_penalty": repetition_penalty,
            "no_repeat_ngram_size": no_repeat_ngram_size,
        },
        "total_tasks": len(tasks),
        "passed_tasks_count": 0,
        "attractor_collapse_count": 0,
        "category_metrics": {},
        "task_results": [],
    }

    print("=" * 75)
    print(f"RUNNING INTELLIGENCE BENCHMARK: {ckpt_label} ({decoding}) [Smoke: {smoke_mode}]")
    print(f"Repetition Penalty: {repetition_penalty} | No-Repeat Ngram: {no_repeat_ngram_size}")
    print("=" * 75)

    category_tallies = {cat: {"total": 0, "passed": 0, "attractor_collapse": 0} for cat in categories}

    for idx, task in enumerate(tasks):
        t_id = task["id"]
        cat = task["category"]
        prompt = task["prompt"]
        expected = task.get("expected_concept", "")
        test_code = task.get("test_code", "")
        scoring = task.get("scoring", "substring")

        category_tallies[cat]["total"] += 1

        t0 = time.perf_counter()
        if smoke_mode:
            gen_text = ""
            gen_ms = 0.1
            tok_count = 0
            stop_reason = "SMOKE_MODE"
        else:
            out = adapter.generate(
                prompt,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
            )
            gen_ms = (time.perf_counter() - t0) * 1000.0
            gen_text = out["generated_text"]
            tok_count = out["output_token_count"]
            stop_reason = out["stop_reason"]

        # Check for repetition attractor loop (e.g. repeated n-grams or consecutive identical words)
        words = gen_text.strip().lower().split()
        is_attractor = False
        if len(words) >= 6:
            tri_grams = [" ".join(words[i:i+3]) for i in range(len(words)-2)]
            if len(tri_grams) > len(set(tri_grams)) + 1:
                is_attractor = True
                results["attractor_collapse_count"] += 1
                category_tallies[cat]["attractor_collapse"] += 1

        # Robust scoring
        passed = False
        score_reason = "SMOKE_MODE"
        if not smoke_mode:
            passed, score_reason = evaluate_response(
                generated_text=gen_text,
                expected_concept=expected,
                scoring_rule=scoring,
                test_code=test_code,
                full_prompt=prompt,
            )

        if passed:
            results["passed_tasks_count"] += 1
            category_tallies[cat]["passed"] += 1

        status_str = "PASS" if passed else "FAIL"
        task_res = {
            "id": t_id,
            "category": cat,
            "prompt": prompt,
            "expected_concept": expected,
            "generated_text": gen_text,
            "scoring_rule": scoring,
            "passed": passed,
            "score_reason": score_reason,
            "attractor_collapse": is_attractor,
            "stop_reason": stop_reason,
            "token_count": tok_count,
            "latency_ms": round(gen_ms, 2),
        }
        results["task_results"].append(task_res)
        print(f"  [{idx+1:02d}/{len(tasks)}] {t_id:<10} ({cat:<28}) | {status_str} | Attractor: {str(is_attractor):<5} | Gen: {repr(gen_text[:40])}")

    for cat, tally in category_tallies.items():
        tot = tally["total"]
        pas = tally["passed"]
        results["category_metrics"][cat] = {
            "total": tot,
            "passed": pas,
            "pass_rate": f"{pas}/{tot} ({round((pas/tot)*100, 1)}%)" if tot > 0 else "0/0",
            "attractor_collapse_count": tally["attractor_collapse"],
        }

    results["overall_pass_rate"] = f"{results['passed_tasks_count']}/{len(tasks)} ({round((results['passed_tasks_count']/len(tasks))*100, 1)}%)"
    results["attractor_rate"] = f"{results['attractor_collapse_count']}/{len(tasks)} ({round((results['attractor_collapse_count']/len(tasks))*100, 1)}%)"

    out_file = Path(output_path) if output_path else DEFAULT_OUTPUT_FILE
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("=" * 75)
    print(f"BENCHMARK SUMMARY: Overall Pass Rate: {results['overall_pass_rate']} | Attractor Collapse: {results['attractor_rate']}")
    print(f"Saved results to: {out_file}")
    print("=" * 75)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ChakrView Native Intelligence Evaluation Suite")
    parser.add_argument("--smoke", action="store_true", help="Run in mock/smoke mode without model weights")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT_KEY, help="Checkpoint key or path")
    parser.add_argument("--decoding", type=str, default="greedy", choices=["greedy", "sampling"])
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    run_benchmark(
        smoke_mode=args.smoke,
        checkpoint_key=args.checkpoint,
        decoding=args.decoding,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        max_new_tokens=args.max_new_tokens,
        output_path=Path(args.output) if args.output else None,
    )

# Backward-compatible alias for existing test suites
evaluate_intelligence_suite = run_benchmark
