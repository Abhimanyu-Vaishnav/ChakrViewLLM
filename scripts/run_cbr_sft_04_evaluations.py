"""
scripts/run_cbr_sft_04_evaluations.py

Comprehensive independent evaluation for CBR-SFT-04 across:
1. Held-Out CBR-SFT-04 Suite (80 tasks: Track A Greetings, Track B Identity, Track C Grounding, Track D Multi-Turn)
   - Evaluated using DualScorer (Legacy word-boundary regex vs. Semantic/Task-specific scorer)
2. Canonical M3 Conversational benchmark (60 tasks)
3. Native Intelligence benchmark (32 tasks)
4. Coding benchmark (12 tasks)
5. Held-Out CBR-SFT-03B Greeting & Identity benchmark (70 tasks)

Evaluated Checkpoints:
- CBR-SFT-03B Step 200 (Parent Checkpoint)
- CBR-SFT-03B Step 300 (Alternate Parent)
- CBR-SFT-04 Step 50
- CBR-SFT-04 Step 100
- CBR-SFT-04 Step 150
- CBR-SFT-04 Step 200
- CBR-SFT-04 Step 250
- CBR-SFT-04 Step 300
"""

import sys
import os
import json
import re
import time
import hashlib
from pathlib import Path
from typing import Dict, Any, List

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.run_m3_evaluations import (
    DirectNeuralRunner,
    evaluate_conversational,
    evaluate_intelligence,
    evaluate_coding,
    file_sha256
)
from evaluation.cbr_sft_04_suite.evaluator import DualScorer
from scripts.run_cbr_sft_03b_evaluations import evaluate_greeting_identity_suite

def evaluate_cbr_sft_04_suite(runner: DirectNeuralRunner, suite_path: Path) -> Dict[str, Any]:
    with open(suite_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    tasks = data["tasks"]

    legacy_passed_count = 0
    semantic_passed_count = 0
    cat_metrics = {}
    results = []
    divergence_cases = []

    for task in tasks:
        t_id = task["id"]
        cat = task["category"]
        prompt = task["prompt"]
        expected = task.get("expected_concept", "").strip()
        max_tokens = task.get("max_tokens", 32)

        cat_metrics.setdefault(cat, {
            "total": 0,
            "legacy_passed": 0,
            "semantic_passed": 0
        })
        cat_metrics[cat]["total"] += 1

        t0 = time.perf_counter()
        resp = runner.generate(prompt=prompt, max_new_tokens=max_tokens, temperature=0.0)
        dur_ms = (time.perf_counter() - t0) * 1000.0
        gen_text = resp["text"]

        leg_pass = DualScorer.score_legacy(gen_text, expected)
        sem_pass, sem_reason = DualScorer.score_semantic(gen_text, expected, cat)

        if leg_pass:
            legacy_passed_count += 1
            cat_metrics[cat]["legacy_passed"] += 1

        if sem_pass:
            semantic_passed_count += 1
            cat_metrics[cat]["semantic_passed"] += 1

        if leg_pass != sem_pass:
            divergence_cases.append({
                "id": t_id,
                "category": cat,
                "expected": expected,
                "output": gen_text,
                "legacy_score": leg_pass,
                "semantic_score": sem_pass,
                "semantic_reason": sem_reason
            })

        results.append({
            "id": t_id,
            "category": cat,
            "prompt": prompt,
            "expected_concept": expected,
            "generated_text": gen_text,
            "stop_reason": resp.get("stop_reason", ""),
            "duration_ms": round(dur_ms, 2),
            "legacy_pass": leg_pass,
            "semantic_pass": sem_pass,
            "semantic_reason": sem_reason
        })

    total = len(tasks)
    return {
        "suite_name": data.get("suite_name", "HELD_OUT_CBR_SFT_04_80"),
        "total_tasks": total,
        "legacy_passed_tasks": legacy_passed_count,
        "legacy_accuracy": legacy_passed_count / total if total > 0 else 0.0,
        "semantic_passed_tasks": semantic_passed_count,
        "semantic_accuracy": semantic_passed_count / total if total > 0 else 0.0,
        "divergence_count": len(divergence_cases),
        "divergence_cases": divergence_cases,
        "category_metrics": {
            c_name: {
                "total": c_val["total"],
                "legacy_passed": c_val["legacy_passed"],
                "legacy_accuracy": c_val["legacy_passed"] / c_val["total"] if c_val["total"] > 0 else 0.0,
                "semantic_passed": c_val["semantic_passed"],
                "semantic_accuracy": c_val["semantic_passed"] / c_val["total"] if c_val["total"] > 0 else 0.0
            } for c_name, c_val in cat_metrics.items()
        },
        "task_results": results
    }

def main():
    print("=" * 70)
    print("CHAKRVIEW — CBR-SFT-04 COMPREHENSIVE EVALUATION HARNESS")
    print("=" * 70)

    run_dir = REPO_ROOT / "experiments" / "cbr01_sft" / "runs" / "cbr_sft_04"
    reports_dir = run_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    cbr4_suite_p = REPO_ROOT / "evaluation" / "cbr_sft_04_suite" / "held_out_cbr_sft_04_80.json"
    cbr3b_suite_p = REPO_ROOT / "evaluation" / "cbr_sft_03b_suite" / "held_out_greeting_identity_70.json"

    checkpoints_to_eval = [
        ("cbr3b_step200_parent", REPO_ROOT / "experiments" / "cbr01_sft" / "runs" / "cbr_sft_03b" / "sft_checkpoint_step_200.pt"),
        ("cbr3b_step300_alternate", REPO_ROOT / "experiments" / "cbr01_sft" / "runs" / "cbr_sft_03b" / "sft_checkpoint_step_300.pt"),
        ("cbr4_step50", run_dir / "checkpoints" / "sft_checkpoint_step_50.pt"),
        ("cbr4_step100", run_dir / "checkpoints" / "sft_checkpoint_step_100.pt"),
        ("cbr4_step150", run_dir / "checkpoints" / "sft_checkpoint_step_150.pt"),
        ("cbr4_step200", run_dir / "checkpoints" / "sft_checkpoint_step_200.pt"),
        ("cbr4_step250", run_dir / "checkpoints" / "sft_checkpoint_step_250.pt"),
        ("cbr4_step300", run_dir / "checkpoints" / "sft_checkpoint_step_300.pt"),
    ]

    master_eval_results = {}
    master_regression_results = {}
    all_raw_outputs = []

    for name, ckpt_p in checkpoints_to_eval:
        print("\n" + "#" * 60)
        print(f"EVALUATING CANDIDATE: {name}")
        print(f"Path: {ckpt_p}")
        sha = file_sha256(ckpt_p)
        print(f"SHA-256: {sha}")
        print("#" * 60)

        runner = DirectNeuralRunner(ckpt_p)

        # 1. CBR-SFT-04 80-task Suite
        print("\n1. Running Held-Out CBR-SFT-04 Suite (80 tasks)...")
        cbr4_eval = evaluate_cbr_sft_04_suite(runner, cbr4_suite_p)
        print(f"  -> CBR-SFT-04 Legacy Pass:   {cbr4_eval['legacy_passed_tasks']}/80 ({cbr4_eval['legacy_accuracy']*100:.1f}%)")
        print(f"  -> CBR-SFT-04 Semantic Pass: {cbr4_eval['semantic_passed_tasks']}/80 ({cbr4_eval['semantic_accuracy']*100:.1f}%)")

        for tr in cbr4_eval["task_results"]:
            all_raw_outputs.append({
                "model": name,
                "checkpoint_sha256": sha,
                "suite": "CBR_SFT_04_80",
                "task_id": tr["id"],
                "category": tr["category"],
                "prompt": tr["prompt"],
                "expected": tr["expected_concept"],
                "output": tr["generated_text"],
                "legacy_pass": tr["legacy_pass"],
                "semantic_pass": tr["semantic_pass"],
                "semantic_reason": tr["semantic_reason"]
            })

        # 2. CBR-SFT-03B 70-task Suite
        print("\n2. Running CBR-SFT-03B Greeting/Identity Suite (70 tasks)...")
        cbr3b_eval = evaluate_greeting_identity_suite(runner, cbr3b_suite_p)
        print(f"  -> CBR-SFT-03B Pass: {cbr3b_eval['passed_tasks']}/70 ({cbr3b_eval['accuracy']*100:.1f}%)")

        # 3. Canonical M3 Conversational Suite (60 tasks)
        print("\n3. Running Canonical M3 Conversational Suite (60 tasks)...")
        conv_res = evaluate_conversational(runner, f"cbr4_{name}", {"label": name, "path": ckpt_p})
        print(f"  -> M3 Conversational Pass: {conv_res['passed_tasks']}/60 ({conv_res['accuracy']:.1f}%)")

        # 4. Native Intelligence Suite (32 tasks)
        print("\n4. Running Native Intelligence Suite (32 tasks)...")
        intel_res = evaluate_intelligence(runner, f"cbr4_{name}", {"label": name, "path": ckpt_p})
        intel_passed = intel_res["passed_tasks_count"]
        intel_total = intel_res["total_tasks"]
        print(f"  -> Native Intelligence Pass: {intel_passed}/{intel_total} ({intel_passed/intel_total*100:.1f}%)")

        # 5. Coding Suite (12 tasks)
        print("\n5. Running Coding Benchmark (12 tasks)...")
        coding_res = evaluate_coding(runner, f"cbr4_{name}", {"label": name, "path": ckpt_p})
        syntax_count = coding_res["syntax_valid_count"]
        test_passed = coding_res["test_passed_count"]
        coding_total = coding_res["total_tasks"]
        print(f"  -> Coding Syntax Valid: {syntax_count}/{coding_total} ({syntax_count/coding_total*100:.1f}%)")
        print(f"  -> Coding Tests Pass:   {test_passed}/{coding_total} ({test_passed/coding_total*100:.1f}%)")

        # Compile Master Summaries
        master_eval_results[name] = {
            "checkpoint_path": str(ckpt_p),
            "sha256": sha,
            "cbr_sft_04_80": {
                "legacy_passed": cbr4_eval["legacy_passed_tasks"],
                "legacy_accuracy": cbr4_eval["legacy_accuracy"],
                "semantic_passed": cbr4_eval["semantic_passed_tasks"],
                "semantic_accuracy": cbr4_eval["semantic_accuracy"],
                "category_metrics": cbr4_eval["category_metrics"],
                "divergence_count": cbr4_eval["divergence_count"]
            },
            "cbr_sft_03b_70": {
                "passed": cbr3b_eval["passed_tasks"],
                "total": cbr3b_eval["total_tasks"],
                "accuracy": cbr3b_eval["accuracy"],
                "category_metrics": cbr3b_eval["category_metrics"]
            }
        }

        master_regression_results[name] = {
            "checkpoint_path": str(ckpt_p),
            "sha256": sha,
            "m3_conversational_60": {
                "passed": conv_res["passed_tasks"],
                "total": conv_res["total_tasks"],
                "accuracy": conv_res["accuracy"]
            },
            "native_intelligence_32": {
                "passed": intel_passed,
                "total": intel_total,
                "accuracy": round(intel_passed / intel_total * 100.0, 2),
                "attractor_collapse_count": intel_res.get("attractor_collapse_count", 0)
            },
            "coding_12": {
                "syntax_valid": syntax_count,
                "syntax_rate": round(syntax_count / coding_total * 100.0, 2),
                "functional_passed": test_passed,
                "functional_rate": round(test_passed / coding_total * 100.0, 2)
            }
        }

    # 6. Save Artifacts to run_dir and reports_dir
    with open(run_dir / "evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(master_eval_results, f, indent=2)

    with open(run_dir / "regression_results.json", "w", encoding="utf-8") as f:
        json.dump(master_regression_results, f, indent=2)

    with open(run_dir / "raw_model_outputs.jsonl", "w", encoding="utf-8") as f:
        for out_rec in all_raw_outputs:
            f.write(json.dumps(out_rec, ensure_ascii=False) + "\n")

    # Save Scorer Comparison JSON
    scorer_comp = {}
    for name in master_eval_results:
        scorer_comp[name] = {
            "legacy_pass": master_eval_results[name]["cbr_sft_04_80"]["legacy_passed"],
            "semantic_pass": master_eval_results[name]["cbr_sft_04_80"]["semantic_passed"],
            "divergence_count": master_eval_results[name]["cbr_sft_04_80"]["divergence_count"]
        }
    with open(run_dir / "scorer_comparison.json", "w", encoding="utf-8") as f:
        json.dump(scorer_comp, f, indent=2)

    print("\n" + "=" * 70)
    print("ALL EVALUATIONS AND REGRESSION SUITES COMPLETED!")
    print(f"Results saved under: {run_dir}")
    print("=" * 70)

if __name__ == "__main__":
    main()
