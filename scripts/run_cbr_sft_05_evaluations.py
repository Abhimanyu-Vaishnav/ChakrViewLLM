"""
scripts/run_cbr_sft_05_evaluations.py

Comprehensive independent evaluation for CBR-SFT-05 across:
1. Held-Out CBR-SFT-05 Suite (120 tasks: exact copy, KV lookup, multi-turn retention, abstention, identity, anchors)
2. Held-Out CBR-SFT-04 Suite (80 tasks)
3. Held-Out CBR-SFT-03B Greeting & Identity benchmark (70 tasks)
4. Canonical M3 Conversational benchmark (60 tasks)
5. Native Intelligence benchmark (32 tasks)
6. Coding benchmark (12 tasks)

Evaluated Checkpoints:
- CBR-SFT-04 Step 200 (Parent Checkpoint)
- CBR-SFT-03B Step 200 (Grandparent Checkpoint)
- CBR-SFT-05 Step 50
- CBR-SFT-05 Step 100
- CBR-SFT-05 Step 150
- CBR-SFT-05 Step 200
- CBR-SFT-05 Step 250
- CBR-SFT-05 Step 300
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
from scripts.run_cbr_sft_04_evaluations import evaluate_cbr_sft_04_suite
from scripts.run_cbr_sft_03b_evaluations import evaluate_greeting_identity_suite
from evaluation.cbr_sft_05_suite.evaluator import CBR05Evaluator, evaluate_suite_file

def main():
    print("=" * 70)
    print("CHAKRVIEW — CBR-SFT-05 FULL EVALUATION & REGRESSION HARNESS")
    print("=" * 70)

    run_dir = REPO_ROOT / "experiments" / "cbr01_sft" / "runs" / "cbr_sft_05"
    reports_dir = run_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    cbr5_suite_p = REPO_ROOT / "evaluation" / "cbr_sft_05_suite" / "held_out_cbr_sft_05_120.json"
    cbr4_suite_p = REPO_ROOT / "evaluation" / "cbr_sft_04_suite" / "held_out_cbr_sft_04_80.json"
    cbr3b_suite_p = REPO_ROOT / "evaluation" / "cbr_sft_03b_suite" / "held_out_greeting_identity_70.json"

    checkpoints_to_eval = [
        ("cbr4_step200_parent", REPO_ROOT / "experiments" / "cbr01_sft" / "runs" / "cbr_sft_04" / "checkpoints" / "sft_checkpoint_step_200.pt"),
        ("cbr3b_step200_grandparent", REPO_ROOT / "experiments" / "cbr01_sft" / "runs" / "cbr_sft_03b" / "sft_checkpoint_step_200.pt"),
        ("cbr5_step50", run_dir / "checkpoints" / "sft_checkpoint_step_50.pt"),
        ("cbr5_step100", run_dir / "checkpoints" / "sft_checkpoint_step_100.pt"),
        ("cbr5_step150", run_dir / "checkpoints" / "sft_checkpoint_step_150.pt"),
        ("cbr5_step200", run_dir / "checkpoints" / "sft_checkpoint_step_200.pt"),
        ("cbr5_step250", run_dir / "checkpoints" / "sft_checkpoint_step_250.pt"),
        ("cbr5_step300", run_dir / "checkpoints" / "sft_checkpoint_step_300.pt"),
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

        # 1. CBR-SFT-05 120-task Suite
        print("\n1. Running Held-Out CBR-SFT-05 Suite (120 tasks)...")
        cbr5_eval = evaluate_suite_file(cbr5_suite_p, runner)
        print(f"  -> CBR-SFT-05 Pass: {cbr5_eval['passed_tasks']}/120 ({cbr5_eval['accuracy']*100:.1f}%)")

        for tr in cbr5_eval["results"]:
            all_raw_outputs.append({
                "model": name,
                "checkpoint_sha256": sha,
                "suite": "CBR_SFT_05_120",
                "task_id": tr["task_id"],
                "category": tr["category"],
                "prompt": tr["prompt"],
                "expected": tr["expected"],
                "output": tr["generated_text"],
                "passed": tr["passed"],
                "reason": tr["reason"]
            })

        # 2. CBR-SFT-04 80-task Suite
        print("\n2. Running CBR-SFT-04 Held-Out Suite (80 tasks)...")
        cbr4_eval = evaluate_cbr_sft_04_suite(runner, cbr4_suite_p)
        print(f"  -> CBR-SFT-04 Legacy Pass:   {cbr4_eval['legacy_passed_tasks']}/80 ({cbr4_eval['legacy_accuracy']*100:.1f}%)")
        print(f"  -> CBR-SFT-04 Semantic Pass: {cbr4_eval['semantic_passed_tasks']}/80 ({cbr4_eval['semantic_accuracy']*100:.1f}%)")

        # 3. CBR-SFT-03B 70-task Suite
        print("\n3. Running CBR-SFT-03B Greeting/Identity Suite (70 tasks)...")
        cbr3b_eval = evaluate_greeting_identity_suite(runner, cbr3b_suite_p)
        print(f"  -> CBR-SFT-03B Pass: {cbr3b_eval['passed_tasks']}/70 ({cbr3b_eval['accuracy']*100:.1f}%)")

        # 4. Canonical M3 Conversational Suite (60 tasks)
        print("\n4. Running Canonical M3 Conversational Suite (60 tasks)...")
        conv_res = evaluate_conversational(runner, f"cbr5_{name}", {"label": name, "path": ckpt_p})
        print(f"  -> M3 Conversational Pass: {conv_res['passed_tasks']}/60 ({conv_res['accuracy']:.1f}%)")

        # 5. Native Intelligence Suite (32 tasks)
        print("\n5. Running Native Intelligence Suite (32 tasks)...")
        intel_res = evaluate_intelligence(runner, f"cbr5_{name}", {"label": name, "path": ckpt_p})
        intel_passed = intel_res["passed_tasks_count"]
        intel_total = intel_res["total_tasks"]
        print(f"  -> Native Intelligence Pass: {intel_passed}/{intel_total} ({intel_passed/intel_total*100:.1f}%)")

        # 6. Coding Suite (12 tasks)
        print("\n6. Running Coding Benchmark (12 tasks)...")
        coding_res = evaluate_coding(runner, f"cbr5_{name}", {"label": name, "path": ckpt_p})
        syntax_count = coding_res["syntax_valid_count"]
        test_passed = coding_res["test_passed_count"]
        coding_total = coding_res["total_tasks"]
        print(f"  -> Coding Syntax Valid: {syntax_count}/{coding_total} ({syntax_count/coding_total*100:.1f}%)")
        print(f"  -> Coding Tests Pass:   {test_passed}/{coding_total} ({test_passed/coding_total*100:.1f}%)")

        # Compile Master Summaries
        master_eval_results[name] = {
            "checkpoint_path": str(ckpt_p),
            "sha256": sha,
            "cbr_sft_05_120": {
                "passed": cbr5_eval["passed_tasks"],
                "total": cbr5_eval["total_tasks"],
                "accuracy": cbr5_eval["accuracy"],
                "category_breakdown": cbr5_eval["category_breakdown"]
            },
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

    # Save Master Artifacts
    with open(run_dir / "evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(master_eval_results, f, indent=2)

    with open(run_dir / "regression_results.json", "w", encoding="utf-8") as f:
        json.dump(master_regression_results, f, indent=2)

    with open(run_dir / "raw_model_outputs.jsonl", "w", encoding="utf-8") as f:
        for out_rec in all_raw_outputs:
            f.write(json.dumps(out_rec, ensure_ascii=False) + "\n")

    print("\n" + "=" * 70)
    print("ALL EVALUATIONS AND REGRESSION SUITES COMPLETED!")
    print(f"Results saved under: {run_dir}")
    print("=" * 70)

if __name__ == "__main__":
    main()
