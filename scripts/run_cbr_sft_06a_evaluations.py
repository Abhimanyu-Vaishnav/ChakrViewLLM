"""
scripts/run_cbr_sft_06a_evaluations.py

Comprehensive independent evaluation for CBR-SFT-06A across:
1. Dedicated CBR-SFT-06A Held-Out Suite (100 tasks):
   - Greetings (morning, afternoon, evening, generic)
   - Identity & capability boundaries
   - Single-token number copy
   - Single-token keyword copy
   - Key-value retrieval
   - Genuine abstention
2. Baseline Reproduction & Anchor Suites:
   - CBR-SFT-04 Held-Out Suite (80 tasks)
   - CBR-SFT-03B Greeting & Identity Suite (70 tasks)
   - Canonical M3 Conversational Suite (60 tasks)
   - Native Intelligence Suite (32 tasks)
   - Coding Suite (12 tasks)

Evaluated Checkpoints:
- CBR-SFT-04 Step 200 (Parent Checkpoint)
- CBR-SFT-06A Step 25
- CBR-SFT-06A Step 50
- CBR-SFT-06A Step 75
- CBR-SFT-06A Step 100
- CBR-SFT-06A Step 150
- CBR-SFT-06A Step 200
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

def score_06a_task(cat: str, gen_text: str, expected: str, scoring_rule: str) -> bool:
    gen_clean = gen_text.strip()
    exp_clean = expected.strip()

    if scoring_rule == "exact_copy":
        # Strict exact match or bounded prefix match
        if gen_clean == exp_clean:
            return True
        if gen_clean.startswith(exp_clean):
            remainder = gen_clean[len(exp_clean):]
            if not remainder or remainder[0] in " \n.,:;)]}":
                return True
        return False

    elif scoring_rule == "abstention":
        refusal_patterns = [
            "not provided", "not specify", "not mentioned", "not state",
            "insufficient", "unknown", "does not contain"
        ]
        return any(p in gen_clean.lower() for p in refusal_patterns)

    elif scoring_rule == "time_greeting":
        # Time conditioned greetings
        if exp_clean.lower() in gen_clean.lower():
            return True
        return False

    elif scoring_rule == "identity":
        # Must identify ChakrMicro / ChakrView or deny third-party models
        if exp_clean.lower() == "no":
            negations = ["no", "not", "neither", "independent"]
            return any(n in gen_clean.lower() for n in negations)
        return exp_clean.lower() in gen_clean.lower()

    elif scoring_rule == "generic_greeting":
        greetings = ["hello", "hi", "greetings", "good day", "hey", "welcome", "assist"]
        return any(g in gen_clean.lower() for g in greetings)

    return exp_clean.lower() in gen_clean.lower()

def evaluate_cbr_sft_06a_suite(runner: DirectNeuralRunner, suite_path: Path) -> Dict[str, Any]:
    with open(suite_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    tasks = data["tasks"]

    passed_count = 0
    cat_summary = {}
    task_results = []

    for t in tasks:
        tid = t["id"]
        cat = t["category"]
        prompt = t["prompt"]
        expected = t.get("expected_concept", "")
        rule = t.get("scoring", "exact_copy")

        cat_summary.setdefault(cat, {"total": 0, "passed": 0})
        cat_summary[cat]["total"] += 1

        t0 = time.perf_counter()
        resp = runner.generate(prompt=prompt, max_new_tokens=32, temperature=0.0)
        dur_ms = (time.perf_counter() - t0) * 1000.0
        gen_text = resp["text"]

        passed = score_06a_task(cat, gen_text, expected, rule)
        if passed:
            passed_count += 1
            cat_summary[cat]["passed"] += 1

        task_results.append({
            "task_id": tid,
            "category": cat,
            "prompt": prompt,
            "expected": expected,
            "scoring_rule": rule,
            "generated_text": gen_text,
            "passed": passed,
            "duration_ms": round(dur_ms, 2)
        })

    total = len(tasks)
    return {
        "suite_name": data.get("suite_name", "HELD_OUT_CBR_SFT_06A_100"),
        "total_tasks": total,
        "passed_tasks": passed_count,
        "accuracy": passed_count / total if total > 0 else 0.0,
        "category_summary": {
            k: {
                "total": v["total"],
                "passed": v["passed"],
                "accuracy": v["passed"] / v["total"] if v["total"] > 0 else 0.0
            } for k, v in cat_summary.items()
        },
        "task_results": task_results
    }

def main():
    print("=" * 70)
    print("CHAKRVIEW — CBR-SFT-06A FULL EVALUATION & REGRESSION HARNESS")
    print("=" * 70)

    run_dir = REPO_ROOT / "experiments" / "cbr01_sft" / "runs" / "cbr_sft_06a"
    reports_dir = run_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    suite_06a_p = REPO_ROOT / "evaluation" / "cbr_sft_06a_suite" / "held_out_cbr_sft_06a_100.json"
    suite_04_p = REPO_ROOT / "evaluation" / "cbr_sft_04_suite" / "held_out_cbr_sft_04_80.json"
    suite_03b_p = REPO_ROOT / "evaluation" / "cbr_sft_03b_suite" / "held_out_greeting_identity_70.json"

    checkpoints_to_eval = [
        ("cbr4_step200_parent", REPO_ROOT / "experiments" / "cbr01_sft" / "runs" / "cbr_sft_04" / "checkpoints" / "sft_checkpoint_step_200.pt"),
        ("cbr6a_step25", run_dir / "checkpoints" / "sft_checkpoint_step_25.pt"),
        ("cbr6a_step50", run_dir / "checkpoints" / "sft_checkpoint_step_50.pt"),
        ("cbr6a_step75", run_dir / "checkpoints" / "sft_checkpoint_step_75.pt"),
        ("cbr6a_step100", run_dir / "checkpoints" / "sft_checkpoint_step_100.pt"),
        ("cbr6a_step150", run_dir / "checkpoints" / "sft_checkpoint_step_150.pt"),
        ("cbr6a_step200", run_dir / "checkpoints" / "sft_checkpoint_step_200.pt"),
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

        # 1. CBR-SFT-06A 100-task Suite
        print("\n1. Running Held-Out CBR-SFT-06A Suite (100 tasks)...")
        eval_06a = evaluate_cbr_sft_06a_suite(runner, suite_06a_p)
        print(f"  -> CBR-SFT-06A Pass: {eval_06a['passed_tasks']}/100 ({eval_06a['accuracy']*100:.1f}%)")

        for tr in eval_06a["task_results"]:
            all_raw_outputs.append({
                "model": name,
                "checkpoint_sha256": sha,
                "suite": "CBR_SFT_06A_100",
                "task_id": tr["task_id"],
                "category": tr["category"],
                "prompt": tr["prompt"],
                "expected": tr["expected"],
                "output": tr["generated_text"],
                "passed": tr["passed"]
            })

        # 2. CBR-SFT-04 80-task Suite
        print("\n2. Running CBR-SFT-04 Suite (80 tasks)...")
        eval_04 = evaluate_cbr_sft_04_suite(runner, suite_04_p)
        print(f"  -> CBR-SFT-04 Semantic Pass: {eval_04['semantic_passed_tasks']}/80 ({eval_04['semantic_accuracy']*100:.1f}%)")

        # 3. CBR-SFT-03B 70-task Suite
        print("\n3. Running CBR-SFT-03B Suite (70 tasks)...")
        eval_03b = evaluate_greeting_identity_suite(runner, suite_03b_p)
        print(f"  -> CBR-SFT-03B Pass: {eval_03b['passed_tasks']}/70 ({eval_03b['accuracy']*100:.1f}%)")

        # 4. Canonical M3 Conversational Suite (60 tasks)
        print("\n4. Running Canonical M3 Conversational Suite (60 tasks)...")
        conv_res = evaluate_conversational(runner, f"cbr6a_{name}", {"label": name, "path": ckpt_p})
        print(f"  -> M3 Conversational Pass: {conv_res['passed_tasks']}/60 ({conv_res['accuracy']:.1f}%)")

        # 5. Native Intelligence Suite (32 tasks)
        print("\n5. Running Native Intelligence Suite (32 tasks)...")
        intel_res = evaluate_intelligence(runner, f"cbr6a_{name}", {"label": name, "path": ckpt_p})
        intel_passed = intel_res["passed_tasks_count"]
        intel_total = intel_res["total_tasks"]
        print(f"  -> Native Intelligence Pass: {intel_passed}/{intel_total} ({intel_passed/intel_total*100:.1f}%)")

        # 6. Coding Suite (12 tasks)
        print("\n6. Running Coding Benchmark (12 tasks)...")
        coding_res = evaluate_coding(runner, f"cbr6a_{name}", {"label": name, "path": ckpt_p})
        syntax_count = coding_res["syntax_valid_count"]
        test_passed = coding_res["test_passed_count"]
        coding_total = coding_res["total_tasks"]
        print(f"  -> Coding Syntax Valid: {syntax_count}/{coding_total} ({syntax_count/coding_total*100:.1f}%)")
        print(f"  -> Coding Tests Pass:   {test_passed}/{coding_total} ({test_passed/coding_total*100:.1f}%)")

        # Compile summaries
        master_eval_results[name] = {
            "checkpoint_path": str(ckpt_p),
            "sha256": sha,
            "cbr_sft_06a_100": {
                "passed": eval_06a["passed_tasks"],
                "total": eval_06a["total_tasks"],
                "accuracy": eval_06a["accuracy"],
                "category_summary": eval_06a["category_summary"]
            },
            "cbr_sft_04_80": {
                "legacy_passed": eval_04["legacy_passed_tasks"],
                "legacy_accuracy": eval_04["legacy_accuracy"],
                "semantic_passed": eval_04["semantic_passed_tasks"],
                "semantic_accuracy": eval_04["semantic_accuracy"],
                "category_metrics": eval_04["category_metrics"]
            },
            "cbr_sft_03b_70": {
                "passed": eval_03b["passed_tasks"],
                "total": eval_03b["total_tasks"],
                "accuracy": eval_03b["accuracy"],
                "category_metrics": eval_03b["category_metrics"]
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

    # Save deliverables
    with open(run_dir / "EVALUATION_RESULTS.json", "w", encoding="utf-8") as f:
        json.dump(master_eval_results, f, indent=2)

    with open(run_dir / "regression_results.json", "w", encoding="utf-8") as f:
        json.dump(master_regression_results, f, indent=2)

    with open(run_dir / "RAW_GENERATION_RESULTS.jsonl", "w", encoding="utf-8") as f:
        for r in all_raw_outputs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("\n" + "=" * 70)
    print("ALL CBR-SFT-06A EVALUATIONS COMPLETE!")
    print(f"Results saved to: {run_dir}")
    print("=" * 70)

if __name__ == "__main__":
    main()

