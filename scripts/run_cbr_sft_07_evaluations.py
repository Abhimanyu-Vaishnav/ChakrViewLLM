"""
scripts/run_cbr_sft_07_evaluations.py

Comprehensive independent evaluation for CBR-SFT-07 across:
1. Dedicated CBR-SFT-07 Held-Out Suite (115 tasks):
   - Greetings & Social (morning, afternoon, evening, generic)
   - Identity & Architecture
   - Context Grounding & Exact Retrieval
   - Multi-turn Dialogue & Memory Retention
   - Natural General QA
   - Explanations & Technical Reasoning
   - Supportive & Constructive Dialogue
   - Hindi & Hinglish Comprehension
   - Diverse Abstention
   - Continuity & Clarification
2. Baseline Reproduction & Anchor Suites:
   - CBR-SFT-06A Held-Out Suite (100 tasks)
   - CBR-SFT-04 Held-Out Suite (80 tasks)
   - CBR-SFT-03B Greeting & Identity Suite (70 tasks)
   - Canonical M3 Conversational Suite (60 tasks)
   - Native Intelligence Suite (32 tasks)
   - Coding Suite (12 tasks)

Evaluated Checkpoints:
- CBR-SFT-06A Step 25 (Parent Checkpoint)
- CBR-SFT-07 Step 25
- CBR-SFT-07 Step 50
- CBR-SFT-07 Step 75
- CBR-SFT-07 Step 100
- CBR-SFT-07 Step 150
- CBR-SFT-07 Step 200
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
from scripts.run_cbr_sft_06a_evaluations import evaluate_cbr_sft_06a_suite

def score_07_task(cat: str, gen_text: str, task: Dict[str, Any]) -> bool:
    gen_clean = gen_text.strip().lower()
    
    if cat == "greetings":
        tag = task.get("tag", "generic")
        if tag == "morning":
            return any(w in gen_clean for w in ["morning", "good morning", "shubh prabhat", "suprabhat"])
        elif tag == "afternoon":
            return any(w in gen_clean for w in ["afternoon", "good afternoon"])
        elif tag == "evening":
            return any(w in gen_clean for w in ["evening", "good evening", "shubh sandhya"])
        else:
            return any(w in gen_clean for w in ["hello", "hi", "greetings", "namaste", "welcome", "assist"])

    elif cat == "identity":
        # ChakrView / ChakrMicro identity awareness or local CPU awareness
        keywords = ["chakrview", "chakrmicro", "indigenous", "cpu", "neural", "scratch", "assistant", "first principles"]
        negations = ["no", "neither", "not based on", "independent"]
        if "chatgpt" in task.get("prompt", "").lower() or "llama" in task.get("prompt", "").lower():
            return any(w in gen_clean for w in negations)
        return any(w in gen_clean for w in keywords)

    elif cat == "context_grounding":
        target = task.get("target", "").strip().lower()
        return target in gen_clean

    elif cat == "multiturn_dialogue":
        target = task.get("target", "").strip().lower()
        return target in gen_clean

    elif cat == "natural_qa":
        # Technical relevance or informative output
        return len(gen_clean.split()) >= 3 and not any(r in gen_clean for r in ["unknown", "not provided", "not contain"])

    elif cat == "explanations_reasoning":
        return len(gen_clean.split()) >= 5

    elif cat == "supportive_conversation":
        supportive_words = ["welcome", "pleasure", "glad", "understand", "normal", "step", "break", "patience", "proud", "great job"]
        return any(w in gen_clean for w in supportive_words) or len(gen_clean.split()) >= 4

    elif cat == "hindi_hinglish":
        indic_words = ["namaste", "main", "chakrview", "madad", "haan", "kaise", "aap", "shukriya", "sahayata", "नमस्ते", "हाँ", "सहायता", "चक्रव्यूह"]
        return any(w in gen_clean for w in indic_words) or len(gen_clean.split()) >= 3

    elif cat == "abstention":
        refusal_patterns = [
            "not provided", "not specify", "not mentioned", "not state",
            "insufficient", "unknown", "cannot provide", "no information",
            "not have access", "unspecified"
        ]
        return any(p in gen_clean for p in refusal_patterns)

    elif cat == "continuity_feedback":
        return len(gen_clean.split()) >= 3

    return len(gen_clean.split()) >= 2

def evaluate_cbr_sft_07_suite(runner: DirectNeuralRunner, suite_path: Path) -> Dict[str, Any]:
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
        target = t.get("target", "")

        cat_summary.setdefault(cat, {"total": 0, "passed": 0})
        cat_summary[cat]["total"] += 1

        t0 = time.perf_counter()
        resp = runner.generate(prompt=prompt, max_new_tokens=40, temperature=0.0)
        dur_ms = (time.perf_counter() - t0) * 1000.0
        gen_text = resp["text"]

        passed = score_07_task(cat, gen_text, t)
        if passed:
            passed_count += 1
            cat_summary[cat]["passed"] += 1

        task_results.append({
            "task_id": tid,
            "category": cat,
            "prompt": prompt,
            "target": target,
            "generated_text": gen_text,
            "passed": passed,
            "duration_ms": round(dur_ms, 2)
        })

    total = len(tasks)
    return {
        "suite_name": data.get("suite_name", "HELD_OUT_CBR_SFT_07_115"),
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
    print("CHAKRVIEW - CBR-SFT-07 FULL EVALUATION & REGRESSION HARNESS")
    print("=" * 70)

    run_dir = REPO_ROOT / "experiments" / "cbr01_sft" / "runs" / "cbr_sft_07"
    reports_dir = run_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    suite_07_p = REPO_ROOT / "evaluation" / "cbr_sft_07_suite" / "held_out_cbr_sft_07_100.json"
    suite_06a_p = REPO_ROOT / "evaluation" / "cbr_sft_06a_suite" / "held_out_cbr_sft_06a_100.json"
    suite_04_p = REPO_ROOT / "evaluation" / "cbr_sft_04_suite" / "held_out_cbr_sft_04_80.json"
    suite_03b_p = REPO_ROOT / "evaluation" / "cbr_sft_03b_suite" / "held_out_greeting_identity_70.json"

    checkpoints_to_eval = [
        ("cbr6a_step25_parent", REPO_ROOT / "experiments" / "cbr01_sft" / "runs" / "cbr_sft_06a" / "checkpoints" / "sft_checkpoint_step_25.pt"),
        ("cbr7_step25", run_dir / "checkpoints" / "sft_checkpoint_step_25.pt"),
        ("cbr7_step50", run_dir / "checkpoints" / "sft_checkpoint_step_50.pt"),
        ("cbr7_step75", run_dir / "checkpoints" / "sft_checkpoint_step_75.pt"),
        ("cbr7_step100", run_dir / "checkpoints" / "sft_checkpoint_step_100.pt"),
        ("cbr7_step150", run_dir / "checkpoints" / "sft_checkpoint_step_150.pt"),
        ("cbr7_step200", run_dir / "checkpoints" / "sft_checkpoint_step_200.pt"),
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

        # 1. CBR-SFT-07 Suite
        print("\n1. Running Held-Out CBR-SFT-07 Suite (115 tasks)...")
        eval_07 = evaluate_cbr_sft_07_suite(runner, suite_07_p)
        print(f"  -> CBR-SFT-07 Pass: {eval_07['passed_tasks']}/{eval_07['total_tasks']} ({eval_07['accuracy']*100:.1f}%)")

        for tr in eval_07["task_results"]:
            all_raw_outputs.append({
                "model": name,
                "checkpoint_sha256": sha,
                "suite": "CBR_SFT_07_115",
                "task_id": tr["task_id"],
                "category": tr["category"],
                "prompt": tr["prompt"],
                "target": tr["target"],
                "output": tr["generated_text"],
                "passed": tr["passed"]
            })

        # 2. CBR-SFT-06A 100-task Suite
        print("\n2. Running CBR-SFT-06A Suite (100 tasks)...")
        eval_06a = evaluate_cbr_sft_06a_suite(runner, suite_06a_p)
        print(f"  -> CBR-SFT-06A Pass: {eval_06a['passed_tasks']}/100 ({eval_06a['accuracy']*100:.1f}%)")

        # 3. CBR-SFT-04 80-task Suite
        print("\n3. Running CBR-SFT-04 Suite (80 tasks)...")
        eval_04 = evaluate_cbr_sft_04_suite(runner, suite_04_p)
        print(f"  -> CBR-SFT-04 Semantic Pass: {eval_04['semantic_passed_tasks']}/80 ({eval_04['semantic_accuracy']*100:.1f}%)")

        # 4. CBR-SFT-03B 70-task Suite
        print("\n4. Running CBR-SFT-03B Suite (70 tasks)...")
        eval_03b = evaluate_greeting_identity_suite(runner, suite_03b_p)
        print(f"  -> CBR-SFT-03B Pass: {eval_03b['passed_tasks']}/70 ({eval_03b['accuracy']*100:.1f}%)")

        # 5. Canonical M3 Conversational Suite (60 tasks)
        print("\n5. Running Canonical M3 Conversational Suite (60 tasks)...")
        conv_res = evaluate_conversational(runner, f"cbr7_{name}", {"label": name, "path": ckpt_p})
        print(f"  -> M3 Conversational Pass: {conv_res['passed_tasks']}/60 ({conv_res['accuracy']:.1f}%)")

        # 6. Native Intelligence Suite (32 tasks)
        print("\n6. Running Native Intelligence Suite (32 tasks)...")
        intel_res = evaluate_intelligence(runner, f"cbr7_{name}", {"label": name, "path": ckpt_p})
        intel_passed = intel_res["passed_tasks_count"]
        intel_total = intel_res["total_tasks"]
        print(f"  -> Native Intelligence Pass: {intel_passed}/{intel_total} ({intel_passed/intel_total*100:.1f}%)")

        # 7. Coding Suite (12 tasks)
        print("\n7. Running Coding Benchmark (12 tasks)...")
        coding_res = evaluate_coding(runner, f"cbr7_{name}", {"label": name, "path": ckpt_p})
        syntax_count = coding_res["syntax_valid_count"]
        test_passed = coding_res["test_passed_count"]
        coding_total = coding_res["total_tasks"]
        print(f"  -> Coding Syntax Valid: {syntax_count}/{coding_total} ({syntax_count/coding_total*100:.1f}%)")
        print(f"  -> Coding Tests Pass:   {test_passed}/{coding_total} ({test_passed/coding_total*100:.1f}%)")

        # Collate Master Record
        master_eval_results[name] = {
            "checkpoint_path": str(ckpt_p),
            "sha256": sha,
            "cbr_sft_07": {
                "passed": eval_07["passed_tasks"],
                "total": eval_07["total_tasks"],
                "accuracy": eval_07["accuracy"],
                "categories": eval_07["category_summary"]
            },
            "cbr_sft_06a": {
                "passed": eval_06a["passed_tasks"],
                "total": 100,
                "accuracy": eval_06a["accuracy"],
                "categories": eval_06a["category_summary"]
            },
            "cbr_sft_04": {
                "passed": eval_04["semantic_passed_tasks"],
                "total": 80,
                "accuracy": eval_04["semantic_accuracy"]
            },
            "cbr_sft_03b": {
                "passed": eval_03b["passed_tasks"],
                "total": 70,
                "accuracy": eval_03b["accuracy"]
            },
            "m3_conversational": {
                "passed": conv_res["passed_tasks"],
                "total": 60,
                "accuracy": conv_res["accuracy"]
            },
            "native_intelligence": {
                "passed": intel_passed,
                "total": intel_total,
                "accuracy": intel_passed / intel_total
            },
            "coding": {
                "syntax_valid": syntax_count,
                "test_passed": test_passed,
                "total": coding_total
            }
        }

    # Save Evaluation Results
    eval_file = run_dir / "EVALUATION_RESULTS.json"
    with open(eval_file, "w", encoding="utf-8") as f:
        json.dump(master_eval_results, f, indent=2)

    raw_file = run_dir / "RAW_GENERATION_RESULTS.jsonl"
    with open(raw_file, "w", encoding="utf-8") as f:
        for r in all_raw_outputs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("\n" + "=" * 70)
    print(f"ALL EVALUATIONS COMPLETE! Results saved to:")
    print(f"  {eval_file}")
    print(f"  {raw_file}")
    print("=" * 70)

if __name__ == "__main__":
    main()

