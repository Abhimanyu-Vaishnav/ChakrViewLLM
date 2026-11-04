"""
scripts/run_cbr_sft_03b_evaluations.py

Comprehensive independent evaluation for CBR-SFT-03B across:
1. Held-out Greeting & Identity Suite (70 tasks)
2. Canonical M3 Conversational benchmark (60 tasks)
3. Native Intelligence benchmark (32 tasks)
4. Coding benchmark (12 tasks)

Evaluated Models:
- Canonical Baseline (Step 10,000)
- CBR-SFT-02 Step 50 (Parent checkpoint)
- CBR-SFT-03B Step 75
- CBR-SFT-03B Step 100
- CBR-SFT-03B Step 150
- CBR-SFT-03B Step 200
- CBR-SFT-03B Step 300
"""

import sys
import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.run_m3_evaluations import (
    DirectNeuralRunner,
    evaluate_conversational,
    evaluate_intelligence,
    evaluate_coding,
    file_sha256
)

def evaluate_greeting_identity_suite(runner: DirectNeuralRunner, suite_path: Path) -> dict:
    with open(suite_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    tasks = data["tasks"]
    
    passed_count = 0
    cat_counts = {}
    results = []
    
    for task in tasks:
        t_id = task["id"]
        cat = task["category"]
        prompt = task["prompt"]
        expected = task.get("expected_concept", "").strip().lower()
        max_tokens = task.get("max_tokens", 24)
        
        cat_counts.setdefault(cat, {"total": 0, "passed": 0})
        cat_counts[cat]["total"] += 1
        
        resp = runner.generate(prompt=prompt, max_new_tokens=max_tokens, temperature=0.0)
        gen_text = resp["text"].strip().lower()
        
        # Word-boundary matching
        pattern = r"\b" + re.escape(expected) + r"\b"
        passed = bool(re.search(pattern, gen_text))
        if passed:
            passed_count += 1
            cat_counts[cat]["passed"] += 1
            
        results.append({
            "id": t_id,
            "category": cat,
            "prompt": prompt,
            "expected_concept": expected,
            "generated_text": resp["text"],
            "stop_reason": resp.get("stop_reason", ""),
            "duration_ms": resp.get("duration_ms", 0),
            "passed": passed
        })
        
    return {
        "suite_name": data.get("suite_name", "HELD_OUT_GREETING_IDENTITY_70"),
        "total_tasks": len(tasks),
        "passed_tasks": passed_count,
        "accuracy": passed_count / len(tasks) if tasks else 0.0,
        "category_metrics": {
            cat: {
                "total": c["total"],
                "passed": c["passed"],
                "accuracy": c["passed"] / c["total"] if c["total"] > 0 else 0.0
            } for cat, c in cat_counts.items()
        },
        "task_results": results
    }

if __name__ == "__main__":
    runs_dir = REPO_ROOT / "experiments" / "cbr01_sft" / "runs" / "cbr_sft_03b"
    out_dir = REPO_ROOT / "docs" / "missions" / "CBR_01" / "experiments" / "CBR_SFT_03B" / "raw_evals"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    eval_suite_p = REPO_ROOT / "evaluation" / "cbr_sft_03b_suite" / "held_out_greeting_identity_70.json"
    
    models = [
        ("canonical_baseline", REPO_ROOT / "experiments" / "batch16_21_combined" / "main_training_10000" / "training_checkpoint_step_10000.pt"),
        ("cbr2_step50_parent", REPO_ROOT / "experiments" / "cbr01_sft" / "runs" / "cbr_sft_02" / "sft_checkpoint_step_50.pt"),
        ("step_75", runs_dir / "sft_checkpoint_step_75.pt"),
        ("step_100", runs_dir / "sft_checkpoint_step_100.pt"),
        ("step_150", runs_dir / "sft_checkpoint_step_150.pt"),
        ("step_200", runs_dir / "sft_checkpoint_step_200.pt"),
        ("step_300", runs_dir / "sft_checkpoint_step_300.pt")
    ]
    
    master_summary = {}
    
    for name, path in models:
        print(f"\n=======================================================")
        print(f"EVALUATING MODEL: {name}")
        print(f"=======================================================")
        runner = DirectNeuralRunner(path)
        
        print("1. Running Held-out Greeting & Identity Suite (70 tasks)...")
        greet_res = evaluate_greeting_identity_suite(runner, eval_suite_p)
        print(f"   Passed: {greet_res['passed_tasks']}/{greet_res['total_tasks']} ({greet_res['accuracy']*100:.1f}%)")
        print(f"   Breakdown: {greet_res['category_metrics']}")
        
        print("2. Running Canonical M3 Conversational Suite (60 tasks)...")
        conv_res = evaluate_conversational(runner, f"cbr3b_{name}", {"label": name, "path": path})
        
        print("3. Running Native Intelligence Suite (32 tasks)...")
        intel_res = evaluate_intelligence(runner, f"cbr3b_{name}", {"label": name, "path": path})
        
        print("4. Running Coding Benchmark (12 tasks)...")
        coding_res = evaluate_coding(runner, f"cbr3b_{name}", {"label": name, "path": path})
        
        master_summary[name] = {
            "held_out_greet_ident_70": {
                "passed": greet_res["passed_tasks"],
                "total": greet_res["total_tasks"],
                "accuracy": greet_res["accuracy"],
                "categories": greet_res["category_metrics"]
            },
            "conversational_60": {
                "passed": conv_res["passed_tasks"],
                "total": conv_res["total_tasks"],
                "accuracy": conv_res["accuracy"],
                "categories": conv_res["category_breakdown"]
            },
            "intelligence_32": {
                "passed": intel_res["passed_tasks_count"],
                "total": intel_res["total_tasks"],
                "attractors": intel_res["attractor_collapse_count"],
                "attractor_rate": intel_res["attractor_rate"]
            },
            "coding_12": {
                "passed": coding_res["test_passed_count"],
                "syntax_valid": coding_res["syntax_valid_count"],
                "total": coding_res["total_tasks"]
            },
            "sha256": file_sha256(path)
        }
        
        with open(out_dir / f"held_out_greet_ident_{name}.json", "w", encoding="utf-8") as f:
            json.dump(greet_res, f, indent=2)
        with open(out_dir / f"conversational_{name}.json", "w", encoding="utf-8") as f:
            json.dump(conv_res, f, indent=2)
        with open(out_dir / f"intelligence_{name}.json", "w", encoding="utf-8") as f:
            json.dump(intel_res, f, indent=2)
        with open(out_dir / f"coding_{name}.json", "w", encoding="utf-8") as f:
            json.dump(coding_res, f, indent=2)
            
    print("\n=======================================================")
    print("MASTER SUMMARY MATRIX")
    print("=======================================================")
    print(json.dumps(master_summary, indent=2))
    with open(out_dir / "master_summary_matrix.json", "w", encoding="utf-8") as f:
        json.dump(master_summary, f, indent=2)

