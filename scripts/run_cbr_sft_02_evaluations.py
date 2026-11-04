"""
scripts/run_cbr_sft_02_evaluations.py

Comprehensive independent evaluation for CBR-SFT-02 across 5 distinct benchmarks:
Suite A: M3 Conversational benchmark (60 tasks)
Suite B: Native Intelligence benchmark (32 tasks)
Suite C: Coding benchmark (12 tasks)
Suite D: Held-out Dynamic Entity Grounding (100 tasks)
Suite E: Held-out Multi-turn Context Retention (50 conversations)
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
    evaluate_coding
)

def evaluate_held_out_suite(runner: DirectNeuralRunner, suite_path: Path, suite_name: str) -> dict:
    with open(suite_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    tasks = data["tasks"]
    
    passed_count = 0
    results = []
    
    for task in tasks:
        t_id = task["id"]
        prompt = task["prompt"]
        expected = task.get("expected_concept", "").strip().lower()
        max_tokens = task.get("max_tokens", 32)
        
        resp = runner.generate(prompt=prompt, max_new_tokens=max_tokens, temperature=0.0)
        gen_text = resp["text"].strip().lower()
        
        # Word-boundary matching
        pattern = r"\b" + re.escape(expected) + r"\b"
        passed = bool(re.search(pattern, gen_text))
        if passed:
            passed_count += 1
            
        results.append({
            "id": t_id,
            "prompt": prompt,
            "expected": expected,
            "generated": resp["text"],
            "passed": passed
        })
        
    return {
        "suite_name": suite_name,
        "total_tasks": len(tasks),
        "passed_tasks": passed_count,
        "accuracy": passed_count / len(tasks) if tasks else 0.0,
        "task_results": results
    }

if __name__ == "__main__":
    runs_dir = REPO_ROOT / "experiments" / "cbr01_sft" / "runs" / "cbr_sft_02"
    out_dir = REPO_ROOT / "docs" / "missions" / "CBR_01" / "experiments" / "CBR_SFT_02" / "raw_evals"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoints = [
        ("canonical_baseline", REPO_ROOT / "experiments" / "batch16_21_combined" / "main_training_10000" / "training_checkpoint_step_10000.pt"),
        ("step_25", runs_dir / "sft_checkpoint_step_25.pt"),
        ("step_50", runs_dir / "sft_checkpoint_step_50.pt"),
        ("step_100", runs_dir / "sft_checkpoint_step_100.pt"),
        ("step_150", runs_dir / "sft_checkpoint_step_150.pt"),
        ("step_200", runs_dir / "sft_checkpoint_step_200.pt")
    ]
    
    entity_suite_path = REPO_ROOT / "evaluation" / "cbr_suite" / "held_out_entity_grounding_100.json"
    mturn_suite_path = REPO_ROOT / "evaluation" / "cbr_suite" / "held_out_multiturn_context_50.json"
    
    master_summary = {}
    
    for name, ckpt_path in checkpoints:
        print(f"\n=======================================================")
        print(f"EVALUATING CHECKPOINT: {name}")
        print(f"=======================================================")
        runner = DirectNeuralRunner(ckpt_path)
        
        print("Running Suite A: M3 Conversational (60 tasks)...")
        conv_res = evaluate_conversational(runner, f"cbr2_{name}", {"label": name, "path": ckpt_path})
        
        print("Running Suite B: Native Intelligence (32 tasks)...")
        intel_res = evaluate_intelligence(runner, f"cbr2_{name}", {"label": name, "path": ckpt_path})
        
        print("Running Suite C: Coding Benchmark (12 tasks)...")
        coding_res = evaluate_coding(runner, f"cbr2_{name}", {"label": name, "path": ckpt_path})
        
        print("Running Suite D: Held-out Entity Grounding (100 tasks)...")
        entity_res = evaluate_held_out_suite(runner, entity_suite_path, "HELD_OUT_ENTITY_GROUNDING_100")
        
        print("Running Suite E: Held-out Multi-turn Context (50 tasks)...")
        mturn_res = evaluate_held_out_suite(runner, mturn_suite_path, "HELD_OUT_MULTITURN_CONTEXT_50")
        
        master_summary[name] = {
            "conversational_60": {
                "passed": conv_res["passed_tasks"],
                "total": conv_res["total_tasks"],
                "accuracy": conv_res["accuracy"],
                "category_breakdown": conv_res["category_breakdown"]
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
            "entity_grounding_100": {
                "passed": entity_res["passed_tasks"],
                "total": entity_res["total_tasks"],
                "accuracy": entity_res["accuracy"]
            },
            "multiturn_context_50": {
                "passed": mturn_res["passed_tasks"],
                "total": mturn_res["total_tasks"],
                "accuracy": mturn_res["accuracy"]
            }
        }
        
        with open(out_dir / f"conversational_{name}.json", "w", encoding="utf-8") as f:
            json.dump(conv_res, f, indent=2)
        with open(out_dir / f"intelligence_{name}.json", "w", encoding="utf-8") as f:
            json.dump(intel_res, f, indent=2)
        with open(out_dir / f"coding_{name}.json", "w", encoding="utf-8") as f:
            json.dump(coding_res, f, indent=2)
        with open(out_dir / f"entity_grounding_{name}.json", "w", encoding="utf-8") as f:
            json.dump(entity_res, f, indent=2)
        with open(out_dir / f"multiturn_{name}.json", "w", encoding="utf-8") as f:
            json.dump(mturn_res, f, indent=2)
            
    print("\n=======================================================")
    print("MASTER EVALUATION MATRIX")
    print("=======================================================")
    print(json.dumps(master_summary, indent=2))
    with open(out_dir / "master_evaluation_matrix.json", "w", encoding="utf-8") as f:
        json.dump(master_summary, f, indent=2)

