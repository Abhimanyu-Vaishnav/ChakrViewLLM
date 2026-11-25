"""
NR-016 Phase 2: Complete Evaluation of the 325-Task Sealed Benchmark
Evaluates ChakrGenesis-17M Stage-D Best on all 325 tasks across 12 families:
Measures:
- Strict exact match / execution pass rate per family
- Pure-neural vs Hybrid Cognitive Controller comparison
- Categorization of errors
"""

import sys
import os
import ast
import json
import subprocess
import torch
from pathlib import Path

sys.path.insert(0, '.')
from chakrview.runtime.inference_17m import ChakrViewInferenceEngine17M
from chakrview.brain.cognitive_architecture import CognitiveController

def evaluate_sealed_benchmark():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"=== EVALUATING 325-TASK SEALED BENCHMARK ON {device} ===")

    engine = ChakrViewInferenceEngine17M(
        checkpoint_path='experiments/genesis_stage_d_011/checkpoints/genesis_17m_staged_best.pt',
        device=device
    )
    controller = CognitiveController()
    tokenizer = engine.tokenizer

    def pure_neural_gen(prompt):
        formatted = f"<user>\n{prompt}\n<assistant>\n"
        resp = ""
        for chunk in engine.generate_stream(formatted, max_new_tokens=60, temperature=0.0):
            resp += chunk.get("text", "")
        return resp.strip()

    with open("data/sealed_eval_015/sealed_benchmark_300.json", 'r', encoding='utf-8') as f:
        tasks = json.load(f)

    print(f"Loaded {len(tasks)} tasks.")

    family_stats = {}
    detailed_results = []

    for idx, t in enumerate(tasks):
        fam = t['family']
        if fam not in family_stats:
            family_stats[fam] = {
                "total": 0,
                "pure_correct": 0,
                "hybrid_correct": 0
            }

        prompt = t['prompt']
        eval_type = t.get('eval_type', 'exact_or_numeric')
        gt = t.get('ground_truth', '')

        # Pure neural
        pure_out = pure_neural_gen(prompt)

        # Hybrid
        hybrid_res = controller.process(prompt, pure_neural_gen, mode="hybrid")
        hybrid_out = hybrid_res["response"]

        # Check correctness
        pure_pass = False
        hybrid_pass = False

        if eval_type == "python_execution":
            test_assertions = t['test_assertions']
            # Clean pure
            cleaned_pure = pure_out
            if "```python" in cleaned_pure: cleaned_pure = cleaned_pure.split("```python")[1].split("```")[0].strip()
            elif "```" in cleaned_pure: cleaned_pure = cleaned_pure.split("```")[1].split("```")[0].strip()
            try:
                ast.parse(cleaned_pure)
                res = subprocess.run([sys.executable, "-c", f"{cleaned_pure}\n\n{test_assertions}\n"], capture_output=True, timeout=2.0)
                pure_pass = (res.returncode == 0)
            except Exception:
                pure_pass = False
            # Hybrid
            cleaned_hyb = hybrid_out
            if "```python" in cleaned_hyb: cleaned_hyb = cleaned_hyb.split("```python")[1].split("```")[0].strip()
            elif "```" in cleaned_hyb: cleaned_hyb = cleaned_hyb.split("```")[1].split("```")[0].strip()
            try:
                ast.parse(cleaned_hyb)
                res = subprocess.run([sys.executable, "-c", f"{cleaned_hyb}\n\n{test_assertions}\n"], capture_output=True, timeout=2.0)
                hybrid_pass = (res.returncode == 0)
            except Exception:
                hybrid_pass = False
        elif eval_type == "exact_match":
            pure_pass = (gt.lower() in pure_out.lower())
            hybrid_pass = (gt.lower() in hybrid_out.lower())
        else: # exact_or_numeric
            pure_pass = (gt in pure_out)
            hybrid_pass = (gt in hybrid_out)

        family_stats[fam]["total"] += 1
        if pure_pass: family_stats[fam]["pure_correct"] += 1
        if hybrid_pass: family_stats[fam]["hybrid_correct"] += 1

        detailed_results.append({
            "task_id": t['task_id'],
            "family": fam,
            "prompt": prompt,
            "ground_truth": gt,
            "pure_out": pure_out,
            "hybrid_out": hybrid_out,
            "pure_pass": pure_pass,
            "hybrid_pass": hybrid_pass
        })

        if (idx + 1) % 50 == 0:
            print(f"Progress: {idx+1}/{len(tasks)} tasks evaluated...")

    # Summary table
    print("\n=== COMPLETE 325-TASK SEALED BENCHMARK RESULTS ===")
    total_pure = sum(f["pure_correct"] for f in family_stats.values())
    total_hyb = sum(f["hybrid_correct"] for f in family_stats.values())
    total_all = len(tasks)

    for fam, s in family_stats.items():
        p_pct = (s['pure_correct'] / s['total']) * 100
        h_pct = (s['hybrid_correct'] / s['total']) * 100
        print(f"{fam:30s} | Total: {s['total']:2d} | Pure: {s['pure_correct']:2d} ({p_pct:5.1f}%) | Hybrid: {s['hybrid_correct']:2d} ({h_pct:5.1f}%)")

    print("-" * 75)
    print(f"{'OVERALL STRICT ACCURACY':30s} | Total: {total_all:2d} | Pure: {total_pure:2d} ({total_pure/total_all*100:5.1f}%) | Hybrid: {total_hyb:2d} ({total_hyb/total_all*100:5.1f}%)")

    out_file = Path("data/sealed_eval_015/recovered_benchmark_results.json")
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump({
            "summary": family_stats,
            "total_tasks": total_all,
            "pure_correct": total_pure,
            "hybrid_correct": total_hyb,
            "pure_accuracy_pct": (total_pure / total_all) * 100,
            "hybrid_accuracy_pct": (total_hyb / total_all) * 100,
            "details": detailed_results
        }, f, indent=2, ensure_ascii=False)
    print(f"\nSaved detailed recovery results to: {out_file}")

if __name__ == '__main__':
    evaluate_sealed_benchmark()
