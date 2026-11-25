"""
ChakrView Comprehensive Multi-Model 260-Task Evaluator.
Mission CV-GENESIS-NR-014 Phase 7 & 8.

Evaluates:
1. Untouched Baseline: ChakrGenesis-17M Stage-D Best (Pure Neural & Hybrid)
2. Prior Experimental: ChakrGenesis-17M Stage-E Best (Pure Neural)
3. New Balanced Candidate: ChakrGenesis-17M Stage-F Best (Pure Neural & Hybrid)

Across all 260 tasks:
- Python codegen with isolated subprocess execution (timeout 2.0s)
- Explanation & debugging
- Algorithmic reasoning
- Exact arithmetic
- English, Hindi, Hinglish conversation
- Instruction following
- Multi-turn context
- Safety abstention
"""

from __future__ import annotations
import os
import sys
import json
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chakrview.runtime.inference_17m import ChakrViewInferenceEngine17M
from experiments.baseline_eval_013.run_baseline_eval import evaluate_benchmark


def run_comprehensive_eval():
    bench_file = Path("data/sealed_eval_014/sealed_benchmark_260.json")
    with open(bench_file, "r", encoding="utf-8") as f:
        items = json.load(f)

    out_dir = Path("experiments/eval_260_014")
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    # Checkpoint paths
    ckpt_d = "experiments/genesis_stage_d_011/checkpoints/genesis_17m_staged_best.pt"
    ckpt_e = "experiments/genesis_stage_e_013/checkpoints/genesis_17m_stage_e_best.pt"
    ckpt_f = "experiments/genesis_stage_f_014/checkpoints/genesis_17m_stage_f_best.pt"

    # 1. Evaluate Stage-F Best (Pure Neural)
    print("\n--- Evaluating Stage-F Best (Pure Neural) ---")
    engine_f = ChakrViewInferenceEngine17M(ckpt_f)
    results["stage_f_pure_neural"] = evaluate_benchmark("ChakrGenesis-17M Stage-F Best", engine_f, "pure_neural", items, max_tokens=80)

    # 2. Evaluate Stage-F Best (Hybrid Cognitive)
    print("\n--- Evaluating Stage-F Best (Hybrid Cognitive) ---")
    results["stage_f_hybrid"] = evaluate_benchmark("ChakrGenesis-17M Stage-F Best", engine_f, "hybrid", items, max_tokens=80)

    # 3. Evaluate Baseline Stage-D (Pure Neural)
    print("\n--- Evaluating Baseline Stage-D (Pure Neural) ---")
    engine_d = ChakrViewInferenceEngine17M(ckpt_d)
    results["stage_d_pure_neural"] = evaluate_benchmark("ChakrGenesis-17M Stage-D Baseline", engine_d, "pure_neural", items, max_tokens=80)

    # 4. Evaluate Experimental Stage-E (Pure Neural)
    print("\n--- Evaluating Experimental Stage-E (Pure Neural) ---")
    engine_e = ChakrViewInferenceEngine17M(ckpt_e)
    results["stage_e_pure_neural"] = evaluate_benchmark("ChakrGenesis-17M Stage-E Prior", engine_e, "pure_neural", items, max_tokens=80)

    # Save summary and complete details
    with open(out_dir / "eval_260_summary.json", "w", encoding="utf-8") as f:
        json.dump({k: v["summary"] for k, v in results.items()}, f, indent=2)

    with open(out_dir / "eval_260_details.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nAll 260-task evaluations complete! Results saved to: {out_dir}")


if __name__ == "__main__":
    run_comprehensive_eval()
