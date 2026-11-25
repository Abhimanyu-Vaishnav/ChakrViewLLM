"""
ChakrView Stage-E Evaluation Runner (210 Tasks).
Mission CV-GENESIS-NR-013 Phase 6, 7 & 8.

Evaluates Stage-E Best Checkpoint on the fresh 210-task sealed benchmark:
1. Pure Neural Mode
2. Hybrid Cognitive Mode
"""

from __future__ import annotations
import os
import sys
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chakrview.runtime.inference_17m import ChakrViewInferenceEngine17M
from experiments.baseline_eval_013.run_baseline_eval import evaluate_benchmark


def run_stage_e_eval():
    bench_file = Path("data/sealed_eval_013/sealed_benchmark_210.json")
    with open(bench_file, "r", encoding="utf-8") as f:
        items = json.load(f)

    ckpt_path = "experiments/genesis_stage_e_013/checkpoints/genesis_17m_stage_e_best.pt"
    print(f"Loading Stage-E Best model from: {ckpt_path}")
    engine = ChakrViewInferenceEngine17M(ckpt_path)

    out_dir = Path("experiments/stage_e_eval_013")
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    # 1. Pure Neural Mode
    print("\n--- Evaluating Stage-E 17M Best (Pure Neural Mode) ---")
    results["stage_e_pure_neural"] = evaluate_benchmark(
        "ChakrGenesis-17M Stage-E Best", engine, "pure_neural", items
    )

    # 2. Hybrid Cognitive Mode
    print("\n--- Evaluating Stage-E 17M Best (Hybrid Cognitive Mode) ---")
    results["stage_e_hybrid"] = evaluate_benchmark(
        "ChakrGenesis-17M Stage-E Best", engine, "hybrid", items
    )

    # Save summary and details
    with open(out_dir / "stage_e_eval_summary.json", "w", encoding="utf-8") as f:
        json.dump({k: v["summary"] for k, v in results.items()}, f, indent=2)

    with open(out_dir / "stage_e_eval_details.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nStage-E benchmark evaluation complete! Saved to {out_dir}")


if __name__ == "__main__":
    run_stage_e_eval()
