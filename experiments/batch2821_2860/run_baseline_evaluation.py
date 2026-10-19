"""Evaluates the Canonical Step-10,000 baseline on the held-out SFT Evaluation Set.

Runs generation without optimizer steps and produces machine-readable baseline outputs.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chakrview.runtime.neural_runtime_adapter import NeuralRuntimeAdapter

logger = logging.getLogger("baseline_eval")


def run_baseline_evaluation():
    exp_dir = REPO_ROOT / "experiments" / "batch2821_2860"
    eval_set_path = exp_dir / "SFT_EVALUATION_SET.json"
    results_path = exp_dir / "SFT_BASELINE_EVALUATION_RESULTS.json"

    with open(eval_set_path, "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    cases: List[Dict[str, Any]] = eval_data["cases"]
    ckpt_path = REPO_ROOT / "experiments" / "batch16_21_combined" / "main_training_10000" / "training_checkpoint_step_10000.pt"

    print(f"Loading Canonical Step-10,000 baseline adapter on CPU: {ckpt_path.name}")
    adapter = NeuralRuntimeAdapter(
        checkpoint_path=str(ckpt_path),
        prompt_mode="raw_completion",
        max_new_tokens=32,
        decoding="greedy",
    )

    results: List[Dict[str, Any]] = []
    print(f"Running baseline evaluation across {len(cases)} held-out test cases...")

    for i, c in enumerate(cases, 1):
        cid = c["id"]
        cat = c["category"]
        prompt = c["prompt"]
        target = c["response"]

        gen = adapter.generate(
            prompt=prompt,
            prompt_mode="raw_completion",
            max_new_tokens=32,
            decoding="greedy",
        )

        out_text = gen["generated_text"]
        
        # Automated checks
        exact_match = target.strip().lower() in out_text.strip().lower()
        json_valid = None
        if cat == "structured_output":
            try:
                json.loads(out_text.strip())
                json_valid = True
            except Exception:
                json_valid = False

        abstention_detected = None
        if cat == "uncertainty_insufficient_info":
            abstention_detected = any(phrase in out_text.lower() for phrase in ["not contain", "unavailable", "does not state", "cannot"])

        rec = {
            "id": cid,
            "category": cat,
            "prompt": prompt,
            "target": target,
            "baseline_generated_text": out_text,
            "generated_tokens": gen["output_token_count"],
            "latency_ms": gen["latency_ms"],
            "stop_reason": gen["stop_reason"],
            "target_exact_match": exact_match,
            "json_valid": json_valid,
            "abstention_detected": abstention_detected,
        }
        results.append(rec)
        print(f"[{i}/{len(cases)}] {cid} ({cat}): match={exact_match}, latency={gen['latency_ms']:.1f}ms")

    payload = {
        "evaluation_name": "sft_baseline_step_10000",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "checkpoint_sha256": adapter._checkpoint_sha256,
        "parameter_hash": adapter._bound_weight_digest,
        "total_cases": len(cases),
        "results": results,
    }

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"\nWrote baseline evaluation results to {results_path}")


if __name__ == "__main__":
    run_baseline_evaluation()
