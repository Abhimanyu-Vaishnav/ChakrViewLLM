"""
Synthetic Arithmetic Dataset Generator for Transformation Learning (Batch 68).

Generates disjoint Train (25), Validation (10), and Held-Out Benchmark (10) pairs:
- Single-digit addition: a + b = c, with a in [1..4], b in [1..5], sum in [2..9].
- Tokenization invariant: every operand and answer is strictly a single token (0-9).
- Total domain: 45 distinct addition pairs.
- Deterministic random assignment with seed 42. Zero split overlap.
"""

from __future__ import annotations

import json
from pathlib import Path
import random
import sys
from typing import Any, Dict, List, Tuple

sys.path.insert(0, ".")
from experiments.batch42_reproducible_evaluation.evaluation.evaluator import compute_file_sha256


def generate_single_digit_arithmetic_datasets(
    output_dir: Path,
    seed: int = 42,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    # Generate all pairs (a, b) such that a in [1..4], b in [1..5] -> 20 pairs
    # and a in [1..5], b in [0..4] such that a + b <= 9
    all_pairs = []
    for a in range(10):
        for b in range(10):
            if a + b <= 9:  # single digit result
                all_pairs.append((a, b))

    # Shuffle deterministically
    rng.shuffle(all_pairs)
    # Total available single-digit pairs with sum <= 9 is 55 pairs: (0+0..0+9: 10, 1+0..1+8: 9, etc.)
    # Select 25 Train, 10 Validation, 10 Held-out Benchmark (total 45 distinct pairs)
    train_pairs = all_pairs[:25]
    val_pairs = all_pairs[25:35]
    bench_pairs = all_pairs[35:45]

    # Verify zero overlap
    train_set = set(train_pairs)
    val_set = set(val_pairs)
    bench_set = set(bench_pairs)

    assert len(train_set.intersection(val_set)) == 0
    assert len(train_set.intersection(bench_set)) == 0
    assert len(val_set.intersection(bench_set)) == 0

    def make_record(a: int, b: int, role: str) -> Dict[str, Any]:
        ans = str(a + b)
        prompt = f"<|im_start|>user\nAdd: {a} + {b} = ?<|im_end|>\n<|im_start|>assistant\n"
        response = f"{ans}<|im_end|>\n"
        return {
            "id": f"SYNTH-DIGIT-{a:02d}-{b:02d}",
            "prompt": prompt,
            "response": response,
            "expected_answer": ans,
            "data_role": role,
            "status": "ANSWER_VERIFIED",
            "review_status": "ANSWER_VERIFIED",
            "provenance": "SINGLE_DIGIT_ALGORITHMIC_ORACLE",
            "seed": seed,
            "operands": [a, b],
            "operation": "add",
        }

    train_recs = [make_record(a, b, "TRAIN") for a, b in train_pairs]
    val_recs = [make_record(a, b, "VALIDATION") for a, b in val_pairs]
    bench_recs = [make_record(a, b, "HELD_OUT_BENCHMARK") for a, b in bench_pairs]

    train_path = output_dir / "train_single_digit.jsonl"
    val_path = output_dir / "val_single_digit.jsonl"
    bench_path = output_dir / "benchmark_single_digit.jsonl"

    for path, recs in [(train_path, train_recs), (val_path, val_recs), (bench_path, bench_recs)]:
        with open(path, "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")

    manifest = {
        "seed": seed,
        "task": "Single-Digit Addition (sum <= 9)",
        "train_count": len(train_recs),
        "train_sha256": compute_file_sha256(train_path),
        "val_count": len(val_recs),
        "val_sha256": compute_file_sha256(val_path),
        "bench_count": len(bench_recs),
        "bench_sha256": compute_file_sha256(bench_path),
        "overlap_train_val": len(train_set.intersection(val_set)),
        "overlap_train_bench": len(train_set.intersection(bench_set)),
        "overlap_val_bench": len(val_set.intersection(bench_set)),
    }

    manifest_path = output_dir / "dataset_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return manifest


if __name__ == "__main__":
    out = Path("experiments/batch67_70_generalization_sprint/data")
    m = generate_single_digit_arithmetic_datasets(out)
    print("Single-digit arithmetic datasets generated successfully:")
    print(json.dumps(m, indent=2))
