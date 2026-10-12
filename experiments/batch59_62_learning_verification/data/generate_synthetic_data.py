"""
Controlled Deterministic Synthetic Arithmetic Generator (Batch 60).
Generates strictly partitioned, non-overlapping pairs for single/double digit addition:
- TRAIN: 32 pairs
- VALIDATION: 8 pairs (never seen in training)
- HELD_OUT_BENCHMARK: 8 pairs (never seen in training or validation)
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
from typing import Dict, List, Tuple


def generate_arithmetic_datasets(
    seed: int = 42,
    output_dir: Path = Path("experiments/batch59_62_learning_verification/data"),
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    # Generate all pairs 1 <= a <= 20, 1 <= b <= 20
    all_pairs = []
    for a in range(1, 21):
        for b in range(1, 21):
            all_pairs.append((a, b))

    # Shuffle deterministically
    rng.shuffle(all_pairs)

    train_pairs = all_pairs[:32]
    val_pairs = all_pairs[32:40]
    bench_pairs = all_pairs[40:48]

    # Verify zero overlap
    assert len(set(train_pairs).intersection(set(val_pairs))) == 0
    assert len(set(train_pairs).intersection(set(bench_pairs))) == 0
    assert len(set(val_pairs).intersection(set(bench_pairs))) == 0

    def make_record(a: int, b: int, role: str) -> Dict[str, Any]:
        ans = a + b
        return {
            "id": f"SYNTH-ADD-{a:02d}-{b:02d}",
            "prompt": f"<|im_start|>user\nCompute: {a} + {b} = ?<|im_end|>\n<|im_start|>assistant\n",
            "response": f"{ans}<|im_end|>\n",
            "expected_answer": str(ans),
            "data_role": role,
            "status": "ANSWER_VERIFIED",
            "review_status": "ANSWER_VERIFIED",
            "provenance": "ALGORITHMIC_GENERATOR_V1",
            "seed": seed,
            "family": f"ADD_{a:02d}_{b:02d}",
        }

    train_recs = [make_record(a, b, "TRAIN") for a, b in train_pairs]
    val_recs = [make_record(a, b, "VALIDATION") for a, b in val_pairs]
    bench_recs = [make_record(a, b, "HELD_OUT_BENCHMARK") for a, b in bench_pairs]

    train_path = output_dir / "train_synthetic.jsonl"
    val_path = output_dir / "val_synthetic.jsonl"
    bench_path = output_dir / "benchmark_synthetic.jsonl"

    with open(train_path, "w", encoding="utf-8") as f:
        for r in train_recs:
            f.write(json.dumps(r) + "\n")

    with open(val_path, "w", encoding="utf-8") as f:
        for r in val_recs:
            f.write(json.dumps(r) + "\n")

    with open(bench_path, "w", encoding="utf-8") as f:
        for r in bench_recs:
            f.write(json.dumps(r) + "\n")

    def sha(path: Path) -> str:
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()

    manifest = {
        "generator_version": "1.0.0",
        "seed": seed,
        "total_generated": len(train_recs) + len(val_recs) + len(bench_recs),
        "train_count": len(train_recs),
        "val_count": len(val_recs),
        "bench_count": len(bench_recs),
        "train_sha256": sha(train_path),
        "val_sha256": sha(val_path),
        "bench_sha256": sha(bench_path),
        "overlap_train_val": 0,
        "overlap_train_bench": 0,
        "overlap_val_bench": 0,
    }

    manifest_path = output_dir / "synthetic_dataset_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return manifest


if __name__ == "__main__":
    m = generate_arithmetic_datasets()
    print("Generated synthetic arithmetic datasets:")
    print(json.dumps(m, indent=2))
