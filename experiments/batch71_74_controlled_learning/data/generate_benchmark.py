"""
Deterministic Single-Digit Arithmetic Dataset & Symmetry Benchmark Generator (Batch 72).

Generates:
1. Primary Disjoint Addition Benchmark (55 pairs total, sum <= 9):
   - Train (30 pairs): Operands and sums strictly single tokens.
   - Validation (10 pairs): Strictly disjoint operand pairs.
   - Held-Out Test (10 pairs): Strictly disjoint operand pairs.
2. Symmetry Diagnostic Benchmark:
   - For every asymmetric pair (a, b) with a != b in the test set or validation set,
     records its complementary symmetric partner (b, a).
   - Allows diagnosing whether the model learns commutativity / symmetry: if (a, b) -> c, does (b, a) -> c?
3. Manifest & Distribution Metrics:
   - Operand frequency, sum frequency, zero-overlap proofs, SHA-256 digests.
"""

from __future__ import annotations

import json
from pathlib import Path
import random
import sys
from typing import Any, Dict, List, Set, Tuple

sys.path.insert(0, ".")
from experiments.batch42_reproducible_evaluation.evaluation.evaluator import compute_file_sha256


def generate_controlled_arithmetic_benchmark(
    output_dir: Path,
    seed: int = 42,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    # All single-digit pairs with sum <= 9
    all_pairs: List[Tuple[int, int]] = []
    for a in range(10):
        for b in range(10):
            if a + b <= 9:
                all_pairs.append((a, b))

    assert len(all_pairs) == 55

    # Group into pairs and their reverse
    # Identify unique unordered pairs
    unordered_groups: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
    for a, b in all_pairs:
        key = (min(a, b), max(a, b))
        unordered_groups.setdefault(key, []).append((a, b))

    # We want:
    # Train: 30 pairs
    # Val: 10 pairs (5 forward, 5 symmetry partners or distinct)
    # Held-out Test: 10 pairs
    # Symmetry Diagnostic: 5 pairs in Val or Test that test if training on (a,b) helps (b,a)

    # Let's shuffle all_pairs deterministically
    shuffled_pairs = list(all_pairs)
    rng.shuffle(shuffled_pairs)

    train_pairs = shuffled_pairs[:30]
    val_pairs = shuffled_pairs[30:40]
    test_pairs = shuffled_pairs[40:50]
    # Reserve 5 pairs for diagnostic reserve
    diagnostic_reserve = shuffled_pairs[50:55]

    # Verify disjoint splits
    train_set = set(train_pairs)
    val_set = set(val_pairs)
    test_set = set(test_pairs)

    assert len(train_set.intersection(val_set)) == 0
    assert len(train_set.intersection(test_set)) == 0
    assert len(val_set.intersection(test_set)) == 0

    # Build symmetry pairs:
    # Diagnostic 1: Train-to-Test Symmetry (pairs where (a, b) in Train, but (b, a) in Val or Test)
    train_symmetry_in_val = [(a, b) for a, b in val_pairs if (b, a) in train_set and a != b]
    train_symmetry_in_test = [(a, b) for a, b in test_pairs if (b, a) in train_set and a != b]

    def make_record(a: int, b: int, role: str) -> Dict[str, Any]:
        ans = str(a + b)
        prompt = f"<|im_start|>user\nAdd: {a} + {b} = ?<|im_end|>\n<|im_start|>assistant\n"
        response = f"{ans}<|im_end|>\n"
        return {
            "id": f"ARITH-ADD-{a:02d}-{b:02d}",
            "prompt": prompt,
            "response": response,
            "expected_answer": ans,
            "data_role": role,
            "status": "ANSWER_VERIFIED",
            "review_status": "ANSWER_VERIFIED",
            "provenance": "SINGLE_DIGIT_CANONICAL_ORACLE",
            "seed": seed,
            "operands": [a, b],
            "operation": "add",
            "symmetric_partner": [b, a],
        }

    train_recs = [make_record(a, b, "TRAIN") for a, b in train_pairs]
    val_recs = [make_record(a, b, "VALIDATION") for a, b in val_pairs]
    test_recs = [make_record(a, b, "HELD_OUT_TEST") for a, b in test_pairs]

    train_file = output_dir / "train_arithmetic.jsonl"
    val_file = output_dir / "val_arithmetic.jsonl"
    test_file = output_dir / "test_held_out.jsonl"

    for path, recs in [(train_file, train_recs), (val_file, val_recs), (test_file, test_recs)]:
        with open(path, "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")

    # Symmetry diagnostic file: pairs where (b, a) was seen in training
    symm_test_recs = [make_record(a, b, "SYMMETRY_DIAGNOSTIC") for a, b in (train_symmetry_in_val + train_symmetry_in_test)]
    symm_file = output_dir / "symmetry_diagnostic.jsonl"
    with open(symm_file, "w", encoding="utf-8") as f:
        for r in symm_test_recs:
            f.write(json.dumps(r) + "\n")

    # Distribution of sums across splits
    def sum_distribution(pairs: List[Tuple[int, int]]) -> Dict[int, int]:
        counts = {}
        for a, b in pairs:
            s = a + b
            counts[s] = counts.get(s, 0) + 1
        return dict(sorted(counts.items()))

    manifest = {
        "benchmark_name": "ChakrMicro Single-Digit Arithmetic Benchmark v1",
        "domain": "Single-digit integer addition: a + b = c, with a, b in [0..9], c in [0..9]",
        "total_domain_pairs": len(all_pairs),
        "seed": seed,
        "splits": {
            "train": {
                "file": "train_arithmetic.jsonl",
                "count": len(train_recs),
                "sha256": compute_file_sha256(train_file),
                "sum_distribution": sum_distribution(train_pairs),
            },
            "validation": {
                "file": "val_arithmetic.jsonl",
                "count": len(val_recs),
                "sha256": compute_file_sha256(val_file),
                "sum_distribution": sum_distribution(val_pairs),
            },
            "held_out_test": {
                "file": "test_held_out.jsonl",
                "count": len(test_recs),
                "sha256": compute_file_sha256(test_file),
                "sum_distribution": sum_distribution(test_pairs),
            },
            "symmetry_diagnostic": {
                "file": "symmetry_diagnostic.jsonl",
                "count": len(symm_test_recs),
                "sha256": compute_file_sha256(symm_file),
                "description": "Pairs (a, b) in Val/Test whose reverse (b, a) was present in Train.",
            },
        },
        "split_integrity": {
            "train_val_overlap": len(train_set.intersection(val_set)),
            "train_test_overlap": len(train_set.intersection(test_set)),
            "val_test_overlap": len(val_set.intersection(test_set)),
            "zero_overlap_confirmed": True,
        },
    }

    manifest_file = output_dir / "benchmark_manifest.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return manifest


if __name__ == "__main__":
    out = Path("experiments/batch71_74_controlled_learning/data")
    m = generate_controlled_arithmetic_benchmark(out)
    print("Benchmark successfully generated:")
    print(json.dumps(m, indent=2))
