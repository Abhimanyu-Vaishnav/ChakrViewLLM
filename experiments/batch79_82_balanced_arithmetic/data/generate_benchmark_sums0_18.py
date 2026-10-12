"""
Batch 79: Full Single-Digit Addition Benchmark Generator (Sums 0-18).

Covers all 100 single-digit addition problems (operands a, b in [0, 9], sums a + b in [0, 18]).
Groups pairs into 55 unordered pairs (min(a, b), max(a, b)) so that (a, b) and (b, a)
always reside in the same split, guaranteeing ZERO reversed-pair leakage.

Partitions into:
- Train: 35 unordered pairs (63 ordered pairs), covering all 19 sum classes (0..18).
- Validation: 10 unordered pairs (19 ordered pairs).
- Held-Out Test: 10 unordered pairs (18 ordered pairs).
- Symmetry Diagnostic: all off-diagonal pairs across Validation and Test.
- Template Robustness Diagnostic: alternate phrasing templates for a disjoint probe.
"""

from __future__ import annotations

import collections
import hashlib
import json
from pathlib import Path
import random
import sys
from typing import Any, Dict, List, Set, Tuple

sys.path.insert(0, ".")
from experiments.batch42_reproducible_evaluation.evaluation.evaluator import (
    load_tokenizer_artifacts,
    compute_file_sha256,
)

OUTPUT_DIR = Path("experiments/batch79_82_balanced_arithmetic/data")
TOKENIZER_DIR = "release/chakrview-v0.5.0/data/experiments/vocab_4096"
SEED = 7901


def generate_benchmark_splits(seed: int = SEED) -> Tuple[Dict[str, Any], Dict[str, List[Dict[str, Any]]]]:
    all_ordered = [(a, b) for a in range(10) for b in range(10)]
    unordered_groups = collections.defaultdict(list)
    for a, b in all_ordered:
        u = (min(a, b), max(a, b))
        unordered_groups[u].append((a, b))

    all_u = sorted(list(unordered_groups.keys()))
    assert len(all_u) == 55

    sums_to_u = collections.defaultdict(list)
    for u in all_u:
        s = u[0] + u[1]
        sums_to_u[s].append(u)

    rng = random.Random(seed)

    # 1. Guarantee every sum class 0..18 is represented in training
    train_u = set()
    for s in range(19):
        chosen = rng.choice(sums_to_u[s])
        train_u.add(chosen)

    remaining_u = [u for u in all_u if u not in train_u]
    rng.shuffle(remaining_u)

    test_u = set(remaining_u[:10])
    val_u = set(remaining_u[10:20])
    for u in remaining_u[20:]:
        train_u.add(u)

    assert len(train_u) == 35
    assert len(val_u) == 10
    assert len(test_u) == 10

    # Assert strict unordered disjointness
    assert train_u.isdisjoint(val_u)
    assert train_u.isdisjoint(test_u)
    assert val_u.isdisjoint(test_u)

    # Build ordered records
    splits_records: Dict[str, List[Dict[str, Any]]] = {
        "train": [],
        "validation": [],
        "test_held_out": [],
        "symmetry_diagnostic": [],
        "template_diagnostic": [],
    }

    def _make_record(a: int, b: int, role: str, template_type: str = "standard") -> Dict[str, Any]:
        c = a + b
        if template_type == "standard":
            prompt = f"<|im_start|>user\nAdd: {a} + {b} = ?<|im_end|>\n<|im_start|>assistant\n"
        elif template_type == "query":
            prompt = f"<|im_start|>user\nWhat is {a} + {b}?<|im_end|>\n<|im_start|>assistant\n"
        elif template_type == "word":
            prompt = f"<|im_start|>user\nCalculate the sum of {a} and {b}.<|im_end|>\n<|im_start|>assistant\n"
        else:
            prompt = f"<|im_start|>user\nAdd: {a} + {b} = ?<|im_end|>\n<|im_start|>assistant\n"

        response = f"{c}<|im_end|>\n"
        rec_id = f"ADD-{role[:3]}-{a:02d}-{b:02d}"
        if template_type != "standard":
            rec_id += f"-{template_type.upper()}"

        return {
            "id": rec_id,
            "prompt": prompt,
            "response": response,
            "expected_answer": str(c),
            "data_role": role,
            "operands": [a, b],
            "unordered_pair": [min(a, b), max(a, b)],
            "sum": c,
            "symmetric_partner": [b, a],
            "template_type": template_type,
        }

    # Train
    for u in sorted(train_u):
        for a, b in unordered_groups[u]:
            splits_records["train"].append(_make_record(a, b, "TRAIN"))

    # Validation
    for u in sorted(val_u):
        for a, b in unordered_groups[u]:
            splits_records["validation"].append(_make_record(a, b, "VALIDATION"))

    # Held-Out Test
    for u in sorted(test_u):
        for a, b in unordered_groups[u]:
            splits_records["test_held_out"].append(_make_record(a, b, "HELD_OUT_TEST"))

    # Symmetry Diagnostic: off-diagonal pairs in validation and test
    for split_name in ["validation", "test_held_out"]:
        for rec in splits_records[split_name]:
            a, b = rec["operands"]
            if a != b:
                sym_rec = _make_record(a, b, "SYMMETRY_DIAGNOSTIC")
                sym_rec["original_split"] = split_name
                splits_records["symmetry_diagnostic"].append(sym_rec)

    # Template Diagnostic: evaluate test pairs under alternate phrasing templates
    for rec in splits_records["test_held_out"]:
        a, b = rec["operands"]
        splits_records["template_diagnostic"].append(_make_record(a, b, "TEMPLATE_DIAGNOSTIC", "query"))
        splits_records["template_diagnostic"].append(_make_record(a, b, "TEMPLATE_DIAGNOSTIC", "word"))

    # Compute constant-output baselines
    train_answers = [r["expected_answer"] for r in splits_records["train"]]
    most_common_train_ans, most_common_count = collections.Counter(train_answers).most_common(1)[0]
    train_baseline_acc = most_common_count / len(train_answers)

    val_answers = [r["expected_answer"] for r in splits_records["validation"]]
    val_baseline_acc = sum(1 for a in val_answers if a == most_common_train_ans) / len(val_answers)

    test_answers = [r["expected_answer"] for r in splits_records["test_held_out"]]
    test_baseline_acc = sum(1 for a in test_answers if a == most_common_train_ans) / len(test_answers)

    manifest = {
        "generator": "generate_benchmark_sums0_18.py",
        "generator_version": "2.0.0",
        "seed": seed,
        "total_domain_pairs": 100,
        "total_unordered_pairs": 55,
        "sum_range": [0, 18],
        "splits": {
            "train": {
                "unordered_pairs_count": len(train_u),
                "ordered_pairs_count": len(splits_records["train"]),
                "sum_classes_covered": sorted(list(set(r["sum"] for r in splits_records["train"]))),
                "unordered_pairs": [list(u) for u in sorted(train_u)],
            },
            "validation": {
                "unordered_pairs_count": len(val_u),
                "ordered_pairs_count": len(splits_records["validation"]),
                "sum_classes_covered": sorted(list(set(r["sum"] for r in splits_records["validation"]))),
                "unordered_pairs": [list(u) for u in sorted(val_u)],
            },
            "test_held_out": {
                "unordered_pairs_count": len(test_u),
                "ordered_pairs_count": len(splits_records["test_held_out"]),
                "sum_classes_covered": sorted(list(set(r["sum"] for r in splits_records["test_held_out"]))),
                "unordered_pairs": [list(u) for u in sorted(test_u)],
            },
            "symmetry_diagnostic": {
                "ordered_pairs_count": len(splits_records["symmetry_diagnostic"]),
            },
            "template_diagnostic": {
                "records_count": len(splits_records["template_diagnostic"]),
            },
        },
        "constant_output_baseline": {
            "most_common_training_answer": most_common_train_ans,
            "train_baseline_accuracy": round(train_baseline_acc, 4),
            "val_baseline_accuracy": round(val_baseline_acc, 4),
            "test_baseline_accuracy": round(test_baseline_acc, 4),
        },
        "disjointness_audit": {
            "train_val_ordered_overlap": 0,
            "train_test_ordered_overlap": 0,
            "val_test_ordered_overlap": 0,
            "train_val_unordered_overlap": 0,
            "train_test_unordered_overlap": 0,
            "val_test_unordered_overlap": 0,
            "reversed_pair_leakage": 0,
        }
    }

    return manifest, splits_records


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest, records = generate_benchmark_splits(seed=SEED)

    # Save jsonl split files
    file_hashes = {}
    for split_name, recs in records.items():
        split_path = OUTPUT_DIR / f"{split_name}.jsonl"
        with open(split_path, "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        sha = compute_file_sha256(str(split_path))
        file_hashes[split_name] = sha
        print(f"Saved {split_name:22s} ({len(recs):2d} recs) -> SHA: {sha}")

    manifest["file_hashes"] = file_hashes
    manifest_path = OUTPUT_DIR / "benchmark_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nBenchmark Generation Complete! Manifest written to {manifest_path}")
    print(f"Constant Output Baseline (predicting '{manifest['constant_output_baseline']['most_common_training_answer']}'):")
    print(f"  Train: {manifest['constant_output_baseline']['train_baseline_accuracy']*100:.1f}%")
    print(f"  Val:   {manifest['constant_output_baseline']['val_baseline_accuracy']*100:.1f}%")
    print(f"  Test:  {manifest['constant_output_baseline']['test_baseline_accuracy']*100:.1f}%")


if __name__ == "__main__":
    main()
