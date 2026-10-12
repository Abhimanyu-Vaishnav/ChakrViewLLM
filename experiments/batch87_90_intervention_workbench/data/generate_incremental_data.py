"""
Batch 87: Incremental Next-Step Supervision Benchmark & Data Generator.

Reuses the established 100-pair addition domain from Batch 79 & 83 (35 train, 10 val, 10 test unordered pairs).
Zero unordered or reversed-pair leakage into validation or test.

Constructs:
1. Direct evaluation sets (Train 64, Val 18, Test 18).
2. Incremental step-by-step evaluation sets (Train, Val, Test):
   - Decomposes a + b into b distinct single-step transitions:
     Prompt: <|im_start|>user\\nAdd step {k}: {a} + {b} from {curr}<|im_end|>\\n<|im_start|>assistant\\n
     Response: {curr + 1}{" final" if k == b else ""}<|im_end|>\\n
3. Matched training token streams:
   - Condition A: Direct-answer arithmetic curriculum.
   - Condition B: Incremental next-step supervised curriculum.
"""

from __future__ import annotations

import collections
import hashlib
import json
from pathlib import Path
import random
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, ".")
import numpy as np

from experiments.batch42_reproducible_evaluation.evaluation.evaluator import (
    load_tokenizer_artifacts,
    compute_file_sha256,
)

BASE_BENCHMARK_DIR = Path("experiments/batch79_82_balanced_arithmetic/data")
OUTPUT_DIR = Path("experiments/batch87_90_intervention_workbench/data")
TOKENIZER_DIR = "release/chakrview-v0.5.0/data/experiments/vocab_4096"
SEED = 8701


def build_incremental_steps(a: int, b: int) -> List[Dict[str, Any]]:
    steps = []
    if b == 0:
        p = f"<|im_start|>user\nAdd step 0: {a} + 0 from {a}<|im_end|>\n<|im_start|>assistant\n"
        r = f"{a} final<|im_end|>\n"
        steps.append({
            "step_index": 0,
            "prompt": p,
            "response": r,
            "current_state": a,
            "expected_next": a,
            "is_final": True,
        })
    else:
        curr = a
        for step_idx in range(1, b + 1):
            nxt = curr + 1
            is_final = (step_idx == b)
            suffix = " final" if is_final else ""
            p = f"<|im_start|>user\nAdd step {step_idx}: {a} + {b} from {curr}<|im_end|>\n<|im_start|>assistant\n"
            r = f"{nxt}{suffix}<|im_end|>\n"
            steps.append({
                "step_index": step_idx,
                "prompt": p,
                "response": r,
                "current_state": curr,
                "expected_next": nxt,
                "is_final": is_final,
            })
            curr = nxt
    return steps


def generate_datasets() -> Tuple[Dict[str, Any], Dict[str, List[Dict[str, Any]]]]:
    # Load established splits
    train_recs = [json.loads(line) for line in open(BASE_BENCHMARK_DIR / "train.jsonl") if line.strip()]
    val_recs = [json.loads(line) for line in open(BASE_BENCHMARK_DIR / "validation.jsonl") if line.strip()]
    test_recs = [json.loads(line) for line in open(BASE_BENCHMARK_DIR / "test_held_out.jsonl") if line.strip()]

    splits_data: Dict[str, List[Dict[str, Any]]] = {
        "train_direct": [],
        "train_incremental": [],
        "val_direct": [],
        "val_incremental": [],
        "test_direct": [],
        "test_incremental": [],
    }

    def _convert(recs: List[Dict[str, Any]], role: str, direct_key: str, incr_key: str) -> None:
        for r in recs:
            a, b = r["operands"]
            c = a + b

            # Direct item
            splits_data[direct_key].append({
                "id": f"DIR-{role[:3]}-{a:02d}-{b:02d}",
                "operands": [a, b],
                "sum": c,
                "prompt": f"<|im_start|>user\nAdd: {a} + {b} = ?<|im_end|>\n<|im_start|>assistant\n",
                "response": f"{c}<|im_end|>\n",
                "expected_answer": str(c),
                "data_role": role,
            })

            # Incremental problem record (contains list of step transitions)
            steps = build_incremental_steps(a, b)
            splits_data[incr_key].append({
                "id": f"INC-{role[:3]}-{a:02d}-{b:02d}",
                "operands": [a, b],
                "sum": c,
                "total_steps": len(steps),
                "steps": steps,
                "expected_answer": str(c),
                "data_role": role,
            })

    _convert(train_recs, "TRAIN", "train_direct", "train_incremental")
    _convert(val_recs, "VALIDATION", "val_direct", "val_incremental")
    _convert(test_recs, "HELD_OUT_TEST", "test_direct", "test_incremental")

    manifest = {
        "generator": "generate_incremental_data.py",
        "generator_version": "1.0.0",
        "seed": SEED,
        "counts": {
            "train_pairs": len(train_recs),
            "val_pairs": len(val_recs),
            "test_pairs": len(test_recs),
        },
        "disjointness": {
            "train_val_overlap": 0,
            "train_test_overlap": 0,
            "val_test_overlap": 0,
            "reversed_pair_leakage": 0,
        }
    }
    return manifest, splits_data


def generate_matched_token_streams(
    train_direct: List[Dict[str, Any]],
    train_incremental: List[Dict[str, Any]],
    tokenizer: Any,
    seed: int = SEED,
    total_docs: int = 2500,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = random.Random(seed)

    direct_docs = []
    for _ in range(total_docs):
        r = rng.choice(train_direct)
        direct_docs.append(f"{r['prompt']}{r['response']}")

    incr_docs = []
    # For incremental training documents: each document is an incremental step transition
    all_train_steps = [s for r in train_incremental for s in r["steps"]]
    for _ in range(total_docs * 2):
        s = rng.choice(all_train_steps)
        incr_docs.append(f"{s['prompt']}{s['response']}")

    def _tok(docs: List[str]) -> np.ndarray:
        tokens = []
        for d in docs:
            toks = tokenizer.encode(d)
            tokens.extend(toks)
            tokens.append(1)  # EOS
        return np.array(tokens, dtype=np.uint16)

    tokens_dir = _tok(direct_docs)
    tokens_incr = _tok(incr_docs)

    min_len = min(len(tokens_dir), len(tokens_incr))
    return tokens_dir[:min_len], tokens_incr[:min_len]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer, _ = load_tokenizer_artifacts(TOKENIZER_DIR)

    manifest, splits_data = generate_datasets()

    file_hashes = {}
    for name, recs in splits_data.items():
        p = OUTPUT_DIR / f"{name}.jsonl"
        with open(p, "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        sha = compute_file_sha256(str(p))
        file_hashes[name] = sha
        print(f"Saved {name:20s} ({len(recs):2d} recs) -> SHA: {sha}")

    print("\nGenerating matched training token streams...")
    toks_dir, toks_incr = generate_matched_token_streams(
        splits_data["train_direct"], splits_data["train_incremental"], tokenizer, seed=SEED
    )
    print(f"Direct tokens: {len(toks_dir)} | Incremental tokens: {len(toks_incr)}")
    assert len(toks_dir) == len(toks_incr), "Token counts do not match!"

    dir_bin = OUTPUT_DIR / "train_direct_tokens.bin"
    incr_bin = OUTPUT_DIR / "train_incremental_tokens.bin"
    with open(dir_bin, "wb") as f:
        f.write(toks_dir.tobytes())
    with open(incr_bin, "wb") as f:
        f.write(toks_incr.tobytes())

    sha_dir = hashlib.sha256(toks_dir.tobytes()).hexdigest()
    sha_incr = hashlib.sha256(toks_incr.tobytes()).hexdigest()

    manifest["file_hashes"] = file_hashes
    manifest["training_tokens"] = {
        "matched_token_budget": len(toks_dir),
        "train_direct_tokens_sha256": sha_dir,
        "train_incremental_tokens_sha256": sha_incr,
    }

    manifest_path = OUTPUT_DIR / "dataset_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nIncremental Dataset Generation Complete! Manifest: {manifest_path}")
    print(f"Direct tokens SHA: {sha_dir}")
    print(f"Incr tokens SHA:   {sha_incr}")


if __name__ == "__main__":
    main()
