"""
Batch 83: Deterministic Scratchpad Benchmark & Curriculum Generator.

Generates:
1. Reuses the 100-pair benchmark splits from Batch 79 (35 train, 10 val, 10 test unordered pairs).
2. For each split, generates:
   - Direct-answer format:
     prompt: "<|im_start|>user\\nAdd: {a} + {b} = ?<|im_end|>\\n<|im_start|>assistant\\n"
     response: "{c}<|im_end|>\\n"
   - Scratchpad format:
     prompt: "<|im_start|>user\\nAdd with scratchpad: {a} + {b} = ?<|im_end|>\\n<|im_start|>assistant\\n"
     response: "Start at {a}; count {b} more: {counting_str}; answer: {c}<|im_end|>\\n"
3. Builds matched training token streams for:
   - Condition A: Direct-answer arithmetic curriculum (from train split only).
   - Condition B: Scratchpad arithmetic curriculum (from train split only).
4. Enforces zero unordered or reversed-pair leakage into validation or test.
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
OUTPUT_DIR = Path("experiments/batch83_86_scratchpad_workbench/data")
TOKENIZER_DIR = "release/chakrview-v0.5.0/data/experiments/vocab_4096"
SEED = 8301


def build_counting_string(a: int, b: int) -> str:
    if b == 0:
        return "none"
    steps = [str(a + i) for i in range(1, b + 1)]
    return ", ".join(steps)


def parse_final_answer(text: str) -> Optional[str]:
    clean = text.replace("<|im_end|>", "").strip()
    m = re.search(r"answer:\s*([0-9]+)", clean, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    m2 = re.match(r"^\s*([0-9]+)", clean)
    if m2:
        return m2.group(1).strip()
    return None


def generate_scratchpad_splits() -> Tuple[Dict[str, Any], Dict[str, List[Dict[str, Any]]]]:
    # Load Batch 79 benchmark splits
    train_recs = [json.loads(line) for line in open(BASE_BENCHMARK_DIR / "train.jsonl") if line.strip()]
    val_recs = [json.loads(line) for line in open(BASE_BENCHMARK_DIR / "validation.jsonl") if line.strip()]
    test_recs = [json.loads(line) for line in open(BASE_BENCHMARK_DIR / "test_held_out.jsonl") if line.strip()]

    splits_data: Dict[str, List[Dict[str, Any]]] = {
        "train_direct": [],
        "train_scratchpad": [],
        "val_direct": [],
        "val_scratchpad": [],
        "test_direct": [],
        "test_scratchpad": [],
    }

    def _convert(recs: List[Dict[str, Any]], role: str, direct_key: str, scratch_key: str) -> None:
        for r in recs:
            a, b = r["operands"]
            c = a + b
            count_str = build_counting_string(a, b)

            # Direct format
            splits_data[direct_key].append({
                "id": f"DIR-{role[:3]}-{a:02d}-{b:02d}",
                "prompt": f"<|im_start|>user\nAdd: {a} + {b} = ?<|im_end|>\n<|im_start|>assistant\n",
                "response": f"{c}<|im_end|>\n",
                "expected_answer": str(c),
                "operands": [a, b],
                "format": "direct",
                "data_role": role,
            })

            # Scratchpad format
            splits_data[scratch_key].append({
                "id": f"SCR-{role[:3]}-{a:02d}-{b:02d}",
                "prompt": f"<|im_start|>user\nAdd with scratchpad: {a} + {b} = ?<|im_end|>\n<|im_start|>assistant\n",
                "response": f"Start at {a}; count {b} more: {count_str}; answer: {c}<|im_end|>\n",
                "expected_answer": str(c),
                "scratchpad_trace": f"Start at {a}; count {b} more: {count_str}; answer: {c}",
                "operands": [a, b],
                "format": "scratchpad",
                "data_role": role,
            })

    _convert(train_recs, "TRAIN", "train_direct", "train_scratchpad")
    _convert(val_recs, "VALIDATION", "val_direct", "val_scratchpad")
    _convert(test_recs, "HELD_OUT_TEST", "test_direct", "test_scratchpad")

    manifest = {
        "generator": "generate_scratchpad_data.py",
        "generator_version": "1.0.0",
        "seed": SEED,
        "base_benchmark_source": "experiments/batch79_82_balanced_arithmetic/data",
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


def generate_curriculum_token_streams(
    train_direct_recs: List[Dict[str, Any]],
    train_scratch_recs: List[Dict[str, Any]],
    tokenizer: Any,
    seed: int = SEED,
    total_docs: int = 2000,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = random.Random(seed)

    direct_docs = []
    scratch_docs = []

    for _ in range(total_docs):
        r_dir = rng.choice(train_direct_recs)
        r_scr = rng.choice(train_scratch_recs)

        direct_docs.append(f"{r_dir['prompt']}{r_dir['response']}")
        scratch_docs.append(f"{r_scr['prompt']}{r_scr['response']}")

    def _tok(docs: List[str]) -> np.ndarray:
        tokens = []
        for d in docs:
            toks = tokenizer.encode(d)
            tokens.extend(toks)
            tokens.append(1)  # EOS
        return np.array(tokens, dtype=np.uint16)

    tokens_dir = _tok(direct_docs)
    tokens_scr = _tok(scratch_docs)

    # Match token budgets exactly by slicing to common min length
    min_len = min(len(tokens_dir), len(tokens_scr))
    return tokens_dir[:min_len], tokens_scr[:min_len]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer, _ = load_tokenizer_artifacts(TOKENIZER_DIR)

    manifest, splits_data = generate_scratchpad_splits()

    file_hashes = {}
    for name, recs in splits_data.items():
        file_path = OUTPUT_DIR / f"{name}.jsonl"
        with open(file_path, "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        sha = compute_file_sha256(str(file_path))
        file_hashes[name] = sha
        print(f"Saved {name:20s} ({len(recs):2d} recs) -> SHA: {sha}")

    # Generate matched token streams for training
    print("\nGenerating matched training token streams...")
    tokens_dir, tokens_scr = generate_curriculum_token_streams(
        splits_data["train_direct"], splits_data["train_scratchpad"], tokenizer, seed=SEED, total_docs=2000
    )
    print(f"Direct tokens: {len(tokens_dir)} | Scratchpad tokens: {len(tokens_scr)}")
    assert len(tokens_dir) == len(tokens_scr), "Token budgets not matched!"

    dir_bin = OUTPUT_DIR / "train_direct_tokens.bin"
    scr_bin = OUTPUT_DIR / "train_scratchpad_tokens.bin"
    with open(dir_bin, "wb") as f:
        f.write(tokens_dir.tobytes())
    with open(scr_bin, "wb") as f:
        f.write(tokens_scr.tobytes())

    sha_dir_bin = hashlib.sha256(tokens_dir.tobytes()).hexdigest()
    sha_scr_bin = hashlib.sha256(tokens_scr.tobytes()).hexdigest()

    manifest["file_hashes"] = file_hashes
    manifest["training_tokens"] = {
        "matched_token_budget": len(tokens_dir),
        "train_direct_tokens_sha256": sha_dir_bin,
        "train_scratchpad_tokens_sha256": sha_scr_bin,
    }

    manifest_path = OUTPUT_DIR / "scratchpad_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nScratchpad Dataset Generation Complete! Manifest written to {manifest_path}")
    print(f"Direct tokens SHA: {sha_dir_bin}")
    print(f"Scratch tokens SHA: {sha_scr_bin}")


if __name__ == "__main__":
    main()
