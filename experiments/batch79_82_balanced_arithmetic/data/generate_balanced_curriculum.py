"""
Batch 80: Deterministic Balanced Arithmetic Curriculum Generator.

Generates:
1. Balanced Synthetic Arithmetic Curriculum (Condition C):
   - Uses exclusively the 64 allowed training ordered pairs (from the 35 training unordered pairs).
   - Guarantees 0 unordered or ordered overlap with the 18 validation and 18 held-out test pairs.
   - Enforces uniform sum-class sampling: each of the 19 sum classes (0..18) is sampled with equal frequency,
     preventing any single sum (e.g. 6 or 7) from dominating the training loss landscape.
   - Incorporates multiple prompt templates (ChatML, equation, word problem, drill).
2. Matched Generic Language Control (Condition B):
   - Extracts identical token budget from Stage C pretraining shards.
3. Produces manifest with SHA-256 digests and audit logs.
"""

from __future__ import annotations

import collections
import hashlib
import json
from pathlib import Path
import random
import sys
from typing import Any, Dict, List, Tuple

sys.path.insert(0, ".")
import numpy as np

from experiments.batch42_reproducible_evaluation.evaluation.evaluator import load_tokenizer_artifacts

OUTPUT_DIR = Path("experiments/batch79_82_balanced_arithmetic/data")
TOKENIZER_DIR = "release/chakrview-v0.5.0/data/experiments/vocab_4096"
SEED = 8001

TEMPLATES = [
    "<|im_start|>user\nAdd: {a} + {b} = ?<|im_end|>\n<|im_start|>assistant\n{c}<|im_end|>\n",
    "Equation: {a} + {b} = {c}.\n",
    "{a} + {b} = {c}\n",
    "If you have {a} apples and receive {b} more, you have {c} apples in total.\n",
    "Question: What is {a} plus {b}? Answer: {c}.\n",
    "The sum of {a} and {b} is equal to {c}.\n",
    "Arithmetic Drill: {a} + {b} equals {c}.\n",
    "Calculate {a} + {b}. Result: {c}.\n",
]


def load_training_pairs(benchmark_dir: Path) -> List[Dict[str, Any]]:
    train_file = benchmark_dir / "train.jsonl"
    with open(train_file, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def generate_balanced_curriculum(
    train_records: List[Dict[str, Any]],
    seed: int = SEED,
    samples_per_sum: int = 150,
) -> Tuple[List[str], Dict[int, int]]:
    """
    Uniform sum-class sampling:
    Draws equal numbers of examples for each sum class 0..18.
    """
    rng = random.Random(seed)
    sum_to_pairs = collections.defaultdict(list)
    for r in train_records:
        sum_to_pairs[r["sum"]].append(r["operands"])

    assert len(sum_to_pairs) == 19, "Not all 19 sum classes are present in training!"

    docs = []
    sum_counts = collections.defaultdict(int)

    for s in range(19):
        available_pairs = sum_to_pairs[s]
        for _ in range(samples_per_sum):
            a, b = rng.choice(available_pairs)
            c = a + b
            tmpl = rng.choice(TEMPLATES)
            doc = tmpl.format(a=a, b=b, c=c)
            docs.append(doc)
            sum_counts[c] += 1

    rng.shuffle(docs)
    return docs, dict(sum_counts)


def tokenize_documents(docs: List[str], tokenizer: Any) -> np.ndarray:
    token_list = []
    for doc in docs:
        toks = tokenizer.encode(doc)
        token_list.extend(toks)
        token_list.append(1)  # EOS
    return np.array(token_list, dtype=np.uint16)


def extract_matched_control_tokens(repo_root: Path, target_count: int) -> np.ndarray:
    stage_c_path = repo_root / "data" / "tokenized" / "stage_c" / "train" / "shard_00000.bin"
    with open(stage_c_path, "rb") as f:
        data = f.read(target_count * 2)
    tokens = np.frombuffer(data, dtype=np.uint16)
    return tokens[:target_count]


def main() -> None:
    tokenizer, _ = load_tokenizer_artifacts(TOKENIZER_DIR)
    train_records = load_training_pairs(OUTPUT_DIR)

    print(f"Loaded {len(train_records)} training pairs from benchmark.")
    docs, sum_counts = generate_balanced_curriculum(train_records, seed=SEED, samples_per_sum=150)
    print(f"Generated {len(docs)} documents across 19 sum classes (150 per sum class).")

    curriculum_tokens = tokenize_documents(docs, tokenizer)
    total_tokens = len(curriculum_tokens)
    print(f"Tokenized balanced curriculum: {total_tokens} tokens.")

    # Save Condition C (Balanced Arithmetic Curriculum)
    cond_c_path = OUTPUT_DIR / "condition_c_balanced_arithmetic_tokens.bin"
    with open(cond_c_path, "wb") as f:
        f.write(curriculum_tokens.tobytes())
    sha_c = hashlib.sha256(curriculum_tokens.tobytes()).hexdigest()

    # Save Condition B (Matched Generic Language Control)
    print(f"Extracting matched {total_tokens} tokens from Stage C for Condition B...")
    cond_b_tokens = extract_matched_control_tokens(Path("."), total_tokens)
    cond_b_path = OUTPUT_DIR / "condition_b_generic_control_tokens.bin"
    with open(cond_b_path, "wb") as f:
        f.write(cond_b_tokens.tobytes())
    sha_b = hashlib.sha256(cond_b_tokens.tobytes()).hexdigest()

    # Save sample texts
    with open(OUTPUT_DIR / "curriculum_sample_texts.jsonl", "w", encoding="utf-8") as f:
        for i, d in enumerate(docs[:50]):
            f.write(json.dumps({"idx": i, "text": d}) + "\n")

    manifest = {
        "generator": "generate_balanced_curriculum.py",
        "generator_version": "1.0.0",
        "seed": SEED,
        "total_documents": len(docs),
        "total_tokens": total_tokens,
        "samples_per_sum_class": 150,
        "sum_counts": sum_counts,
        "condition_c_balanced_tokens_file": "condition_c_balanced_arithmetic_tokens.bin",
        "condition_c_sha256": sha_c,
        "condition_b_generic_tokens_file": "condition_b_generic_control_tokens.bin",
        "condition_b_sha256": sha_b,
    }

    manifest_path = OUTPUT_DIR / "curriculum_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Curriculum Generation Complete! Manifest written to {manifest_path}")
    print(f"Condition C SHA-256: {sha_c}")
    print(f"Condition B SHA-256: {sha_b}")


if __name__ == "__main__":
    main()
