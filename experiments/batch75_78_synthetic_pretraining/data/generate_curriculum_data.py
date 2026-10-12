"""
Batch 76: Deterministic Synthetic Mathematical Curriculum Data Generator.

Generates:
1. Strict Single-Digit Addition Curriculum:
   - Built exclusively from the 30 canonical training operand pairs from Batch 72.
   - 0 leakage to the 10 validation pairs and 10 held-out test pairs.
   - Features multiple templates:
     * Standard arithmetic equation: 'a + b = c'
     * Conversational user/assistant prompt: '<|im_start|>user\\nAdd: a + b = ?<|im_end|>\\n<|im_start|>assistant\\nc<|im_end|>'
     * Word problem: 'If you have a apples and get b more apples, you have c apples.'
     * Natural query: 'What is the sum of a and b? The sum is c.'
     * Drill style: 'Problem: a plus b. Answer: c.'
2. Strict split audit asserting zero contamination with validation and test pairs.
3. Serialization of token streams for Condition B (Synthetic Arithmetic Curriculum)
   and matched token streams for Condition A (Stage C Language Control).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
import sys
from typing import Any, Dict, List, Set, Tuple

sys.path.insert(0, ".")
import numpy as np

from experiments.batch42_reproducible_evaluation.evaluation.evaluator import load_tokenizer_artifacts

TOKENIZER_DIR = "release/chakrview-v0.5.0/data/experiments/vocab_4096"
BENCHMARK_DIR = Path("experiments/batch71_74_controlled_learning/data")
OUTPUT_DIR = Path("experiments/batch75_78_synthetic_pretraining/data")


def load_benchmark_pairs() -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]], List[Tuple[int, int]], List[Tuple[int, int]]]:
    """Loads benchmark pairs from Batch 72 records."""
    def _read_pairs(p: Path) -> List[Tuple[int, int]]:
        with open(p, "r", encoding="utf-8") as f:
            return [tuple(json.loads(line)["operands"]) for line in f if line.strip()]

    train_pairs = _read_pairs(BENCHMARK_DIR / "train_arithmetic.jsonl")
    val_pairs = _read_pairs(BENCHMARK_DIR / "val_arithmetic.jsonl")
    test_pairs = _read_pairs(BENCHMARK_DIR / "test_held_out.jsonl")
    symm_pairs = _read_pairs(BENCHMARK_DIR / "symmetry_diagnostic.jsonl")
    return train_pairs, val_pairs, test_pairs, symm_pairs


TEMPLATES = [
    # 1. ChatML format (matching evaluation prompt style)
    "<|im_start|>user\nAdd: {a} + {b} = ?<|im_end|>\n<|im_start|>assistant\n{c}<|im_end|>\n",
    # 2. Direct equation
    "Equation: {a} + {b} = {c}.\n",
    # 3. Plain arithmetic expression
    "{a} + {b} = {c}\n",
    # 4. Natural language word problem
    "If you have {a} apples and receive {b} more, you have {c} apples in total.\n",
    # 5. Question and answer
    "Question: What is {a} plus {b}? Answer: {c}.\n",
    # 6. Sum query
    "The sum of {a} and {b} is equal to {c}.\n",
    # 7. Drill problem
    "Arithmetic Drill: {a} + {b} equals {c}.\n",
    # 8. Step-by-step identity
    "Calculate {a} + {b}. Result: {c}.\n",
]


def generate_curriculum_documents(
    train_pairs: List[Tuple[int, int]],
    seed: int = 7601,
    target_docs: int = 1500,
) -> List[str]:
    """
    Generates synthetic documents using ONLY allowed training pairs.
    """
    rng = random.Random(seed)
    docs = []
    for _ in range(target_docs):
        # Pick a training pair
        a, b = rng.choice(train_pairs)
        c = a + b
        tmpl = rng.choice(TEMPLATES)
        text = tmpl.format(a=a, b=b, c=c)
        docs.append(text)
    return docs


def tokenize_documents(docs: List[str], tokenizer: Any) -> np.ndarray:
    """Tokenizes list of documents into a single contiguous uint16 token stream separated by EOS."""
    token_list = []
    for doc in docs:
        toks = tokenizer.encode(doc)
        token_list.extend(toks)
        # Append EOS token id 1
        token_list.append(1)
    return np.array(token_list, dtype=np.uint16)


def extract_matched_control_tokens(repo_root: Path, target_token_count: int) -> np.ndarray:
    """
    Extracts exactly target_token_count tokens from Stage C training shards
    to serve as Condition A (Matched Language Control).
    """
    stage_c_train = repo_root / "data" / "tokenized" / "stage_c" / "train" / "shard_00000.bin"
    with open(stage_c_train, "rb") as f:
        data = f.read(target_token_count * 2)
    tokens = np.frombuffer(data, dtype=np.uint16)
    if len(tokens) < target_token_count:
        raise ValueError(f"Shard smaller than requested target tokens: {len(tokens)} < {target_token_count}")
    return tokens[:target_token_count]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer, _ = load_tokenizer_artifacts(TOKENIZER_DIR)
    train_pairs, val_pairs, test_pairs, symm_pairs = load_benchmark_pairs()

    print(f"Loaded {len(train_pairs)} train pairs, {len(val_pairs)} val pairs, {len(test_pairs)} test pairs.")

    # 1. Verification of disjointness
    val_set = set(val_pairs)
    test_set = set(test_pairs)
    assert len(set(train_pairs) & val_set) == 0, "Train and Val overlap!"
    assert len(set(train_pairs) & test_set) == 0, "Train and Test overlap!"
    assert len(val_set & test_set) == 0, "Val and Test overlap!"

    # 2. Generate curriculum documents
    print("Generating curriculum documents for Condition B...")
    docs = generate_curriculum_documents(train_pairs, seed=7601, target_docs=2500)
    curriculum_tokens = tokenize_documents(docs, tokenizer)
    total_tokens = len(curriculum_tokens)
    print(f"Generated {len(docs)} documents -> {total_tokens} tokens.")

    # Save curriculum raw texts for auditing
    sample_texts_path = OUTPUT_DIR / "curriculum_sample_texts.jsonl"
    with open(sample_texts_path, "w", encoding="utf-8") as f:
        for i, d in enumerate(docs[:50]):
            f.write(json.dumps({"idx": i, "text": d}) + "\n")

    # Save binary tokens for Condition B
    cond_b_path = OUTPUT_DIR / "condition_b_arithmetic_tokens.bin"
    with open(cond_b_path, "wb") as f:
        f.write(curriculum_tokens.tobytes())

    # 3. Extract matched tokens for Condition A (Control)
    print(f"Extracting matched {total_tokens} tokens for Condition A from Stage C...")
    cond_a_tokens = extract_matched_control_tokens(Path("."), total_tokens)
    cond_a_path = OUTPUT_DIR / "condition_a_control_tokens.bin"
    with open(cond_a_path, "wb") as f:
        f.write(cond_a_tokens.tobytes())

    # 4. Hash verification
    sha_b = hashlib.sha256(curriculum_tokens.tobytes()).hexdigest()
    sha_a = hashlib.sha256(cond_a_tokens.tobytes()).hexdigest()

    manifest = {
        "generator": "generate_curriculum_data.py",
        "generator_version": "1.0.0",
        "seed": 7601,
        "num_documents": len(docs),
        "total_tokens": total_tokens,
        "templates_count": len(TEMPLATES),
        "allowed_train_pairs": [list(p) for p in sorted(train_pairs)],
        "isolated_val_pairs": [list(p) for p in sorted(val_pairs)],
        "isolated_test_pairs": [list(p) for p in sorted(test_pairs)],
        "condition_b_tokens_file": "condition_b_arithmetic_tokens.bin",
        "condition_b_sha256": sha_b,
        "condition_a_tokens_file": "condition_a_control_tokens.bin",
        "condition_a_sha256": sha_a,
        "split_integrity": {
            "train_val_overlap": 0,
            "train_test_overlap": 0,
            "val_test_overlap": 0,
            "leakage_audited": True,
        }
    }

    manifest_path = OUTPUT_DIR / "curriculum_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("Batch 76 Data Generation Complete!")
    print(f"Manifest written to {manifest_path}")
    print(f"Condition A SHA-256: {sha_a}")
    print(f"Condition B SHA-256: {sha_b}")


if __name__ == "__main__":
    main()
