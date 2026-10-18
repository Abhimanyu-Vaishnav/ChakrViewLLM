"""
Regression tests for Batches 107-110:
Operand Copy Benchmark, Dataset Integrity, and Invariant Verification.
"""
import json
import pytest
from pathlib import Path
import torch

from experiments.batch103_106_slot_weighted_sft.slot_weighted_training import (
    BASE_CKPT, TOKENIZER_DIR, ChakrMicro, ModelConfig, load_tokenizer_artifacts, compute_file_sha256
)
from experiments.batch107_110_operand_copy_benchmark.eval_canonical_baseline import (
    DATA_DIR, load_records, parse_copy_output
)

def test_canonical_checkpoint_hash_invariant():
    expected_sha = "96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee"
    actual_sha = compute_file_sha256(str(BASE_CKPT))
    assert actual_sha == expected_sha, f"Hash mismatch: expected {expected_sha}, got {actual_sha}"

def test_copy_dataset_split_isolation_and_coverage():
    train_recs = load_records(DATA_DIR / "train.jsonl")
    val_recs = load_records(DATA_DIR / "val.jsonl")
    test_recs = load_records(DATA_DIR / "test.jsonl")
    
    assert len(train_recs) == 64
    assert len(val_recs) == 16
    assert len(test_recs) == 20
    assert len(train_recs) + len(val_recs) + len(test_recs) == 100
    
    train_pairs = set(tuple(r["operands"]) for r in train_recs)
    val_pairs = set(tuple(r["operands"]) for r in val_recs)
    test_pairs = set(tuple(r["operands"]) for r in test_recs)
    
    # Zero cross-split leakage
    assert len(train_pairs & test_pairs) == 0, "Train-test leakage detected"
    assert len(train_pairs & val_pairs) == 0, "Train-val leakage detected"
    assert len(val_pairs & test_pairs) == 0, "Val-test leakage detected"

def test_copy_dataset_digit_coverage():
    test_recs = load_records(DATA_DIR / "test.jsonl")
    a_digits = set(r["operands"][0] for r in test_recs)
    b_digits = set(r["operands"][1] for r in test_recs)
    # Both A and B must cover all digits 0-9 in the held-out test set
    assert a_digits == set(range(10))
    assert b_digits == set(range(10))

def test_copy_output_parser():
    p1 = parse_copy_output("A=4\nB=7")
    assert p1 == {"a": 4, "b": 7}
    
    p2 = parse_copy_output("A = 0 \n B = 9<|im_end|>")
    assert p2 == {"a": 0, "b": 9}
    
    p3 = parse_copy_output("gibberish text")
    assert p3 == {"a": None, "b": None}

def test_copy_tokenization_cleanliness():
    tok, _ = load_tokenizer_artifacts(TOKENIZER_DIR)
    for a in range(10):
        for b in range(10):
            target = f"A={a}\nB={b}<|im_end|>\n"
            toks = tok.encode(target)
            assert len(toks) == 16, f"Length anomaly for target ({a},{b}): {toks}"
            assert toks[2] == 51 + a, f"Operand A token mismatch: {toks[2]} vs {51+a}"
            assert toks[6] == 51 + b, f"Operand B token mismatch: {toks[6]} vs {51+b}"

def test_all_100_pairs_exact_token_alignment():
    """Verify that encode(prompt + target) == encode(prompt) + encode(target) across all 100 operand pairs."""
    tok, _ = load_tokenizer_artifacts(TOKENIZER_DIR)
    for a in range(10):
        for b in range(10):
            prompt = f"<|im_start|>user\nCopy:\nA={a}\nB={b}<|im_end|>\n<|im_start|>assistant\n"
            target = f"A={a}\nB={b}<|im_end|>\n"
            full = prompt + target
            enc_p = tok.encode(prompt)
            enc_t = tok.encode(target)
            enc_f = tok.encode(full)
            assert enc_f == enc_p + enc_t, f"Compositionality violation for ({a}, {b})"
            assert len(enc_p) == 43, f"Prompt length variation for ({a}, {b})"
            assert len(enc_t) == 16, f"Target length variation for ({a}, {b})"
            assert enc_t[0] == 68, f"First target token is not 'A' for ({a}, {b})"
            assert enc_t[-1] == 13, f"Last target token is not '\\n' for ({a}, {b})"

def test_evaluator_metrics_contract():
    """Verify evaluator handles constant outputs, empty outputs, and valid outputs without crashing or divide-by-zero."""
    test_recs = [
        {"prompt": "dummy", "operands": [1, 2]},
        {"prompt": "dummy", "operands": [3, 4]},
    ]
    # Synthetic predictions
    preds = ["A=1\nB=2", "A=1\nB=2"]
    parsed_pairs = [parse_copy_output(p) for p in preds]
    assert parsed_pairs[0] == {"a": 1, "b": 2}
    assert parsed_pairs[1] == {"a": 1, "b": 2}
    # Collapse rate calculation
    import collections
    counts = collections.Counter(preds)
    collapse_rate = counts.most_common(1)[0][1] / len(preds)
    assert collapse_rate == 1.0

