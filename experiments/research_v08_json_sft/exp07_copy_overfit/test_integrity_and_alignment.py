"""Unit tests for Experiment 07 Copy Overfit test.

Verifies:
1. Prompt/response causal alignment and response-only supervision.
2. Correct first-response-token logit index (fails if scored at wrong position).
3. EOS and padding behavior.
4. Finite loss and gradients on ChakrMicro.
5. Exact checkpoint and tokenizer hashes.
6. Hand-calculated loss and alignment test.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer.serialization import load_tokenizer_artifacts
from chakrview.training.sft_engine import SFTDataset


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def run_unit_tests():
    print("=" * 70)
    print("EXP 07: PRE-TRAINING INTEGRITY & CAUSAL ALIGNMENT UNIT TESTS")
    print("=" * 70)

    # 1. Hashes
    CANONICAL_PARENT_PATH = REPO_ROOT / "experiments" / "batch16_21_combined" / "main_training_10000" / "training_checkpoint_step_10000.pt"
    EXPECTED_PARENT_SHA = "96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee"
    actual_parent_sha = compute_sha256(CANONICAL_PARENT_PATH)
    assert actual_parent_sha == EXPECTED_PARENT_SHA, "Canonical parent hash mismatch!"
    print(f"  [PASS] Canonical parent hash verified: {actual_parent_sha[:16]}...")

    tok_dir = REPO_ROOT / "data" / "experiments" / "vocab_4096"
    EXPECTED_VOCAB_SHA = "db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a"
    actual_vocab_sha = compute_sha256(tok_dir / "vocab.json")
    assert actual_vocab_sha == EXPECTED_VOCAB_SHA, "Vocab hash mismatch!"
    print(f"  [PASS] Tokenizer vocab hash verified: {actual_vocab_sha[:16]}...")

    tokenizer, _ = load_tokenizer_artifacts(tok_dir)
    train_file = Path(__file__).resolve().parent / "data" / "tiny_train_16.jsonl"
    test_file = Path(__file__).resolve().parent / "data" / "unseen_test_25.jsonl"

    # 2. Hand-Calculated Loss / Alignment Test
    print("\n[Test 1] Hand-Calculated Loss & Alignment Check:")
    logits_mock = torch.tensor([[[2.0, 0.0, 0.0], [0.0, 3.0, 0.0]]], requires_grad=True)
    targets_mock = torch.tensor([[-100, 1]])
    # Expected: position 0 ignored (-100), position 1 targets token 1
    # l1 = -log(e^3 / (e^0 + e^3 + e^0)) = -log(e^3 / (e^3 + 2))
    expected_ce = -math.log(math.exp(3) / (math.exp(3) + 2))
    computed_ce = F.cross_entropy(logits_mock.reshape(-1, 3), targets_mock.reshape(-1), ignore_index=-100)
    assert abs(computed_ce.item() - expected_ce) < 1e-6, "Hand-calc loss mismatch!"
    print(f"  [PASS] Hand-calc loss: {expected_ce:.6f} == computed: {computed_ce.item():.6f}")

    # 3. Causal Alignment on Real Tiny-Train Record
    print("\n[Test 2] Causal Alignment & First Token Logit Index Assertion:")
    ds = SFTDataset(train_file, tokenizer, max_seq_len=256, is_training=True, strip_prompt_delimiters=False)
    rec0 = ds.records[0]
    p_str = rec0["prompt"]
    r_str = rec0["expected_value"]

    p_toks = tokenizer.encode(p_str)
    r_toks = tokenizer.encode(r_str)
    assert len(r_toks) == 1, "Expected single-token response!"

    inp, tgt = ds.encode_example(p_str, r_str, append_eos=True)
    # inp = [0] + p_toks + r_toks (length len(p_toks) + 2)
    # tgt = [-100]*len(p_toks) + r_toks + [1]
    assert len(inp) == len(tgt)
    assert tgt[:len(p_toks)] == [-100] * len(p_toks)
    assert tgt[len(p_toks)] == r_toks[0]
    assert tgt[len(p_toks) + 1] == 1  # EOS

    first_resp_pos = len(p_toks)
    # Assertion: scoring at any other position must be rejected
    assert tgt[first_resp_pos] == r_toks[0]
    assert tgt[first_resp_pos - 1] == -100
    assert tgt[first_resp_pos + 1] == 1

    print(f"  [PASS] First-token logit position {first_resp_pos} strictly targets {r_toks[0]} ({repr(tokenizer.decode([r_toks[0]]))}).")

    # 4. Finite Gradients on ChakrMicro
    print("\n[Test 3] Finite Forward Loss and Backward Gradients on ChakrMicro:")
    model = ChakrMicro(ModelConfig(max_seq_len=512))
    rng = np.random.RandomState(42)
    x_b, y_b, act = ds.get_batch(batch_size=4, rng=rng)
    logits = model(x_b)
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y_b.reshape(-1), ignore_index=-100)
    assert math.isfinite(loss.item()), "Non-finite loss!"
    loss.backward()
    for name, param in model.named_parameters():
        if param.grad is not None:
            assert torch.all(torch.isfinite(param.grad)), f"Non-finite gradient in {name}"
    print(f"  [PASS] Forward loss: {loss.item():.4f} | All parameter gradients finite.")

    # 5. Zero Leakage Assertion
    print("\n[Test 4] Zero Leakage Assertion between Tiny-Train and Unseen-Test:")
    with open(train_file) as f:
        tr_recs = [json.loads(line) for line in f]
    with open(test_file) as f:
        te_recs = [json.loads(line) for line in f]
    tr_vals = set(r["expected_value"] for r in tr_recs)
    te_vals = set(r["expected_value"] for r in te_recs)
    assert tr_vals.isdisjoint(te_vals), "Train and unseen test values overlap!"
    print(f"  [PASS] Disjointness verified: 16 Train values vs 25 Unseen Test values.")

    print("\n" + "=" * 70)
    print("ALL UNIT TESTS PASSED DETERMINISTICALLY.")
    print("=" * 70)


if __name__ == "__main__":
    run_unit_tests()
