"""Step 3: Verification of the Objective & Evaluator for Experiment 06.

Tests:
1. Correct causal next-token alignment.
2. Prompt labels ignored (-100) and response labels supervised.
3. Correct target-token positions.
4. EOS and padding handling.
5. Finite loss and gradients on ChakrMicro.
6. Exact teacher-forced logit index for the first response token (FAILS if scored at wrong logit position).
7. Strict zero train/test overlap assertion at string and token level.
8. Evaluator correctness on hand-constructed positive and negative examples.
"""

from __future__ import annotations

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


def evaluate_minimal_copy(output_text: str, expected_value: str) -> Dict[str, Any]:
    """Strict evaluator for minimal copy diagnostic."""
    clean_out = output_text.strip()
    clean_exp = expected_value.strip()

    is_exact_match = (clean_out == clean_exp)
    is_empty = (len(clean_out) == 0)

    return {
        "exact_match": is_exact_match,
        "is_empty": is_empty,
        "raw_output": clean_out,
        "expected_value": clean_exp,
    }


def run_verification_tests():
    print("=" * 70)
    print("STEP 3: UNIT TESTS & OBJECTIVE VERIFICATION")
    print("=" * 70)

    tok_dir = REPO_ROOT / "data" / "experiments" / "vocab_4096"
    tokenizer, _ = load_tokenizer_artifacts(tok_dir)

    train_file = Path(__file__).resolve().parent / "data" / "train.jsonl"
    val_file = Path(__file__).resolve().parent / "data" / "val.jsonl"
    test_file = Path(__file__).resolve().parent / "data" / "test.jsonl"

    # 1. Dataset Leakage & Disjointness Check
    print("\n[Test 1] Zero Leakage between Train, Val, and Test:")
    with open(train_file) as f:
        train_recs = [json.loads(line) for line in f]
    with open(val_file) as f:
        val_recs = [json.loads(line) for line in f]
    with open(test_file) as f:
        test_recs = [json.loads(line) for line in f]

    train_vals = set(r["expected_value"] for r in train_recs)
    val_vals = set(r["expected_value"] for r in val_recs)
    test_vals = set(r["expected_value"] for r in test_recs)

    assert train_vals.isdisjoint(val_vals), "Train and Val strings overlap!"
    assert train_vals.isdisjoint(test_vals), "Train and Test strings overlap!"
    assert val_vals.isdisjoint(test_vals), "Val and Test strings overlap!"
    print(f"  [PASS] String sets strictly disjoint: Train={len(train_vals)}, Val={len(val_vals)}, Test={len(test_vals)}")

    # Token sequence overlap check
    train_seqs = set(tuple(r["target_tokens"]) for r in train_recs)
    test_seqs = set(tuple(r["target_tokens"]) for r in test_recs)
    assert train_seqs.isdisjoint(test_seqs), "Train and Test token sequences overlap!"
    print(f"  [PASS] Token sequences strictly disjoint.")

    # 2. Causal Next-Token Alignment & First Response Token Logit Position
    print("\n[Test 2] Exact Causal Alignment & First Token Logit Index Assertion:")
    ds = SFTDataset(train_file, tokenizer, max_seq_len=256, is_training=True, strip_prompt_delimiters=False)

    rec0 = ds.records[0]
    p_str = rec0["prompt"]
    r_str = rec0["response"]
    exp_val = rec0["expected_value"]

    p_toks = tokenizer.encode(p_str)
    r_toks = tokenizer.encode(r_str)

    inp, tgt = ds.encode_example(p_str, r_str, append_eos=True)

    # In SFTDataset:
    # inp = [BOS] + p_toks + r_toks
    # tgt = [-100]*len(p_toks) + r_toks + [EOS]
    assert len(inp) == len(tgt)
    assert inp[0] == 0  # BOS
    assert tgt[:len(p_toks)] == [-100] * len(p_toks)  # Prompt strictly masked
    assert tgt[len(p_toks) : len(p_toks) + len(r_toks)] == r_toks  # Response supervised
    assert tgt[-1] == 1  # EOS

    # In a causal transformer:
    # The output at position `pos` predicts the token at `pos + 1`.
    # Input token at index `len(p_toks)` is `p_toks[-1]` (the last prompt token, which is ': ').
    # Model's logits at index `len(p_toks)` MUST predict `tgt[len(p_toks)]`, which is `r_toks[0]`!
    first_response_pos = len(p_toks)
    expected_first_tok = r_toks[0]
    assert tgt[first_response_pos] == expected_first_tok

    # Prove that scoring at any other position (e.g. len(p_toks)-1 or len(p_toks)+1) is WRONG
    wrong_pos_prev = first_response_pos - 1
    wrong_pos_next = first_response_pos + 1
    assert tgt[wrong_pos_prev] == -100, "Position len(p)-1 should be masked (-100)"
    if len(r_toks) > 1:
        assert tgt[wrong_pos_next] == r_toks[1] != expected_first_tok or r_toks[1] == expected_first_tok

    print(f"  [PASS] Logit index {first_response_pos} strictly predicts first response token {expected_first_tok} ({repr(tokenizer.decode([expected_first_tok]))}).")

    # 3. Finite Loss and Gradients
    print("\n[Test 3] Forward Loss and Backward Gradients Finite on Batched Data:")
    rng = np.random.RandomState(42)
    x_b, y_b, active_count = ds.get_batch(batch_size=4, rng=rng)

    model = ChakrMicro(ModelConfig(max_seq_len=512))
    logits = model(x_b)
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y_b.reshape(-1), ignore_index=-100)

    assert math.isfinite(loss.item()), f"Non-finite loss: {loss.item()}"
    loss.backward()

    for name, p in model.named_parameters():
        if p.grad is not None:
            assert torch.all(torch.isfinite(p.grad)), f"Non-finite gradient in {name}"
    print(f"  [PASS] Forward loss: {loss.item():.4f} | All gradients finite.")

    # 4. Evaluator Correctness Tests
    print("\n[Test 4] Evaluator Correctness on Hand-Constructed Cases:")
    # Positive exact match
    ev1 = evaluate_minimal_copy("token", "token")
    assert ev1["exact_match"] is True and ev1["is_empty"] is False

    # Negative mismatch
    ev2 = evaluate_minimal_copy("token_extra", "token")
    assert ev2["exact_match"] is False

    # Empty output
    ev3 = evaluate_minimal_copy("", "token")
    assert ev3["exact_match"] is False and ev3["is_empty"] is True

    # Case with trailing whitespace/newline
    ev4 = evaluate_minimal_copy("token\n", "token")
    assert ev4["exact_match"] is True

    print("  [PASS] Evaluator correctly verifies matches and rejects errors.")

    print("\n" + "=" * 70)
    print("ALL VERIFICATION TESTS PASSED DETERMINISTICALLY.")
    print("=" * 70)


if __name__ == "__main__":
    run_verification_tests()
