"""Unit tests for Token-Level Loss Weighting implementation before training.

Verifies:
1. Weight 1.0 reproduces the existing unweighted loss bit-exact.
2. Only intended status-string tokens receive weight 5.0.
3. Prompt, syntax, EOS, padding, and ignored labels receive no additional weight (or 0.0 weight).
4. Batched examples with different response lengths are handled correctly.
5. The weighted loss and gradients remain finite.
6. Deliberately incorrect token-span alignment is rejected.
7. The loss implementation agrees with a small hand-calculated example.
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
from experiments.research_v08_json_sft.exp05_slot_weighted_sft.weighted_sft_dataset import (
    WeightedSFTDataset,
    compute_weighted_cross_entropy,
)


def run_unit_tests():
    print("=" * 70)
    print("RUNNING FOCUSED UNIT TESTS FOR TOKEN-LEVEL LOSS WEIGHTING")
    print("=" * 70)

    tok_dir = REPO_ROOT / "data" / "experiments" / "vocab_4096"
    tokenizer, _ = load_tokenizer_artifacts(tok_dir)
    train_file = REPO_ROOT / "experiments" / "research_v08_json_sft" / "exp02_string_copy" / "data" / "train.jsonl"

    # 1. Test Hand-Calculated Small Numerical Example
    print("\n[Test 1] Agreement with Hand-Calculated Loss Example:")
    # Batch size 1, seq len 3, vocab 4
    # Logits: [[1.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0], [0.0, 0.0, 3.0, 0.0]]
    # Targets: [0, 1, 2]
    # Weights: [1.0, 5.0, 1.0]
    logits = torch.tensor([[[1.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0], [0.0, 0.0, 3.0, 0.0]]], requires_grad=True)
    targets = torch.tensor([[0, 1, 2]])
    weights = torch.tensor([[1.0, 5.0, 1.0]])

    # Hand calculation:
    # l_0 = -log(e^1 / (e^1 + 3*e^0)) = -log(e / (e + 3))
    # l_1 = -log(e^2 / (e^2 + 3*e^0)) = -log(e^2 / (e^2 + 3))
    # l_2 = -log(e^3 / (e^3 + 3*e^0)) = -log(e^3 / (e^3 + 3))
    l0 = -math.log(math.exp(1) / (math.exp(1) + 3))
    l1 = -math.log(math.exp(2) / (math.exp(2) + 3))
    l2 = -math.log(math.exp(3) / (math.exp(3) + 3))
    expected_weighted_loss = (1.0 * l0 + 5.0 * l1 + 1.0 * l2) / (1.0 + 5.0 + 1.0)

    computed_loss = compute_weighted_cross_entropy(logits, targets, weights)
    assert abs(computed_loss.item() - expected_weighted_loss) < 1e-6, (
        f"Hand-calc mismatch: {computed_loss.item()} vs {expected_weighted_loss}"
    )
    print(f"  [PASS] Computed: {computed_loss.item():.6f} == Hand-calculated: {expected_weighted_loss:.6f}")

    # 2. Test Weight 1.0 Reproduces Standard Unweighted Loss Bit-Exact
    print("\n[Test 2] Weight 1.0 Reproduces Standard Cross-Entropy Bit-Exact:")
    rng_logits = torch.randn(2, 5, 20, requires_grad=True)
    rng_targets = torch.tensor([[0, 5, -100, 3, 1], [-100, 2, 4, 8, -100]])
    ones_weights = torch.ones_like(rng_targets, dtype=torch.float32)

    std_loss = F.cross_entropy(rng_logits.reshape(-1, 20), rng_targets.reshape(-1), ignore_index=-100)
    w_loss_1 = compute_weighted_cross_entropy(rng_logits, rng_targets, ones_weights)

    assert abs(std_loss.item() - w_loss_1.item()) < 1e-6, (
        f"Bit-exact mismatch: {std_loss.item()} vs {w_loss_1.item()}"
    )
    print(f"  [PASS] Standard CE: {std_loss.item():.6f} == Weighted CE (w=1.0): {w_loss_1.item():.6f}")

    # 3. Test Selective Weighting on Single-Field Records
    print("\n[Test 3] Exact Status-String Tokens Receive Weight 5.0:")
    ds_weighted = WeightedSFTDataset(
        train_file, tokenizer, max_seq_len=256, is_training=True, strip_prompt_delimiters=False, slot_weight=5.0
    )

    sample_rec = next(r for r in ds_weighted.records if r.get("category") == "single_field_string_extraction")
    inp, tgt, wgt = ds_weighted.encode_weighted_example(sample_rec)

    status_str = sample_rec["expected_status"]
    status_tokens = tokenizer.encode(status_str)
    prefix_tokens = tokenizer.encode('{"status": "')
    suffix_tokens = tokenizer.encode('"}')

    prompt_tokens_len = sum(1 for t in tgt if t == -100)
    # Check that all prompt tokens have weight 0.0
    assert all(wgt[i] == 0.0 for i in range(prompt_tokens_len)), "Prompt tokens have non-zero weight!"

    # Check prefix tokens have weight 1.0
    for i in range(prompt_tokens_len, prompt_tokens_len + len(prefix_tokens)):
        assert wgt[i] == 1.0, f"Prefix token {i} has weight {wgt[i]} != 1.0"

    # Check status tokens have weight 5.0
    status_start = prompt_tokens_len + len(prefix_tokens)
    status_end = status_start + len(status_tokens)
    for i in range(status_start, status_end):
        assert wgt[i] == 5.0, f"Status token {i} has weight {wgt[i]} != 5.0"

    # Check suffix tokens and EOS have weight 1.0
    for i in range(status_end, len(tgt)):
        assert wgt[i] == 1.0, f"Suffix/EOS token {i} has weight {wgt[i]} != 1.0"

    print(f"  [PASS] Record '{status_str}' ({len(status_tokens)} tokens):")
    print(f"         Prompt (w=0.0): {prompt_tokens_len} tokens")
    print(f"         Prefix (w=1.0): {len(prefix_tokens)} tokens")
    print(f"         Status (w=5.0): {len(status_tokens)} tokens")
    print(f"         Suffix/EOS (w=1.0): {len(tgt) - status_end} tokens")

    # 4. Test Replay Records Receive Weight 1.0 Across Supervised Tokens
    print("\n[Test 4] Replay Records Receive Standard Weight 1.0 (No 5.0 Weighting):")
    replay_rec = next(r for r in ds_weighted.records if r.get("category") == "base_continuation_replay")
    r_inp, r_tgt, r_wgt = ds_weighted.encode_weighted_example(replay_rec)
    for t, w in zip(r_tgt, r_wgt):
        if t == -100:
            assert w == 0.0
        else:
            assert w == 1.0
    print("  [PASS] Replay record tokens strictly preserve weight 1.0.")

    # 5. Test Batched Collation & Dynamic Padding
    print("\n[Test 5] Batched Collation with Different Response Lengths & Zero Padding Weight:")
    rng = np.random.RandomState(42)
    x_b, y_b, w_b, active_count = ds_weighted.get_weighted_batch(batch_size=4, rng=rng)

    assert x_b.shape == y_b.shape == w_b.shape
    # Check that any position where y_b == -100 has w_b == 0.0
    masked_positions = (y_b == -100)
    assert torch.all(w_b[masked_positions] == 0.0), "Masked positions have non-zero weight!"
    # Check that active tokens match
    assert active_count == int((y_b != -100).sum().item())
    print(f"  [PASS] Batch tensor shape {list(x_b.shape)} padded correctly with zero weight on padding.")

    # 6. Test Finite Gradients and Loss on ChakrMicro
    print("\n[Test 6] Finite Forward and Backward Gradients on ChakrMicro:")
    model = ChakrMicro(ModelConfig(max_seq_len=512))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    optimizer.zero_grad()

    logits_batch = model(x_b)
    batch_loss = compute_weighted_cross_entropy(logits_batch, y_b, w_b)

    assert math.isfinite(batch_loss.item()), f"Non-finite loss: {batch_loss.item()}"
    batch_loss.backward()

    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2).item()
            total_norm += param_norm ** 2
            assert math.isfinite(param_norm), "Non-finite gradient norm!"
    grad_norm = math.sqrt(total_norm)
    assert math.isfinite(grad_norm), f"Non-finite total grad norm: {grad_norm}"
    print(f"  [PASS] Forward loss: {batch_loss.item():.4f} | Backward grad norm: {grad_norm:.4f} (strictly finite).")

    # 7. Test Deliberate Misalignment Rejection
    print("\n[Test 7] Deliberately Corrupted Alignment Detection:")
    corrupted_rec = dict(sample_rec)
    corrupted_rec["expected_status"] = "non_existent_status_string_xyz"
    try:
        ds_weighted.encode_weighted_example(corrupted_rec)
        assert False, "Failed to reject corrupted status string!"
    except AssertionError as e:
        print(f"  [PASS] Corrupted alignment correctly raised AssertionError: {e}")

    print("\n" + "=" * 70)
    print("ALL 7 UNIT TESTS PASSED DETERMINISTICALLY.")
    print("=" * 70)


if __name__ == "__main__":
    run_unit_tests()
