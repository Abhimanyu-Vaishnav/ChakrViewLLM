"""
Tests for Batch 103-106: Slot-Weighted SFT Dynamic Argument Learning.

Covers:
  - Token map completeness (all 10 digits for a and b fields)
  - Dynamic argument position detection correctness
  - Weight mask construction (slot vs. non-slot weights, IGNORE_INDEX passthrough)
  - Training sequence length invariant
  - Slot positions exclusively within the target (non-masked) portion
  - Token-level verification against empirically confirmed values
  - Weighted loss normalizer non-zero guarantee
  - parse_tool_call: valid JSON, broken JSON, missing TOOL_CALL prefix
  - Reproducibility: same seed produces same batch order
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest
import torch

import sys
sys.path.insert(0, ".")

from experiments.batch103_106_slot_weighted_sft.slot_weighted_training import (
    A_VALUE_TOKENS,
    B_VALUE_TOKENS,
    BOS_ID,
    EOS_ID,
    IGNORE_INDEX,
    SLOT_WEIGHT,
    TOK_A,
    TOK_B,
    TOK_COLON_SP,
    TOK_QUOTE,
    VOCAB_SIZE,
    build_training_sequence,
    build_weight_mask,
    find_arg_value_positions,
    load_jsonl,
    parse_tool_call,
)
from experiments.batch42_reproducible_evaluation.evaluation.evaluator import (
    load_tokenizer_artifacts,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
TOKENIZER_DIR = "release/chakrview-v0.5.0/data/experiments/vocab_4096"
DATA_DIR = Path("experiments/batch95_98_slot_binding_diagnostics/data")


@pytest.fixture(scope="module")
def tokenizer():
    tok, _ = load_tokenizer_artifacts(TOKENIZER_DIR)
    return tok


@pytest.fixture(scope="module")
def train_records():
    return load_jsonl(DATA_DIR / "train.jsonl")


@pytest.fixture(scope="module")
def all_records(train_records):
    val = load_jsonl(DATA_DIR / "val.jsonl")
    test = load_jsonl(DATA_DIR / "test.jsonl")
    return train_records + val + test


# ---------------------------------------------------------------------------
# Token map tests
# ---------------------------------------------------------------------------
class TestTokenMaps:
    def test_a_value_tokens_covers_all_digits(self):
        for d in range(10):
            assert d in A_VALUE_TOKENS, f"A_VALUE_TOKENS missing digit {d}"
            assert len(A_VALUE_TOKENS[d]) >= 1

    def test_b_value_tokens_covers_all_digits(self):
        for d in range(10):
            assert d in B_VALUE_TOKENS, f"B_VALUE_TOKENS missing digit {d}"
            assert len(B_VALUE_TOKENS[d]) >= 1

    def test_anchor_constants_are_distinct(self):
        assert TOK_QUOTE != TOK_A
        assert TOK_QUOTE != TOK_B
        assert TOK_A != TOK_B
        assert TOK_COLON_SP not in (TOK_QUOTE, TOK_A, TOK_B)

    def test_a_and_b_maps_share_no_value_tokens_for_non_ambiguous_digits(self, tokenizer):
        # For digits 0-8, a-field and b-field should use different token IDs
        # (because a-field values are followed by comma, b-field by brace)
        for d in range(9):
            a_toks = set(A_VALUE_TOKENS[d])
            b_toks = set(B_VALUE_TOKENS[d])
            # They should not share the same set (different formatting)
            assert a_toks != b_toks, (
                f"a-field and b-field token maps for digit {d} are identical: {a_toks}"
            )

    def test_digit_9_has_two_tokens_in_both_maps(self):
        assert len(A_VALUE_TOKENS[9]) == 2, "Digit 9 in a-field should have 2 tokens"
        assert len(B_VALUE_TOKENS[9]) == 2, "Digit 9 in b-field should have 2 tokens"
        # Both share the '9' token (60) as first token
        assert A_VALUE_TOKENS[9][0] == B_VALUE_TOKENS[9][0] == 60


# ---------------------------------------------------------------------------
# build_training_sequence tests
# ---------------------------------------------------------------------------
class TestBuildTrainingSequence:
    def test_sequence_length_equality(self, tokenizer, train_records):
        for rec in train_records:
            inp, tgt = build_training_sequence(rec, tokenizer)
            assert len(inp) == len(tgt), (
                f"Length mismatch for operands={rec['operands']}: "
                f"inp={len(inp)}, tgt={len(tgt)}"
            )

    def test_bos_is_first_input_token(self, tokenizer, train_records):
        for rec in train_records[:10]:
            inp, _ = build_training_sequence(rec, tokenizer)
            assert inp[0] == BOS_ID

    def test_eos_is_last_input_and_target_token(self, tokenizer, train_records):
        for rec in train_records[:10]:
            inp, tgt = build_training_sequence(rec, tokenizer)
            assert inp[-1] == EOS_ID
            assert tgt[-1] == EOS_ID

    def test_prompt_positions_masked(self, tokenizer, train_records):
        for rec in train_records[:10]:
            inp, tgt = build_training_sequence(rec, tokenizer)
            prompt_len = len(tokenizer.encode(rec["prompt"])) + 1  # +1 for BOS
            for i in range(prompt_len):
                assert tgt[i] == IGNORE_INDEX, (
                    f"Position {i} (prompt region) should be IGNORE_INDEX"
                )

    def test_target_portion_not_fully_masked(self, tokenizer, train_records):
        for rec in train_records[:10]:
            inp, tgt = build_training_sequence(rec, tokenizer)
            non_masked = sum(1 for t in tgt if t != IGNORE_INDEX)
            assert non_masked > 0, f"All target tokens masked for operands={rec['operands']}"


# ---------------------------------------------------------------------------
# find_arg_value_positions tests
# ---------------------------------------------------------------------------
class TestFindArgValuePositions:
    def test_all_training_records_have_nonzero_slots(self, tokenizer, train_records):
        missing = []
        for rec in train_records:
            inp, tgt = build_training_sequence(rec, tokenizer)
            slots = find_arg_value_positions(inp, rec["operands"])
            if len(slots) == 0:
                missing.append(rec["operands"])
        assert len(missing) == 0, f"No slots detected for: {missing}"

    def test_slot_positions_in_target_region(self, tokenizer, train_records):
        for rec in train_records:
            inp, tgt = build_training_sequence(rec, tokenizer)
            slots = find_arg_value_positions(inp, rec["operands"])
            for p in slots:
                assert tgt[p] != IGNORE_INDEX, (
                    f"Slot position {p} is in masked prompt region "
                    f"for operands={rec['operands']}"
                )

    def test_tokens_at_slot_positions_match_value_maps(self, tokenizer, train_records):
        for rec in train_records:
            inp, tgt = build_training_sequence(rec, tokenizer)
            slots = find_arg_value_positions(inp, rec["operands"])
            a_val = int(rec["operands"][0])
            b_val = int(rec["operands"][1])
            toks_at_slots = [inp[p] for p in slots]
            for expected_tok in A_VALUE_TOKENS[a_val]:
                assert expected_tok in toks_at_slots, (
                    f"a={a_val}: expected tok {expected_tok} not in {toks_at_slots} "
                    f"for operands={rec['operands']}"
                )
            for expected_tok in B_VALUE_TOKENS[b_val]:
                assert expected_tok in toks_at_slots, (
                    f"b={b_val}: expected tok {expected_tok} not in {toks_at_slots} "
                    f"for operands={rec['operands']}"
                )

    def test_slot_positions_sorted_and_unique(self, tokenizer, train_records):
        for rec in train_records[:20]:
            inp, tgt = build_training_sequence(rec, tokenizer)
            slots = find_arg_value_positions(inp, rec["operands"])
            assert slots == sorted(set(slots)), "Slot positions should be sorted and unique"

    def test_slot_count_consistent_with_expected_token_lengths(
        self, tokenizer, train_records
    ):
        for rec in train_records:
            inp, tgt = build_training_sequence(rec, tokenizer)
            slots = find_arg_value_positions(inp, rec["operands"])
            a_val = int(rec["operands"][0])
            b_val = int(rec["operands"][1])
            expected_count = len(A_VALUE_TOKENS[a_val]) + len(B_VALUE_TOKENS[b_val])
            assert len(slots) == expected_count, (
                f"operands=({a_val},{b_val}): expected {expected_count} slots, "
                f"got {len(slots)}"
            )


# ---------------------------------------------------------------------------
# build_weight_mask tests
# ---------------------------------------------------------------------------
class TestBuildWeightMask:
    def test_slot_positions_have_slot_weight(self, tokenizer, train_records):
        for rec in train_records[:20]:
            inp, tgt = build_training_sequence(rec, tokenizer)
            slots = find_arg_value_positions(inp, rec["operands"])
            mask = build_weight_mask(len(inp), slots, tgt)
            for p in slots:
                assert abs(mask[p].item() - SLOT_WEIGHT) < 1e-6, (
                    f"Expected {SLOT_WEIGHT} at slot pos {p}, got {mask[p].item()}"
                )

    def test_non_slot_positions_have_weight_one(self, tokenizer, train_records):
        rec = train_records[0]
        inp, tgt = build_training_sequence(rec, tokenizer)
        slots = find_arg_value_positions(inp, rec["operands"])
        mask = build_weight_mask(len(inp), slots, tgt)
        for i in range(len(inp)):
            if i not in slots:
                assert abs(mask[i].item() - 1.0) < 1e-6, (
                    f"Non-slot pos {i} should have weight 1.0, got {mask[i].item()}"
                )

    def test_mask_tensor_shape(self, tokenizer, train_records):
        rec = train_records[0]
        inp, tgt = build_training_sequence(rec, tokenizer)
        slots = find_arg_value_positions(inp, rec["operands"])
        mask = build_weight_mask(len(inp), slots, tgt)
        assert mask.shape == (len(inp),)
        assert mask.dtype == torch.float32

    def test_weighted_normalizer_is_nonzero(self, tokenizer, train_records):
        """The weighted normalizer must be > 0 to prevent division by zero."""
        for rec in train_records[:10]:
            inp, tgt = build_training_sequence(rec, tokenizer)
            slots = find_arg_value_positions(inp, rec["operands"])
            mask = build_weight_mask(len(inp), slots, tgt)
            tgt_tensor = torch.tensor(tgt)
            active = (tgt_tensor != IGNORE_INDEX).float()
            normalizer = (active * mask).sum().item()
            assert normalizer > 1e-8, (
                f"Normalizer is zero for operands={rec['operands']}"
            )


# ---------------------------------------------------------------------------
# parse_tool_call tests
# ---------------------------------------------------------------------------
class TestParseToolCall:
    def test_valid_tool_call_parse(self):
        text = 'TOOL_CALL: {"tool": "calculator", "op": "add", "a": 3, "b": 7}'
        parsed = parse_tool_call(text)
        assert parsed is not None
        assert parsed["tool"] == "calculator"
        assert parsed["a"] == 3
        assert parsed["b"] == 7

    def test_missing_prefix_returns_none(self):
        text = '{"tool": "calculator", "op": "add", "a": 3, "b": 7}'
        assert parse_tool_call(text) is None

    def test_broken_json_returns_none(self):
        text = "TOOL_CALL: {invalid json}"
        assert parse_tool_call(text) is None

    def test_empty_string_returns_none(self):
        assert parse_tool_call("") is None

    def test_integer_operand_values(self):
        text = 'TOOL_CALL: {"tool": "calculator", "op": "add", "a": 0, "b": 9}'
        parsed = parse_tool_call(text)
        assert parsed is not None
        assert parsed["a"] == 0
        assert parsed["b"] == 9

    def test_all_operand_pairs_parseable(self):
        for a in range(10):
            for b in range(10):
                text = (
                    f'TOOL_CALL: {{"tool": "calculator", "op": "add", '
                    f'"a": {a}, "b": {b}}}'
                )
                parsed = parse_tool_call(text)
                assert parsed is not None, f"Failed to parse for a={a}, b={b}"
                assert parsed["a"] == a
                assert parsed["b"] == b


# ---------------------------------------------------------------------------
# Cross-split consistency tests
# ---------------------------------------------------------------------------
class TestCrossSplitConsistency:
    def test_all_splits_have_nonzero_slots(self, tokenizer):
        for sn in ("train", "val", "test"):
            path = DATA_DIR / f"{sn}.jsonl"
            records = load_jsonl(path)
            missing = []
            for rec in records:
                inp, tgt = build_training_sequence(rec, tokenizer)
                slots = find_arg_value_positions(inp, rec["operands"])
                if len(slots) == 0:
                    missing.append(rec["operands"])
            assert len(missing) == 0, (
                f"{sn} split: No slots for operands: {missing}"
            )

    def test_no_operand_pair_repeats_across_val_and_test(self):
        val = load_jsonl(DATA_DIR / "val.jsonl")
        test = load_jsonl(DATA_DIR / "test.jsonl")
        val_pairs = {tuple(r["operands"]) for r in val}
        test_pairs = {tuple(r["operands"]) for r in test}
        overlap = val_pairs & test_pairs
        assert len(overlap) == 0, (
            f"Val and test share operand pairs: {overlap}"
        )


# ---------------------------------------------------------------------------
# Reproducibility test
# ---------------------------------------------------------------------------
class TestReproducibility:
    def test_same_seed_produces_same_batch_order(self, train_records):
        """Deterministic batch sampling with the same seed and state."""
        rng1 = np.random.RandomState(42)
        rng2 = np.random.RandomState(42)
        idxs1 = [rng1.randint(0, len(train_records), size=8).tolist() for _ in range(5)]
        idxs2 = [rng2.randint(0, len(train_records), size=8).tolist() for _ in range(5)]
        assert idxs1 == idxs2, "Same seed should produce identical batch order"

    def test_different_seeds_produce_different_orders(self, train_records):
        rng1 = np.random.RandomState(42)
        rng2 = np.random.RandomState(99)
        idxs1 = [rng1.randint(0, len(train_records), size=8).tolist() for _ in range(10)]
        idxs2 = [rng2.randint(0, len(train_records), size=8).tolist() for _ in range(10)]
        assert idxs1 != idxs2, "Different seeds should produce different batch orders"
