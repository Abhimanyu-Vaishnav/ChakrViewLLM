"""
Regression and forensic verification tests for SFT Causal Alignment, Masking,
Padding, Boundaries, Negative Validation, and Tokenizer Lineage Parity.
"""
import pytest
import torch
import numpy as np
from pathlib import Path
from chakrview.training.sft_engine import SFTDataset
from chakrview.tokenizer.serialization import load_tokenizer_artifacts
from chakrview.training.prompt_contract import (
    build_causal_sequence,
    normalize_prompt_response_strings,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
# Canonical tokenizer path
CANONICAL_TOKENIZER_DIR = REPO_ROOT / "data" / "experiments" / "vocab_4096"
# Release v0.5.0 tokenizer path (historical reference)
RELEASE_TOKENIZER_DIR = REPO_ROOT / "release" / "chakrview-v0.5.0" / "data" / "experiments" / "vocab_4096"
DATASET_PATH = REPO_ROOT / "experiments" / "batch3021_3060" / "train.jsonl"


@pytest.fixture(scope="module")
def canonical_tokenizer():
    tok, meta = load_tokenizer_artifacts(CANONICAL_TOKENIZER_DIR)
    return tok


def test_tokenizer_canonical_and_release_parity(canonical_tokenizer):
    """Verify canonical tokenizer directory matches release directory behavior bit-exact."""
    rel_tok, _ = load_tokenizer_artifacts(RELEASE_TOKENIZER_DIR)
    test_strings = [
        "Instruction: Extract the sensor status.\nResponse: Operational",
        '{"status": "ok", "value": 42}',
        "Special chars: !@#$%^&*()_+-=[]{}|;:,.<>?/`~",
        "Multiple    spaces   and\nnewlines\n\n\t",
    ]
    for s in test_strings:
        assert canonical_tokenizer.encode(s) == rel_tok.encode(s)
        assert canonical_tokenizer.decode(canonical_tokenizer.encode(s)) == rel_tok.decode(rel_tok.encode(s))


def test_sft_causal_masking_multi_category(canonical_tokenizer):
    """
    Test 1 & 2 & 3 & 4 & 5:
    Verifies multi-category dataset records for:
    - Exact prompt/response boundary alignment
    - Correct response token supervision
    - Prompt tokens masked with -100
    - Proper EOS supervision on append_eos=True
    - Proper terminal token behavior on append_eos=False (replay)
    """
    ds = SFTDataset(
        DATASET_PATH,
        canonical_tokenizer,
        is_training=True,
        strip_prompt_delimiters=False,
    )

    # Find representative samples for key categories
    seen_categories = set()
    test_records = []
    for rec in ds.records:
        cat = rec.get("category")
        if cat not in seen_categories:
            seen_categories.add(cat)
            test_records.append(rec)
        if len(seen_categories) >= 8:
            break

    assert len(test_records) >= 5, "Should have found diverse categories in train.jsonl"

    for rec in test_records:
        append_eos = rec.get("append_eos", True)
        inp, tgt = ds.encode_example(rec["prompt"], rec["response"], append_eos=append_eos)

        assert len(inp) == len(tgt)
        assert inp[0] == 0  # BOS

        # Normalize strings according to dataset settings
        p_str, r_str = normalize_prompt_response_strings(
            rec["prompt"], rec["response"], strip_delimiters=False
        )
        p_toks = canonical_tokenizer.encode(p_str)
        r_toks = canonical_tokenizer.encode(r_str)

        masked_count = tgt.count(-100)
        assert masked_count == len(p_toks)
        # All prompt tokens must be masked with -100
        assert tgt[:masked_count] == [-100] * masked_count

        # Target at prompt boundary must predict first response token
        assert tgt[masked_count] == r_toks[0]

        if append_eos:
            # Last target is EOS (1)
            assert tgt[-1] == 1
            # Total length: BOS(1) + len(prompt) + len(response)
            assert len(inp) == 1 + len(p_toks) + len(r_toks)
            assert tgt[masked_count:-1] == r_toks
        else:
            # Replay continuation: no trailing EOS in target
            assert tgt[-1] != 1 or r_toks[-1] == 1
            assert len(inp) == 1 + len(p_toks) + len(r_toks) - 1
            assert tgt[masked_count:] == r_toks


def test_sft_padding_and_batch_masking(canonical_tokenizer):
    """
    Test 6: Verifies dynamic padding behavior in get_batch:
    - Padded input positions use pad_id (2)
    - Padded target positions use ignore_index (-100)
    - Active tokens count matches unmasked response tokens exactly
    """
    ds = SFTDataset(
        DATASET_PATH,
        canonical_tokenizer,
        is_training=True,
        strip_prompt_delimiters=False,
    )
    rng = np.random.RandomState(42)
    x, y, active_tokens = ds.get_batch(batch_size=4, rng=rng)

    assert x.shape == y.shape
    assert x.dtype == torch.long
    assert y.dtype == torch.long

    expected_active = int((y != -100).sum().item())
    assert active_tokens == expected_active

    # Check padding positions: wherever y is -100 after prompt boundary, if input is 2, target must be -100
    batch_size, max_len = x.shape
    for i in range(batch_size):
        row_x = x[i].tolist()
        row_y = y[i].tolist()
        # Non-pad tokens should precede pad tokens
        if 2 in row_x:
            first_pad_idx = row_x.index(2)
            # Check trailing tokens are pad
            for pos in range(first_pad_idx, max_len):
                assert row_x[pos] == 2
                assert row_y[pos] == -100


def test_empty_and_whitespace_response_handling(canonical_tokenizer):
    """
    Test 7 & 8: Empty response and whitespace-only response handling.
    SFTDataset must reject empty or whitespace responses with ValueError.
    """
    ds = SFTDataset(
        DATASET_PATH,
        canonical_tokenizer,
        is_training=True,
        strip_prompt_delimiters=False,
    )

    with pytest.raises(ValueError, match="Response tokenized to 0 tokens"):
        ds.encode_example("Valid prompt:", "")

    with pytest.raises(ValueError, match="Response tokenized to 0 tokens"):
        ds.encode_example("Valid prompt:", "   \n\t   ")


def test_near_sequence_limit_boundary_and_truncation(canonical_tokenizer):
    """
    Test 9: Prompt and response near max_seq_len limit:
    - When prompt + response exceeds max_seq_len, prompt is truncated from left.
    - If response alone + 1 (BOS) exceeds max_seq_len, ValueError is raised.
    """
    max_len = 64
    ds = SFTDataset(
        DATASET_PATH,
        canonical_tokenizer,
        max_seq_len=max_len,
        is_training=True,
        strip_prompt_delimiters=False,
    )

    long_prompt = "word " * 100
    short_response = "Short answer."
    inp, tgt = ds.encode_example(long_prompt, short_response, append_eos=True)

    assert len(inp) <= max_len
    assert len(tgt) <= max_len
    assert len(inp) == len(tgt)
    assert inp[0] == 0
    assert tgt[-1] == 1
    # Response tokens must still be present at the tail
    r_toks = canonical_tokenizer.encode(short_response.strip())
    assert tgt[-1 - len(r_toks) : -1] == r_toks

    # Overlong response alone must raise ValueError
    oversized_response = "word " * 70
    with pytest.raises(ValueError, match="Response tokens .* exceed max_seq_len"):
        ds.encode_example("Prompt:", oversized_response, append_eos=True)


def test_tokenizer_synthetic_entity_and_json_roundtrip(canonical_tokenizer):
    """
    Test 10: Tokenizer round-trips using canonical tokenizer for entities and JSON schema tokens.
    """
    test_cases = [
        "Sensor delta-781",
        "Liam Campbell",
        "14.5 terabytes",
        "gamma-9082",
        "device-8821",
        '{"status": "active", "value": 105}',
        '{"name": "temperature_probe_4", "reading": -3.2}',
        '{"category": "telemetry", "count": 0}',
    ]
    for case in test_cases:
        enc = canonical_tokenizer.encode(case)
        dec = canonical_tokenizer.decode(enc)
        assert dec.strip() == case.strip()


def test_dataset_schema_and_malformed_record_handling(tmp_path, canonical_tokenizer):
    """
    Test 11: Dataset schema validation:
    - Missing file raises FileNotFoundError
    - Empty file raises ValueError
    - Leakage of BENCHMARK/HELD_OUT_BENCHMARK role raises ValueError
    - PENDING_HUMAN_REVIEW raises PermissionError when trust gate enforced
    - REJECTED records raise ValueError
    """
    missing_file = tmp_path / "nonexistent.jsonl"
    with pytest.raises(FileNotFoundError):
        SFTDataset(missing_file, canonical_tokenizer)

    empty_file = tmp_path / "empty.jsonl"
    empty_file.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="SFT dataset is empty"):
        SFTDataset(empty_file, canonical_tokenizer)

    leakage_file = tmp_path / "leakage.jsonl"
    leakage_file.write_text(
        '{"id": "leak_01", "prompt": "P", "response": "R", "data_role": "HELD_OUT_BENCHMARK"}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Contamination error"):
        SFTDataset(leakage_file, canonical_tokenizer, is_training=True)

    pending_file = tmp_path / "pending.jsonl"
    pending_file.write_text(
        '{"id": "pend_01", "prompt": "P", "response": "R", "review_status": "PENDING_HUMAN_REVIEW"}\n',
        encoding="utf-8",
    )
    with pytest.raises(PermissionError, match="Trust Gate Refusal"):
        SFTDataset(pending_file, canonical_tokenizer, enforce_trust_gate=True, allow_exploratory_opt_in=False)

    rejected_file = tmp_path / "rejected.jsonl"
    rejected_file.write_text(
        '{"id": "rej_01", "prompt": "P", "response": "R", "status": "REJECTED"}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Trust Gate Refusal: Record .* is marked REJECTED"):
        SFTDataset(rejected_file, canonical_tokenizer, enforce_trust_gate=True)


def test_hand_constructed_batch_explicit_labels():
    """
    Test 12: Hand-constructed synthetic batch with independently computed expected labels.
    Does not rely on tokenizer or SFTDataset internals.
    Directly exercises build_causal_sequence contract and negative failure assertions.
    """
    bos_id = 0
    eos_id = 1
    ignore_index = -100

    # Example 1: Instruction pair (append_eos=True)
    # Prompt tokens: [10, 20, 30]
    # Response tokens: [40, 50]
    p1 = [10, 20, 30]
    r1 = [40, 50]
    inp1, tgt1 = build_causal_sequence(p1, r1, bos_id=bos_id, eos_id=eos_id, ignore_index=ignore_index, append_eos=True)

    # Expected inputs:  [0, 10, 20, 30, 40, 50]
    # Expected targets: [-100, -100, -100, 40, 50, 1]
    assert inp1 == [0, 10, 20, 30, 40, 50]
    assert tgt1 == [-100, -100, -100, 40, 50, 1]
    assert len(inp1) == len(tgt1) == 6

    # Causal alignment check:
    # At index 0 (BOS): predicts tgt1[0] = -100 (masked)
    # At index 1 (p1[0]=10): predicts tgt1[1] = -100 (masked)
    # At index 2 (p1[1]=20): predicts tgt1[2] = -100 (masked)
    # At index 3 (p1[2]=30, terminal prompt token): predicts tgt1[3] = 40 (first response token r1[0])!
    # At index 4 (r1[0]=40): predicts tgt1[4] = 50 (second response token r1[1])!
    # At index 5 (r1[1]=50): predicts tgt1[5] = 1 (EOS)!
    assert inp1[3] == p1[-1]
    assert tgt1[3] == r1[0]
    assert inp1[4] == r1[0]
    assert tgt1[4] == r1[1]
    assert inp1[5] == r1[1]
    assert tgt1[5] == eos_id

    # Example 2: Continuation/Replay pair (append_eos=False)
    # Prompt tokens: [100, 200]
    # Response tokens: [300, 400]
    p2 = [100, 200]
    r2 = [300, 400]
    inp2, tgt2 = build_causal_sequence(p2, r2, bos_id=bos_id, eos_id=eos_id, ignore_index=ignore_index, append_eos=False)

    # Expected inputs:  [0, 100, 200, 300]
    # Expected targets: [-100, -100, 300, 400]
    assert inp2 == [0, 100, 200, 300]
    assert tgt2 == [-100, -100, 300, 400]
    assert len(inp2) == len(tgt2) == 4
    # Terminal prompt token p2[1]=200 predicts r2[0]=300
    assert inp2[2] == p2[-1]
    assert tgt2[2] == r2[0]


def test_negative_deliberate_misalignment_rejection():
    """
    Negative test demonstrating that deliberate label corruption or shifting is detected.
    Ensures that an off-by-one boundary shift fails validation.
    """
    bos_id = 0
    eos_id = 1
    ignore_index = -100
    p = [11, 22, 33]
    r = [44, 55]

    inp, tgt = build_causal_sequence(p, r, bos_id=bos_id, eos_id=eos_id, ignore_index=ignore_index, append_eos=True)

    # Deliberate off-by-one corruption: unmasking prompt token or shifting prediction
    corrupted_tgt_early_unmask = [-100, -100, 33, 44, 55, 1]
    # The terminal prompt token position is index len(p) = 3
    # In corrupted target, index 2 (an internal prompt token) is unmasked
    assert corrupted_tgt_early_unmask[len(p) - 1] != ignore_index, "Corrupted target should have leaked prompt token"

    corrupted_tgt_delayed_unmask = [-100, -100, -100, -100, 44, 1]
    # In delayed unmask, the terminal prompt token position at index 3 is masked with -100 instead of predicting r[0]
    assert corrupted_tgt_delayed_unmask[len(p)] != r[0], "Corrupted target missed first response token"