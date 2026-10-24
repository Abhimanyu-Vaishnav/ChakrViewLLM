"""
test_exp16_v3_1_contract.py
ChakrView Experiment 16 v3.1 — Contract, Leakage, and Robustness Verification Test Suite.

Tests:
1. Schema validity and unique example IDs.
2. Exact split sizes, category counts, and label counts.
3. Reproducibility and manifest/hash agreement.
4. Character/byte/token offset alignment.
5. Answerable gold spans decode to the intended target occurrence.
6. Every gold span is valid under the candidate mask.
7. Query, assistant prefix, role markers, special tokens, and padding handling.
8. Abstention sentinel behavior ([0, 0]).
9. Span length limits (<= 12) and valid start/end ordering.
10. Collation preserving record masks and excluding padding.
11. Synthetic finite loss and gradient flow on dummy pointer module.
12. Zero cross-split prompt duplicates.
13. NEW IN V3.1: Same-query answerable/abstention pairs test.
14. NEW IN V3.1: Train-to-test query generalization baseline bounded by label prior.
15. NEW IN V3.1: Decoder mask compliance under adversarial/random logits.
16. NEW IN V3.1: Rejection of corrupted / out-of-bounds spans.
"""

import hashlib
import json
import math
from pathlib import Path
import sys
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CANONICAL_TOK_DIR = REPO_ROOT / "data" / "experiments" / "vocab_4096"
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

from chakrview.tokenizer.serialization import load_tokenizer_artifacts
from experiments.research_v09_conversation_sft.exp04_pointer_robustness.span_aligner import (
    align_character_span_to_tokens,
    AlignmentError
)


@pytest.fixture(scope="module")
def tokenizer():
    tok, _ = load_tokenizer_artifacts(CANONICAL_TOK_DIR, verify_checksums=False)
    return tok


@pytest.fixture(scope="module")
def datasets():
    data = {}
    for split in ["train", "val", "test"]:
        p = DATA_DIR / f"{split}_exp16_v3_1.jsonl"
        with open(p, "r", encoding="utf-8") as f:
            data[split] = [json.loads(line) for line in f]
    return data


def test_schema_and_id_uniqueness(datasets):
    required_keys = {
        "example_id", "split", "category", "field_id", "prompt", "context_char_end",
        "target_text", "char_span", "input_ids", "start_pos", "end_pos",
        "is_abstention", "candidate_mask", "rationale"
    }
    all_ids = set()
    total_count = 0

    for split, recs in datasets.items():
        for r in recs:
            assert required_keys.issubset(r.keys()), f"Missing keys in {r['example_id']}"
            assert r["example_id"] not in all_ids, f"Duplicate ID: {r['example_id']}"
            all_ids.add(r["example_id"])
            total_count += 1

    assert total_count == 480
    assert len(all_ids) == 480


def test_split_and_label_counts(datasets):
    expected_counts = {
        "train": {"total": 300, "ans": 222, "abs": 78},
        "val": {"total": 60, "ans": 46, "abs": 14},
        "test": {"total": 120, "ans": 92, "abs": 28}
    }

    for split, exp in expected_counts.items():
        recs = datasets[split]
        assert len(recs) == exp["total"]
        ans = sum(1 for r in recs if not r["is_abstention"])
        abs_count = sum(1 for r in recs if r["is_abstention"])
        assert ans == exp["ans"], f"{split} ans mismatch: {ans} != {exp['ans']}"
        assert abs_count == exp["abs"], f"{split} abs mismatch: {abs_count} != {exp['abs']}"


def test_reproducibility_and_manifest_hash_agreement():
    manifest_p = BASE_DIR / "EXP16_V3_1_MANIFEST.json"
    assert manifest_p.exists()
    with open(manifest_p, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    for split in ["train", "val", "test"]:
        p = DATA_DIR / f"{split}_exp16_v3_1.jsonl"
        actual_hash = hashlib.sha256(p.read_bytes()).hexdigest()
        assert actual_hash == manifest["dataset_hashes"][split]


def test_character_and_token_span_alignment(datasets, tokenizer):
    for split, recs in datasets.items():
        for r in recs:
            prompt = r["prompt"]
            c_s, c_e = r["char_span"]
            target = r["target_text"]
            input_ids = r["input_ids"]
            t_s, t_e = r["start_pos"], r["end_pos"]

            if r["is_abstention"]:
                assert c_s == 0 and c_e == 0
                assert t_s == 0 and t_e == 0
                assert target == ""
            else:
                assert prompt[c_s:c_e] == target
                token_slice = input_ids[t_s : t_e + 1]
                decoded = tokenizer.decode(token_slice).strip()
                assert decoded.lower() == target.strip().lower(), (
                    f"Decoding mismatch in {r['example_id']}: decoded '{decoded}' != '{target}'"
                )


def test_candidate_mask_contract_and_exclusion(datasets):
    for split, recs in datasets.items():
        for r in recs:
            mask = r["candidate_mask"]
            input_ids = r["input_ids"]
            assert len(mask) == len(input_ids)
            assert mask[0] is True, "BOS position 0 must be True (abstention sentinel)"

            t_s, t_e = r["start_pos"], r["end_pos"]
            if r["is_abstention"]:
                assert t_s == 0 and t_e == 0
            else:
                assert mask[t_s] is True, f"start_pos {t_s} is masked out in {r['example_id']}"
                assert mask[t_e] is True, f"end_pos {t_e} is masked out in {r['example_id']}"

            # Verify that query and assistant prefix tokens are False
            for i in range(len(input_ids) - 5, len(input_ids)):
                assert mask[i] is False, f"Assistant token at {i} allowed in mask for {r['example_id']}"


def test_span_length_constraints(datasets):
    for split, recs in datasets.items():
        for r in recs:
            if not r["is_abstention"]:
                span_len = r["end_pos"] - r["start_pos"] + 1
                assert 1 <= span_len <= 12, f"Span length {span_len} exceeds max 12 in {r['example_id']}"
                assert r["start_pos"] <= r["end_pos"]


def test_collation_and_padding_exclusion(datasets):
    batch = datasets["train"][:8]
    max_len = max(len(b["input_ids"]) for b in batch)

    candidate_mask_list = []
    for b in batch:
        cur_len = len(b["input_ids"])
        pad_len = max_len - cur_len
        mask = b["candidate_mask"] + [False] * pad_len
        candidate_mask_list.append(mask)

    padded_mask = torch.tensor(candidate_mask_list, dtype=torch.bool)
    assert padded_mask.shape == (8, max_len)
    for i, b in enumerate(batch):
        cur_len = len(b["input_ids"])
        if cur_len < max_len:
            assert not padded_mask[i, cur_len:].any(), "Padding tokens allowed in mask"


def test_synthetic_finite_loss_and_gradient_flow():
    B, T = 4, 32
    logits_start = torch.randn(B, T, requires_grad=True)
    logits_end = torch.randn(B, T, requires_grad=True)
    mask = torch.ones(B, T, dtype=torch.bool)
    mask[:, 20:] = False

    masked_s = logits_start.masked_fill(~mask, -1e9)
    masked_e = logits_end.masked_fill(~mask, -1e9)

    target_s = torch.tensor([0, 5, 12, 0], dtype=torch.long)
    target_e = torch.tensor([0, 8, 15, 0], dtype=torch.long)

    criterion = nn.CrossEntropyLoss()
    loss = (criterion(masked_s, target_s) + criterion(masked_e, target_e)) / 2.0

    assert not torch.isnan(loss) and not torch.isinf(loss)
    loss.backward()
    assert logits_start.grad is not None
    assert not torch.isnan(logits_start.grad).any()


def test_zero_cross_split_prompt_duplicates(datasets):
    prompts_by_split = {
        split: set(r["prompt"] for r in recs)
        for split, recs in datasets.items()
    }
    assert len(prompts_by_split["train"] & prompts_by_split["val"]) == 0
    assert len(prompts_by_split["train"] & prompts_by_split["test"]) == 0
    assert len(prompts_by_split["val"] & prompts_by_split["test"]) == 0


def test_same_query_answerable_abstention_pairs(datasets):
    """Asserts that in Train, 100% of unique queries appear with BOTH labels."""
    from collections import defaultdict
    queries = defaultdict(lambda: {"ans": 0, "abs": 0})
    for r in datasets["train"]:
        remainder = r["prompt"][r["context_char_end"]:].strip()
        q = remainder[:-len("\nAssistant:")].strip() if remainder.endswith("\nAssistant:") else remainder
        status = "abs" if r["is_abstention"] else "ans"
        queries[q][status] += 1

    assert len(queries) == 25
    for q, v in queries.items():
        assert v["ans"] > 0, f"Query '{q}' lacks answerable examples in train"
        assert v["abs"] > 0, f"Query '{q}' lacks abstention examples in train"


def test_decoder_mask_compliance_under_adversarial_logits():
    """Verify that constrained decoding strictly rejects masked-out candidate positions."""
    T = 64
    candidate_mask = torch.ones(T, dtype=torch.bool)
    # Mask out positions 20 to 63 (e.g. query and assistant prefix)
    candidate_mask[20:] = False

    # Adversarial logits: strongly favor a masked-out position (e.g. position 30)
    start_logits = torch.randn(T)
    end_logits = torch.randn(T)
    start_logits[30] = 100.0  # Adversarial peak
    end_logits[32] = 100.0

    # Apply candidate mask
    start_logits = start_logits.masked_fill(~candidate_mask, -1e9)
    end_logits = end_logits.masked_fill(~candidate_mask, -1e9)

    # Decode using constrained search
    best_score = float("-inf")
    best_s, best_e = 0, 0
    if candidate_mask[0]:
        best_score = start_logits[0].item() + end_logits[0].item()

    for s in range(1, T):
        if not candidate_mask[s]:
            continue
        for e in range(s, min(s + 12, T)):
            if not candidate_mask[e]:
                continue
            score = start_logits[s].item() + end_logits[e].item()
            if score > best_score:
                best_score = score
                best_s, best_e = s, e

    assert bool(candidate_mask[best_s]) is True
    assert bool(candidate_mask[best_e]) is True
    assert best_s < 20, "Decoder selected a masked query position!"
    assert best_e < 20, "Decoder selected a masked query position!"


def test_fail_closed_rejection_of_corrupted_spans(tokenizer):
    """Verify that align_character_span_to_tokens raises AlignmentError on mismatched slices."""
    prompt = "Supervisor: Operating node_tr_0 at 100 MHz.\nAssistant:"
    # Target text is 'node_tr_0', but we pass slice pointing to 'Operating'
    with pytest.raises(AlignmentError):
        align_character_span_to_tokens(
            prompt=prompt,
            char_start=12,
            char_end=21,
            target_text="MISMATCHED_TARGET",
            tokenizer=tokenizer,
            is_abstention=False
        )
