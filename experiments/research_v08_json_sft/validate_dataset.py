"""Automated Dataset Validation and Leakage Detection Suite for ChakrView v0.8.

Performs:
1. Exact record counts & schema conformance.
2. Cross-split duplicate detection (exact prompt string & normalized prompt string).
3. Entity disjointness assertion (Component names in test must NOT appear in train or val).
4. Status string disjointness assertion (Status values in test must NOT appear in train or val).
5. Integer code disjointness assertion (Code values in test must NOT appear in train or val).
6. Replay text disjointness assertion.
7. Sequence length and tokenizer encoding compatibility check.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Set

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chakrview.tokenizer.serialization import load_tokenizer_artifacts

DATA_DIR = REPO_ROOT / "experiments" / "research_v08_json_sft" / "data"
CANONICAL_TOKENIZER_DIR = REPO_ROOT / "data" / "experiments" / "vocab_4096"


def load_split(name: str) -> List[Dict[str, Any]]:
    path = DATA_DIR / f"{name}.jsonl"
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def normalize_str(s: str) -> str:
    return " ".join(s.lower().strip().split())


def validate():
    print("=" * 60)
    print("VALIDATING CHAKRVIEW v0.8 SINGLE-TASK JSON DATASET")
    print("=" * 60)

    train = load_split("train")
    val = load_split("val")
    test = load_split("test")

    # 1. Counts
    assert len(train) == 140, f"Train count mismatch: {len(train)} != 140"
    assert len(val) == 30, f"Val count mismatch: {len(val)} != 30"
    assert len(test) == 30, f"Test count mismatch: {len(test)} != 30"

    train_json = [r for r in train if r["category"] == "two_key_json_extraction"]
    train_rep = [r for r in train if r["category"] == "base_continuation_replay"]
    assert len(train_json) == 100
    assert len(train_rep) == 40

    val_json = [r for r in val if r["category"] == "two_key_json_extraction"]
    val_rep = [r for r in val if r["category"] == "base_continuation_replay"]
    assert len(val_json) == 20
    assert len(val_rep) == 10

    test_json = [r for r in test if r["category"] == "two_key_json_extraction"]
    test_rep = [r for r in test if r["category"] == "base_continuation_replay"]
    assert len(test_json) == 20
    assert len(test_rep) == 10

    print("[PASS] Record and Category counts verified:")
    print(f"       Train: {len(train_json)} JSON + {len(train_rep)} Replay = 140")
    print(f"       Val:   {len(val_json)} JSON + {len(val_rep)} Replay = 30")
    print(f"       Test:  {len(test_json)} JSON + {len(test_rep)} Replay = 30")

    # 2. Duplicate Detection
    train_prompts = {r["prompt"] for r in train}
    val_prompts = {r["prompt"] for r in val}
    test_prompts = {r["prompt"] for r in test}

    assert len(train_prompts) == 140, "Duplicate prompts within train split!"
    assert len(val_prompts) == 30, "Duplicate prompts within val split!"
    assert len(test_prompts) == 30, "Duplicate prompts within test split!"

    train_val_overlap = train_prompts.intersection(val_prompts)
    train_test_overlap = train_prompts.intersection(test_prompts)
    val_test_overlap = val_prompts.intersection(test_prompts)

    assert not train_val_overlap, f"Prompt overlap between train and val: {train_val_overlap}"
    assert not train_test_overlap, f"Prompt overlap between train and test: {train_test_overlap}"
    assert not val_test_overlap, f"Prompt overlap between val and test: {val_test_overlap}"

    # Normalized prompt overlap check
    train_norm = {normalize_str(r["prompt"]) for r in train}
    test_norm = {normalize_str(r["prompt"]) for r in test}
    norm_overlap = train_norm.intersection(test_norm)
    assert not norm_overlap, f"Normalized prompt overlap between train and test: {norm_overlap}"

    print("[PASS] Exact & Normalized Duplicate Checks: Zero overlap across splits.")

    # 3. Entity (Component) Leakage Check
    train_comps = {r["component"] for r in train_json}
    val_comps = {r["component"] for r in val_json}
    test_comps = {r["component"] for r in test_json}

    assert not test_comps.intersection(train_comps), "Test components leaked into train!"
    assert not test_comps.intersection(val_comps), "Test components leaked into val!"
    assert not val_comps.intersection(train_comps), "Val components leaked into train!"

    print("[PASS] Component Entity Leakage: 100% disjoint.")
    print(f"       Train components: {len(train_comps)}, Val: {len(val_comps)}, Test: {len(test_comps)}")

    # 4. Status String Leakage Check
    train_statuses = {r["expected_status"] for r in train_json}
    val_statuses = {r["expected_status"] for r in val_json}
    test_statuses = {r["expected_status"] for r in test_json}

    assert not test_statuses.intersection(train_statuses), f"Test status leaked into train: {test_statuses.intersection(train_statuses)}"
    assert not test_statuses.intersection(val_statuses), f"Test status leaked into val: {test_statuses.intersection(val_statuses)}"

    print("[PASS] Status Value Leakage: 100% disjoint.")
    print(f"       Test statuses: {test_statuses}")
    print(f"       Val statuses:  {val_statuses}")

    # 5. Numeric Code Range Leakage Check
    train_codes = {r["expected_code"] for r in train_json}
    val_codes = {r["expected_code"] for r in val_json}
    test_codes = {r["expected_code"] for r in test_json}

    assert all(100 <= c <= 699 for c in train_codes), "Train codes outside [100, 699]"
    assert all(700 <= c <= 799 for c in val_codes), "Val codes outside [700, 799]"
    assert all(800 <= c <= 899 for c in test_codes), "Test codes outside [800, 899]"

    assert not test_codes.intersection(train_codes), "Test codes leaked into train!"
    assert not test_codes.intersection(val_codes), "Test codes leaked into val!"

    print("[PASS] Integer Code Range Leakage: Strictly partitioned.")
    print(f"       Train range: [{min(train_codes)}, {max(train_codes)}] (N={len(train_codes)})")
    print(f"       Val range:   [{min(val_codes)}, {max(val_codes)}] (N={len(val_codes)})")
    print(f"       Test range:  [{min(test_codes)}, {max(test_codes)}] (N={len(test_codes)})")

    # 6. Replay Disjointness Check
    train_rep_prompts = {r["prompt"] for r in train_rep}
    val_rep_prompts = {r["prompt"] for r in val_rep}
    test_rep_prompts = {r["prompt"] for r in test_rep}

    assert not test_rep_prompts.intersection(train_rep_prompts), "Test replay leaked into train!"
    assert not test_rep_prompts.intersection(val_rep_prompts), "Test replay leaked into val!"

    print("[PASS] Replay Corpus Disjointness: 100% disjoint.")

    # 7. Tokenizer & Sequence Length Compatibility
    tok, _ = load_tokenizer_artifacts(CANONICAL_TOKENIZER_DIR)
    max_observed_len = 0
    all_records = train + val + test
    for r in all_records:
        p_toks = tok.encode(r["prompt"])
        r_toks = tok.encode(r["response"])
        total_len = 1 + len(p_toks) + len(r_toks)
        if total_len > max_observed_len:
            max_observed_len = total_len
        assert total_len <= 256, f"Record {r['id']} exceeds max_seq_len (256): {total_len}"

    print(f"[PASS] Tokenizer & Sequence Length Check: Max observed sequence length is {max_observed_len} tokens (well under 256).")
    print("=" * 60)
    print("ALL DATASET INTEGRITY AND LEAKAGE GATES PASSED PERFECTLY")
    print("=" * 60)


if __name__ == "__main__":
    validate()
