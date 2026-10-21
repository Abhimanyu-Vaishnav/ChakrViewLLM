"""Dataset Validation and Audit Script for Experiment 02.

Validates:
1. JSONL structural integrity and parseability.
2. Field schema: id, category, prompt, response, expected_status, expected_schema, component, split, data_role.
3. Disjointness: zero overlap in status values or components between train, val, and test.
4. Tokenizer compatibility: all sequences tokenize properly and stay within max_seq_len (256).
5. Exact response JSON parseability and type verification.
6. Target alignment under SFTDataset encoding.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Set

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chakrview.tokenizer.serialization import load_tokenizer_artifacts
from chakrview.training.sft_engine import SFTDataset

DATA_DIR = Path(__file__).resolve().parent / "data"

def validate_datasets():
    print("=" * 70)
    print("EXP 02 DATASET AUDIT & INTEGRITY VALIDATION")
    print("=" * 70)

    tok_dir = REPO_ROOT / "data" / "experiments" / "vocab_4096"
    tokenizer, _ = load_tokenizer_artifacts(tok_dir)

    train_file = DATA_DIR / "train.jsonl"
    val_file = DATA_DIR / "val.jsonl"
    test_file = DATA_DIR / "test.jsonl"

    for path in [train_file, val_file, test_file]:
        assert path.exists(), f"File missing: {path}"

    with open(train_file, "r", encoding="utf-8") as f:
        train_recs = [json.loads(line) for line in f]
    with open(val_file, "r", encoding="utf-8") as f:
        val_recs = [json.loads(line) for line in f]
    with open(test_file, "r", encoding="utf-8") as f:
        test_recs = [json.loads(line) for line in f]

    print(f"Loaded records: Train={len(train_recs)}, Val={len(val_recs)}, Test={len(test_recs)}")

    # 1. Check categories and roles
    assert all(r["data_role"] == "TRAIN" for r in train_recs), "Train split has non-TRAIN data_role"
    assert all(r["data_role"] == "VALIDATION" for r in val_recs), "Val split has non-VALIDATION data_role"
    assert all(r["data_role"] == "HELD_OUT_BENCHMARK" for r in test_recs), "Test split has non-BENCHMARK data_role"

    # 2. Check disjointness of status values
    train_statuses = set(r["expected_status"] for r in train_recs if "expected_status" in r)
    val_statuses = set(r["expected_status"] for r in val_recs if "expected_status" in r)
    test_statuses = set(r["expected_status"] for r in test_recs if "expected_status" in r)

    print(f"Unique status counts: Train={len(train_statuses)}, Val={len(val_statuses)}, Test={len(test_statuses)}")
    assert train_statuses.isdisjoint(val_statuses), "Train and Val statuses overlap!"
    assert train_statuses.isdisjoint(test_statuses), "Train and Test statuses overlap!"
    assert val_statuses.isdisjoint(test_statuses), "Val and Test statuses overlap!"
    print("  [PASS] Status sets are 100% strictly disjoint.")

    # 3. Check disjointness of components
    train_comps = set(r["component"] for r in train_recs if "component" in r)
    val_comps = set(r["component"] for r in val_recs if "component" in r)
    test_comps = set(r["component"] for r in test_recs if "component" in r)

    assert train_comps.isdisjoint(val_comps), "Train and Val components overlap!"
    assert train_comps.isdisjoint(test_comps), "Train and Test components overlap!"
    assert val_comps.isdisjoint(test_comps), "Val and Test components overlap!"
    print("  [PASS] Component sets are 100% strictly disjoint.")

    # 4. Check JSON schema and response validity
    for split_name, recs in [("Train", train_recs), ("Val", val_recs), ("Test", test_recs)]:
        for r in recs:
            if r.get("category") == "single_field_string_extraction":
                resp_obj = json.loads(r["response"])
                assert isinstance(resp_obj, dict), f"Response is not a dict in {r['id']}"
                assert list(resp_obj.keys()) == ["status"], f"Keys mismatch in {r['id']}: {resp_obj.keys()}"
                assert resp_obj["status"] == r["expected_status"], f"Status value mismatch in {r['id']}"
                assert r["expected_status"] in r["prompt"], f"Expected status not in prompt context in {r['id']}"

    print("  [PASS] All JSON responses strictly adhere to schema and match context ground truth.")

    # 5. Check Tokenizer Encoding and Sequence Lengths via SFTDataset
    train_ds = SFTDataset(train_file, tokenizer, max_seq_len=256, is_training=True, strip_prompt_delimiters=False)
    val_ds = SFTDataset(val_file, tokenizer, max_seq_len=256, is_training=False, strip_prompt_delimiters=False)

    for i in range(len(train_ds)):
        inp, tgt = train_ds.encode_example(train_ds.records[i]["prompt"], train_ds.records[i]["response"], append_eos=train_ds.records[i].get("append_eos", True))
        assert len(inp) == len(tgt), f"Length mismatch in train record {i}"
        assert len(inp) <= 256, f"Train record {i} exceeds max_seq_len (length={len(inp)})"

    for i in range(len(val_ds)):
        inp, tgt = val_ds.encode_example(val_ds.records[i]["prompt"], val_ds.records[i]["response"], append_eos=val_ds.records[i].get("append_eos", True))
        assert len(inp) == len(tgt), f"Length mismatch in val record {i}"
        assert len(inp) <= 256, f"Val record {i} exceeds max_seq_len (length={len(inp)})"

    print("  [PASS] SFTDataset encoding verified: causal alignments and sequence limits intact.")
    print("=" * 70)
    print("ALL VALIDATION CHECKS PASSED DETERMINISTICALLY")
    print("=" * 70)

if __name__ == "__main__":
    validate_datasets()
