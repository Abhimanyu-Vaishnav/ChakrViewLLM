"""Dataset Preparation for ChakrView v0.8 Experiment 07 Copy Overfit Test.

Selects:
- Fixed tiny training set of exactly 16 Stage A single-token examples from Exp06 train.jsonl.
- Untouched unseen evaluation set of 25 Stage A single-token examples from Exp06 test.jsonl.
- Verifies every value encodes to exactly 1 token under the canonical tokenizer.
- Fixed prompt template: `Target value: {value}\nCopy: `
- Saves splits and cryptographic manifest.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chakrview.tokenizer.serialization import load_tokenizer_artifacts

OUTPUT_DIR = Path(__file__).resolve().parent / "data"


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def prepare_dataset():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tok_dir = REPO_ROOT / "data" / "experiments" / "vocab_4096"
    tokenizer, _ = load_tokenizer_artifacts(tok_dir)

    exp06_train_file = REPO_ROOT / "experiments" / "research_v08_json_sft" / "exp06_minimal_copy_diagnostic" / "data" / "train.jsonl"
    exp06_test_file = REPO_ROOT / "experiments" / "research_v08_json_sft" / "exp06_minimal_copy_diagnostic" / "data" / "test.jsonl"

    with open(exp06_train_file, "r", encoding="utf-8") as f:
        exp06_train_recs = [json.loads(line) for line in f if json.loads(line)["stage"] == "Stage_A"]

    with open(exp06_test_file, "r", encoding="utf-8") as f:
        exp06_test_recs = [json.loads(line) for line in f if json.loads(line)["stage"] == "Stage_A"]

    # Select 16 Stage A single-token examples for tiny training overfit set
    tiny_train_recs = exp06_train_recs[:16]
    # Keep 25 Stage A unseen examples for untouched diagnostic evaluation
    unseen_test_recs = exp06_test_recs[:25]

    assert len(tiny_train_recs) == 16
    assert len(unseen_test_recs) == 25

    # Verify each response encodes to exactly 1 token
    for r in tiny_train_recs:
        val = r["expected_value"]
        toks = tokenizer.encode(val)
        assert len(toks) == 1, f"Train value {val} encodes to {len(toks)} tokens"
        p_toks = tokenizer.encode(r["prompt"])
        full_toks = tokenizer.encode(r["prompt"] + val)
        assert full_toks == p_toks + toks, f"Alignment mismatch in {val}"

    for r in unseen_test_recs:
        val = r["expected_value"]
        toks = tokenizer.encode(val)
        assert len(toks) == 1, f"Test value {val} encodes to {len(toks)} tokens"
        p_toks = tokenizer.encode(r["prompt"])
        full_toks = tokenizer.encode(r["prompt"] + val)
        assert full_toks == p_toks + toks, f"Alignment mismatch in {val}"

    # Verify zero string or token sequence overlap
    train_vals = set(r["expected_value"] for r in tiny_train_recs)
    test_vals = set(r["expected_value"] for r in unseen_test_recs)
    assert train_vals.isdisjoint(test_vals), "Train and unseen test values overlap!"

    # Write files
    train_out_path = OUTPUT_DIR / "tiny_train_16.jsonl"
    test_out_path = OUTPUT_DIR / "unseen_test_25.jsonl"

    with open(train_out_path, "w", encoding="utf-8") as f:
        for r in tiny_train_recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(test_out_path, "w", encoding="utf-8") as f:
        for r in unseen_test_recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    train_sha = compute_sha256(train_out_path)
    test_sha = compute_sha256(test_out_path)

    manifest = {
        "experiment": "exp07_copy_overfit",
        "description": "Tiny 16-example Stage A single-token overfit dataset",
        "train_file": str(train_out_path),
        "train_records": len(tiny_train_recs),
        "train_sha256": train_sha,
        "train_values": sorted(list(train_vals)),
        "unseen_test_file": str(test_out_path),
        "unseen_test_records": len(unseen_test_recs),
        "unseen_test_sha256": test_sha,
    }

    manifest_path = OUTPUT_DIR / "dataset_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("Dataset preparation complete:")
    print(f"  Tiny Train Set: {train_out_path.name} ({len(tiny_train_recs)} records, SHA: {train_sha})")
    print(f"  Train Values: {sorted(list(train_vals))}")
    print(f"  Unseen Test Set: {test_out_path.name} ({len(unseen_test_recs)} records, SHA: {test_sha})")


if __name__ == "__main__":
    prepare_dataset()
