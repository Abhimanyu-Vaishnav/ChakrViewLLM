"""
Preflight audit script for Single-Field Entity Copy Training Pilot.
Verifies all 10 mandatory preflight checks before any training process can be launched.
"""

import json
import hashlib
from pathlib import Path

CANONICAL_CHECKPOINT_PATH = Path("experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt")
EXPECTED_CHECKPOINT_HASH = "96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee"

VOCAB_PATH = Path("data/experiments/vocab_4096/vocab.json")
EXPECTED_VOCAB_HASH = "db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a"

MERGES_PATH = Path("data/experiments/vocab_4096/merges.json")
EXPECTED_MERGES_HASH = "ada4b1dcfbb3d962a2e60203089881872a5063de73b2fef9eeff81ce0016efa7"

TRAIN_FILE = Path("experiments/single_field_copy_pilot/authorized_run_001/data/train.jsonl")
VAL_FILE = Path("experiments/single_field_copy_pilot/authorized_run_001/data/val.jsonl")


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def audit():
    errors = []
    
    # 1. Base Checkpoint & Tokenizer Hash Checks
    ckpt_hash = compute_sha256(CANONICAL_CHECKPOINT_PATH)
    if ckpt_hash != EXPECTED_CHECKPOINT_HASH:
        errors.append(f"Checkpoint hash mismatch: got {ckpt_hash}, expected {EXPECTED_CHECKPOINT_HASH}")
        
    vocab_hash = compute_sha256(VOCAB_PATH)
    if vocab_hash != EXPECTED_VOCAB_HASH:
        errors.append(f"Vocab hash mismatch: got {vocab_hash}, expected {EXPECTED_VOCAB_HASH}")
        
    merges_hash = compute_sha256(MERGES_PATH)
    if merges_hash != EXPECTED_MERGES_HASH:
        errors.append(f"Merges hash mismatch: got {merges_hash}, expected {EXPECTED_MERGES_HASH}")

    # 2. Dataset checks
    if not TRAIN_FILE.exists():
        errors.append(f"Train file does not exist: {TRAIN_FILE}")
        return errors
    if not VAL_FILE.exists():
        errors.append(f"Val file does not exist: {VAL_FILE}")
        return errors
        
    train_records = [json.loads(line) for line in open(TRAIN_FILE, encoding="utf-8")]
    val_records = [json.loads(line) for line in open(VAL_FILE, encoding="utf-8")]
    
    if len(train_records) != 200:
        errors.append(f"Expected 200 train records, got {len(train_records)}")
    if len(val_records) != 40:
        errors.append(f"Expected 40 val records, got {len(val_records)}")
        
    # 3. Target verbatim in context check
    for idx, r in enumerate(train_records):
        if r["target_value"] not in r["context"]:
            errors.append(f"Train record {idx} target '{r['target_value']}' not verbatim in context '{r['context']}'")
            
    for idx, r in enumerate(val_records):
        if r["target_value"] not in r["context"]:
            errors.append(f"Val record {idx} target '{r['target_value']}' not verbatim in context '{r['context']}'")

    # 4. Strict disjointness between train and val
    train_contexts = set(r["context"] for r in train_records)
    val_contexts = set(r["context"] for r in val_records)
    context_overlap = train_contexts.intersection(val_contexts)
    if context_overlap:
        errors.append(f"Train and val share overlapping contexts: {context_overlap}")
        
    train_targets = set(r["target_value"] for r in train_records)
    val_targets = set(r["target_value"] for r in val_records)
    target_overlap = train_targets.intersection(val_targets)
    if target_overlap:
        errors.append(f"Train and val share overlapping target values: {target_overlap}")
        
    train_families = set(r["family"] for r in train_records)
    val_families = set(r["family"] for r in val_records)
    family_overlap = train_families.intersection(val_families)
    if family_overlap:
        errors.append(f"Train and val share overlapping families: {family_overlap}")

    # 5. Dialog format check
    for r in train_records + val_records:
        if not r["prompt"].startswith("User: In the following context: "):
            errors.append(f"Record {r['id']} prompt format invalid: {r['prompt'][:40]}")
        if not r["prompt"].endswith("Assistant:"):
            errors.append(f"Record {r['id']} prompt must end with 'Assistant:'")
        if not r["target"].startswith(" "):
            errors.append(f"Record {r['id']} target should have leading space for BPE")

    train_hash = compute_sha256(TRAIN_FILE)
    val_hash = compute_sha256(VAL_FILE)

    if errors:
        print("PREFLIGHT AUDIT FAILED:")
        for e in errors:
            print(" -", e)
        return False
    else:
        print("PREFLIGHT AUDIT PASSED:")
        print(f" - Canonical Checkpoint SHA-256: {ckpt_hash} (OK)")
        print(f" - Vocab SHA-256: {vocab_hash} (OK)")
        print(f" - Merges SHA-256: {merges_hash} (OK)")
        print(f" - Train Records: {len(train_records)}, SHA-256: {train_hash} (OK)")
        print(f" - Val Records: {len(val_records)}, SHA-256: {val_hash} (OK)")
        print(f" - Verbatim target containment: 100% in train and val (OK)")
        print(f" - Context overlap: 0 (OK)")
        print(f" - Target overlap: 0 (OK)")
        print(f" - Family overlap: 0 (OK)")
        return True

if __name__ == "__main__":
    import sys
    success = audit()
    if not success:
        sys.exit(1)
