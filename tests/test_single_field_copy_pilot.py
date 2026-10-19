"""
Unit tests for the Single-Field Entity Copy Training Pilot.
Verifies dataset invariants, preflight checks, early-stop logic, and non-tampering of canonical assets.
"""

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = REPO_ROOT / "experiments" / "single_field_copy_pilot" / "authorized_run_001"
DATA_DIR = EXP_DIR / "data"

CANONICAL_CHECKPOINT = REPO_ROOT / "experiments" / "batch16_21_combined" / "main_training_10000" / "training_checkpoint_step_10000.pt"
TOKENIZER_DIR = REPO_ROOT / "data" / "experiments" / "vocab_4096"

EXPECTED_BASE_HASH = "96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee"
EXPECTED_VOCAB_HASH = "db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a"
EXPECTED_MERGES_HASH = "ada4b1dcfbb3d962a2e60203089881872a5063de73b2fef9eeff81ce0016efa7"


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def test_canonical_integrity_preserved():
    """Verify that canonical model and tokenizer assets were not overwritten or altered."""
    assert compute_sha256(CANONICAL_CHECKPOINT) == EXPECTED_BASE_HASH
    assert compute_sha256(TOKENIZER_DIR / "vocab.json") == EXPECTED_VOCAB_HASH
    assert compute_sha256(TOKENIZER_DIR / "merges.json") == EXPECTED_MERGES_HASH


def test_protected_session_file_unmodified():
    """Verify that chakrview/runtime/session.py has zero git diff."""
    diff = subprocess.check_output(["git", "diff", "--name-only", "chakrview/runtime/session.py"]).decode("utf-8").strip()
    assert diff == "", "Protected session.py file was modified!"


def test_dataset_invariants():
    """Verify dataset sizes, verbatim substring containment, and strict disjointness."""
    train_file = DATA_DIR / "train.jsonl"
    val_file = DATA_DIR / "val.jsonl"
    
    assert train_file.exists()
    assert val_file.exists()
    
    train_recs = [json.loads(line) for line in open(train_file, encoding="utf-8")]
    val_recs = [json.loads(line) for line in open(val_file, encoding="utf-8")]
    
    assert len(train_recs) == 200
    assert len(val_recs) == 40
    
    # Verbatim containment
    for r in train_recs:
        assert r["target_value"] in r["context"], f"Target {r['target_value']} not in train context"
    for r in val_recs:
        assert r["target_value"] in r["context"], f"Target {r['target_value']} not in val context"
        
    # Zero overlap
    train_ctxs = set(r["context"] for r in train_recs)
    val_ctxs = set(r["context"] for r in val_recs)
    assert len(train_ctxs.intersection(val_ctxs)) == 0
    
    train_targets = set(r["target_value"] for r in train_recs)
    val_targets = set(r["target_value"] for r in val_recs)
    assert len(train_targets.intersection(val_targets)) == 0


def test_pilot_results_and_early_stop_gate():
    """Verify that the pilot results file exists and early-stop rule was strictly executed."""
    results_path = EXP_DIR / "PILOT_RESULTS.json"
    assert results_path.exists()
    
    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    assert data["completed_steps"] == 200
    assert data["early_stopped"] is True
    assert data["step_200_metrics"]["exact_match_count"] < 4
    assert data["verdict"].startswith("FAIL")
    assert (EXP_DIR / "checkpoint_step_200.pt").exists()
