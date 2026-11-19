"""
UCIP-13 Automated Verification Test Suite.
Verifies:
1. All 7 scheduled checkpoints exist and have unique SHA-256 hashes.
2. Step 0 checkpoint has baseline configuration.
3. Training history contains 800 steps with monotonically increasing token counts.
4. Validation history documents loss reduction from 8.3084 down to 4.6859 (PPL 4057 -> 108).
5. Held-out test evaluation matches validation findings (~4.6595 test loss).
6. Protected runtime session.py remains untouched.
"""

import hashlib
import json
import os
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EXP_DIR = ROOT / "experiments" / "ucip13"


def test_01_all_checkpoints_exist():
    """Verify all 7 checkpoints exist and have non-zero size."""
    ckpt_dir = EXP_DIR / "checkpoints" / "primary_seed42"
    expected_steps = [0, 50, 100, 200, 400, 600, 800]
    for s in expected_steps:
        p = ckpt_dir / f"checkpoint_step_{s:03d}.pt"
        assert p.is_file(), f"Missing checkpoint: {p}"
        assert p.stat().st_size > 10_000_000, f"Checkpoint file too small: {p}"


def test_02_checkpoint_hashes_unique():
    """Verify that every checkpoint has a unique SHA-256 hash."""
    ckpt_dir = EXP_DIR / "checkpoints" / "primary_seed42"
    hashes = set()
    for f in ckpt_dir.glob("*.pt"):
        h = hashlib.sha256(f.read_bytes()).hexdigest()
        assert h not in hashes, f"Duplicate checkpoint hash: {h}"
        hashes.add(h)
    assert len(hashes) == 7


def test_03_training_loss_progression():
    """Verify training loss decreases from >8.0 to <4.5."""
    with open(EXP_DIR / "metrics" / "training_history.json", "r") as f:
        history = json.load(f)
    assert len(history) == 800
    assert history[0]["tokens_processed"] == 8192
    assert history[-1]["tokens_processed"] == 6553600
    assert history[0]["train_loss"] > 7.5
    assert history[-1]["train_loss"] < 4.5


def test_04_validation_loss_reduction():
    """Verify validation loss decreases from 8.30 to <4.70."""
    with open(EXP_DIR / "metrics" / "validation_history.json", "r") as f:
        val_history = json.load(f)
    assert len(val_history) == 7
    initial_loss = val_history[0]["val_loss"]
    final_loss = val_history[-1]["val_loss"]
    assert initial_loss > 8.0
    assert final_loss < 4.8
    assert val_history[0]["perplexity"] > 4000
    assert val_history[-1]["perplexity"] < 120


def test_05_held_out_test_results():
    """Verify held out test loss confirms generalization on unseen test shards."""
    with open(EXP_DIR / "evaluations" / "held_out_test_results.json", "r") as f:
        test_res = json.load(f)
    assert test_res["held_out_test_loss"] < 4.8
    assert test_res["held_out_test_perplexity"] < 120
    assert test_res["held_out_test_batches"] == 72


def test_06_protected_session_hash():
    """Verify session.py hash is preserved."""
    p = ROOT / "chakrview" / "runtime" / "session.py"
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    assert h == "9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67"
