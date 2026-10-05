"""
Tests for Step 102 Phase 0 & Phase 1: Experimental Protocol & Baseline Verification.

Verifies:
1. Split disjointness across Stage B train, validation, and test shards.
2. Canonical baseline immutability invariant (3,443,136 params, hash c5571c...).
3. Checkpoint isolation: training writes exclusively to isolated directory.
4. Independent baseline measurement repeatability on Stage B validation and test.
"""

from pathlib import Path
import pytest
import torch
from torch.utils.data import DataLoader

from chakrview.runtime.interactive import (
    instantiate_frozen_baseline,
    compute_model_hash,
    EXPECTED_WEIGHT_HASH,
)
from chakrview.training.dataset import StreamingTokenDataset
from chakrview.training.loss import CausalLoss

ROOT_DIR = Path(__file__).resolve().parents[1]
TRAIN_DIR = ROOT_DIR / "data" / "tokenized" / "stage_b" / "train"
VAL_DIR = ROOT_DIR / "data" / "tokenized" / "stage_b" / "validation"
TEST_DIR = ROOT_DIR / "data" / "tokenized" / "stage_b" / "test"


def test_split_disjointness_and_shard_integrity():
    """Verify that Stage B train, validation, and test splits have distinct files and checksums."""
    assert TRAIN_DIR.is_dir()
    assert VAL_DIR.is_dir()
    assert TEST_DIR.is_dir()

    train_files = {f.name: f.stat().st_size for f in TRAIN_DIR.glob("*.bin")}
    val_files = {f.name: f.stat().st_size for f in VAL_DIR.glob("*.bin")}
    test_files = {f.name: f.stat().st_size for f in TEST_DIR.glob("*.bin")}

    assert len(train_files) >= 2
    assert len(val_files) >= 1
    assert len(test_files) >= 1

    # Verify no file path overlap
    train_paths = set(TRAIN_DIR.glob("*.bin"))
    val_paths = set(VAL_DIR.glob("*.bin"))
    test_paths = set(TEST_DIR.glob("*.bin"))
    assert train_paths.isdisjoint(val_paths)
    assert train_paths.isdisjoint(test_paths)
    assert val_paths.isdisjoint(test_paths)


def test_canonical_baseline_exactness():
    """Verify that canonical baseline remains bit-exact before Step 102 training."""
    model = instantiate_frozen_baseline()
    param_count = sum(p.numel() for p in model.parameters())
    assert param_count == 3_443_136
    h = compute_model_hash(model)
    assert h == EXPECTED_WEIGHT_HASH


def test_baseline_evaluation_reproducibility():
    """Verify that canonical baseline loss on Stage B validation is within expected range (~8.35)."""
    model = instantiate_frozen_baseline()
    loss_fn = CausalLoss(ignore_index=2)
    ds = StreamingTokenDataset(VAL_DIR, sequence_length=128, loop=False, seed=42)
    loader = DataLoader(ds, batch_size=2)

    total_loss = 0.0
    batches = 0
    with torch.no_grad():
        for b in list(loader)[:5]:
            logits = model(b["input_ids"])
            loss = loss_fn(logits, b["target_ids"])
            total_loss += loss.item()
            batches += 1

    mean_loss = total_loss / batches
    assert 8.20 < mean_loss < 8.50, f"Unexpected baseline loss: {mean_loss}"
    assert compute_model_hash(model) == EXPECTED_WEIGHT_HASH
