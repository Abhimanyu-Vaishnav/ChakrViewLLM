"""
Tests for Atomic Checkpoint System and Failure Safety (Phase 10, 15 & 18).
"""

import os
import json
from pathlib import Path
import pytest
import torch

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.training.config import TrainingHyperparameters
from chakrview.training.optimizer import build_optimizer, build_lr_scheduler
from chakrview.training.checkpoint import CheckpointManager
from chakrview.training.seed import get_rng_state


def test_atomic_checkpoint_save_and_pointer(tmp_path: Path):
    """Verify atomic checkpoint save writes valid payload and pointer file."""
    cfg = ModelConfig()
    model = ChakrMicro(cfg)
    hypers = TrainingHyperparameters()
    optimizer = build_optimizer(model, hypers)
    scheduler = build_lr_scheduler(optimizer, hypers)

    ckpt_mgr = CheckpointManager(checkpoint_dir=tmp_path, keep_last_n=3)
    saved_path = ckpt_mgr.save(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        step=42,
        epoch=1,
        config={"test": "val"},
        rng_state=get_rng_state(),
        train_metrics={"loss": 2.5},
        val_metrics={"val_loss": 2.6},
    )

    assert saved_path.is_file()
    assert saved_path.name == "checkpoint_0000042.pt"

    # Verify no .tmp files linger
    tmp_files = list(tmp_path.glob("*.tmp"))
    assert len(tmp_files) == 0

    # Verify pointer file
    latest_path = ckpt_mgr.get_latest_checkpoint_path()
    assert latest_path == saved_path

    # Verify payload content
    payload = CheckpointManager.load(saved_path)
    assert payload["step"] == 42
    assert payload["epoch"] == 1
    assert payload["config"] == {"test": "val"}
    assert payload["train_metrics"] == {"loss": 2.5}
    assert payload["val_metrics"] == {"val_loss": 2.6}
    assert "model_state_dict" in payload
    assert "optimizer_state_dict" in payload
    assert "scheduler_state_dict" in payload
    assert "rng_state" in payload


def test_checkpoint_pruning_keeps_last_n(tmp_path: Path):
    """Verify CheckpointManager retains only keep_last_n checkpoints."""
    cfg = ModelConfig()
    model = ChakrMicro(cfg)
    ckpt_mgr = CheckpointManager(checkpoint_dir=tmp_path, keep_last_n=2)

    # Save 4 checkpoints
    for step in [10, 20, 30, 40]:
        ckpt_mgr.save(model=model, step=step)

    remaining = sorted(list(tmp_path.glob("checkpoint_*.pt")))
    assert len(remaining) == 2
    assert remaining[0].name == "checkpoint_0000030.pt"
    assert remaining[1].name == "checkpoint_0000040.pt"

    latest = ckpt_mgr.get_latest_checkpoint_path()
    assert latest.name == "checkpoint_0000040.pt"


def test_failure_safety_corrupted_tmp_does_not_corrupt_valid_checkpoint(tmp_path: Path):
    """Simulate interrupted/corrupted tmp write: existing checkpoint remains valid."""
    cfg = ModelConfig()
    model = ChakrMicro(cfg)
    ckpt_mgr = CheckpointManager(checkpoint_dir=tmp_path, keep_last_n=3)

    # 1. Save valid checkpoint at step 100
    valid_path = ckpt_mgr.save(model=model, step=100)
    assert valid_path.is_file()

    # 2. Simulate failed / interrupted write by writing a garbage .tmp file
    corrupt_tmp = tmp_path / "checkpoint_0000200.pt.tmp"
    with open(corrupt_tmp, "w") as f:
        f.write("PARTIAL_CORRUPT_BYTES")

    # Pointer and latest should still point to valid checkpoint 100
    latest = ckpt_mgr.get_latest_checkpoint_path()
    assert latest == valid_path

    # Checkpoint 100 loads without error
    loaded = CheckpointManager.load(latest)
    assert loaded["step"] == 100
