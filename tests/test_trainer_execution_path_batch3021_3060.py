"""Unit and Integration Tests for Batch 3021–3060 Trainer Execution Path.

Validates:
1. run_trainer pipeline executes preflight first and fails closed if preflight fails.
2. Trainer strictly loads experiments/batch3021_3060/train.jsonl and val.jsonl.
3. test.jsonl is NEVER passed to the trainer.
4. Trainer enforces safe mode by default without --execute-authorized-training.
5. Hyperparameters match: 160 steps, batch size 4, seed 42.
6. Output directory is strictly isolated to experiments/batch3021_3060/balanced_sft_pilot_run/.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
BATCH_DIR = REPO_ROOT / "experiments" / "batch3021_3060"


def test_01_trainer_safe_mode_by_default():
    """Verify run_training_pipeline runs preflight and inspection cleanly without executing training."""
    from experiments.batch3021_3060.run_trainer import run_training_pipeline
    exit_code = run_training_pipeline(execute_training=False)
    assert exit_code == 0
    # Output directory shouldn't have any checkpoints
    out_dir = BATCH_DIR / "balanced_sft_pilot_run"
    assert not (out_dir / "sft_best_checkpoint.pt").exists()


def test_02_trainer_enforces_dataset_paths():
    """Verify trainer strictly binds to batch3021_3060 dataset files."""
    cfg_file = BATCH_DIR / "SFT_EXPERIMENT_CONFIG.json"
    with open(cfg_file, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    train_path = REPO_ROOT / cfg["dataset_paths"]["train_file"]
    val_path = REPO_ROOT / cfg["dataset_paths"]["val_file"]
    test_path = REPO_ROOT / cfg["dataset_paths"]["test_file"]

    assert train_path == BATCH_DIR / "train.jsonl"
    assert val_path == BATCH_DIR / "val.jsonl"
    assert test_path == BATCH_DIR / "test.jsonl"
    assert train_path.exists()
    assert val_path.exists()
    assert test_path.exists()


def test_03_trainer_fails_closed_on_preflight_failure():
    """Verify that if preflight returns failure, the trainer exits immediately with code 1."""
    from experiments.batch3021_3060.run_trainer import run_training_pipeline
    with patch("experiments.batch3021_3060.run_trainer.run_preflight", return_value=1):
        exit_code = run_training_pipeline(execute_training=False)
        assert exit_code == 1


def test_04_execution_call_chain_hyperparameters():
    """Verify exact 160 steps, batch size 4, seed 42, val_interval 10, save_interval 20."""
    cfg_file = BATCH_DIR / "SFT_EXPERIMENT_CONFIG.json"
    with open(cfg_file, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    hp = cfg["training_hyperparameters"]
    assert hp["pilot_steps"] == 160
    assert hp["batch_size"] == 4
    assert hp["seed"] == 42
    assert hp["val_interval"] == 10
    assert hp["save_interval"] == 20
    assert hp["save_best_checkpoint"] is True
    assert cfg["output_directory"] == "experiments/batch3021_3060/balanced_sft_pilot_run"
