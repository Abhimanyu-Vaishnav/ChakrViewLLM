"""
Tests for Phase 5: Deterministic Smoke-Training Experiment Reproducibility.

Runs two identical training passes from scratch on a small controlled shard subset
with seed=42 and verifies bit-exact parameter replication and loss curves.
"""

from pathlib import Path
import pytest
import torch
from torch.utils.data import DataLoader

from chakrview.brain.config import ModelConfig
from chakrview.training.config import (
    PretrainingConfig,
    TrainingHyperparameters,
    CheckpointConfig,
    EvaluationConfig,
    DataConfig,
)
from chakrview.training.trainer import Trainer
from chakrview.training.dataset import StreamingTokenDataset
from chakrview.runtime.interactive import compute_model_hash

ROOT_DIR = Path(__file__).resolve().parents[1]
TRAIN_SHARD_DIR = ROOT_DIR / "data" / "tokenized" / "train"
VAL_SHARD_DIR = ROOT_DIR / "data" / "tokenized" / "val"


def test_first_model_training_experiment_reproducibility(tmp_path: Path):
    """
    Execute two independent training runs with identical configuration and seed=42.
    Verifies that loss trajectories and final weights match bit-exactly.
    """
    seq_len = 32
    max_steps = 6

    def run_training_experiment(run_id: str):
        run_ckpt_dir = tmp_path / f"ckpt_run_{run_id}"
        config = PretrainingConfig(
            model=ModelConfig(),
            training=TrainingHyperparameters(
                max_steps=max_steps,
                learning_rate=2e-3,
                warmup_steps=1,
                gradient_accumulation_steps=1,
                gradient_clipping=1.0,
                seed=42,
            ),
            checkpoint=CheckpointConfig(
                directory=str(run_ckpt_dir),
                save_interval=3,
                keep_last_n=2,
            ),
            evaluation=EvaluationConfig(
                eval_interval=3,
                eval_batches=1,
                eval_on_start=False,
            ),
            data=DataConfig(
                train_path=str(TRAIN_SHARD_DIR),
                validation_path=str(VAL_SHARD_DIR),
                sequence_length=seq_len,
            ),
        )

        train_ds = StreamingTokenDataset(
            shard_dir=TRAIN_SHARD_DIR,
            sequence_length=seq_len,
            loop=True,
            seed=42,
        )
        val_ds = StreamingTokenDataset(
            shard_dir=VAL_SHARD_DIR,
            sequence_length=seq_len,
            loop=True,
            seed=42,
        )

        train_loader = DataLoader(train_ds, batch_size=config.training.batch_size)
        val_loader = DataLoader(val_ds, batch_size=config.training.batch_size)

        trainer = Trainer(
            config=config,
            train_loader=train_loader,
            val_loader=val_loader,
        )
        res = trainer.train()
        losses = [step["train_loss"] for step in trainer.metrics.history]
        weight_hash = compute_model_hash(trainer.model)
        return losses, weight_hash

    losses1, hash1 = run_training_experiment("1")
    losses2, hash2 = run_training_experiment("2")

    # Verify identical loss progression
    assert len(losses1) == max_steps
    assert len(losses2) == max_steps
    for l1, l2 in zip(losses1, losses2):
        assert abs(l1 - l2) < 1e-6, f"Divergence in loss: {l1} vs {l2}"

    # Verify bit-exact final weights
    assert hash1 == hash2, f"Weight hashes diverged: {hash1} vs {hash2}"
