"""
Tests for Pre-Training Trainer and Micro Training Test (Phase 9, 13 & 18).
"""

from pathlib import Path
import pytest
import torch

from chakrview.brain.config import ModelConfig
from chakrview.training.config import (
    PretrainingConfig,
    TrainingHyperparameters,
    CheckpointConfig,
    EvaluationConfig,
)
from chakrview.training.trainer import Trainer


def test_micro_training_loss_decreases(tmp_path: Path):
    """
    Phase 13: Micro training test.
    Verify model learns a simple deterministic repeating pattern,
    initial loss > final loss, and checkpoint is saved.
    """
    ckpt_dir = tmp_path / "checkpoints"
    config = PretrainingConfig(
        model=ModelConfig(),
        training=TrainingHyperparameters(
            max_steps=12,
            learning_rate=5e-3,
            warmup_steps=2,
            gradient_accumulation_steps=1,
            gradient_clipping=1.0,
        ),
        checkpoint=CheckpointConfig(
            directory=str(ckpt_dir),
            save_interval=6,
            keep_last_n=2,
        ),
        evaluation=EvaluationConfig(
            eval_interval=6,
            eval_batches=1,
            eval_on_start=False,
        ),
    )

    # Repeatable synthetic pattern: [10, 20, 30, 40, 50, ...]
    seq = list(range(10, 26))  # 16 tokens
    batch = {
        "input_ids": torch.tensor([seq[:-1]], dtype=torch.long),  # [1, 15]
        "target_ids": torch.tensor([seq[1:]], dtype=torch.long),  # [1, 15]
        "attention_mask": torch.ones(1, 15, dtype=torch.long),
    }

    # Endless synthetic dataset yielding the exact same batch
    class InfiniteDataset:
        def __iter__(self):
            return self
        def __next__(self):
            return batch

    trainer = Trainer(
        config=config,
        train_loader=InfiniteDataset(),
        val_loader=InfiniteDataset(),
    )

    result = trainer.train()
    history = result["history"]

    assert len(history) == 12
    initial_loss = history[0]["train_loss"]
    final_loss = history[-1]["train_loss"]

    # Strict learnability check: loss must decrease
    assert final_loss < initial_loss, f"Expected decrease: {initial_loss} -> {final_loss}"

    # Verify final checkpoint exists
    final_ckpt = Path(result["final_checkpoint"])
    assert final_ckpt.is_file()

    # Verify latest pointer points to final checkpoint
    latest = trainer.checkpoint_manager.get_latest_checkpoint_path()
    assert latest == final_ckpt


def test_gradient_accumulation_optimizer_step(tmp_path: Path):
    """Verify gradient accumulation steps delay optimizer update until window completes."""
    config = PretrainingConfig(
        model=ModelConfig(),
        training=TrainingHyperparameters(
            max_steps=2,
            learning_rate=1e-3,
            gradient_accumulation_steps=2,
        ),
        checkpoint=CheckpointConfig(directory=str(tmp_path)),
    )

    trainer = Trainer(config=config)
    batch = {
        "input_ids": torch.tensor([[1, 2, 3, 4]], dtype=torch.long),
        "target_ids": torch.tensor([[2, 3, 4, 5]], dtype=torch.long),
        "attention_mask": torch.ones(1, 4, dtype=torch.long),
    }

    # Micro-step 1: accumulates gradient, current_step remains 0
    trainer.train_step(batch)
    assert trainer.micro_step == 1
    assert trainer.current_step == 0

    # Micro-step 2: completes window, current_step advances to 1
    trainer.train_step(batch)
    assert trainer.micro_step == 2
    assert trainer.current_step == 1
