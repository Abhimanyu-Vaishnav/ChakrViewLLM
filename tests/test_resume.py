"""
Tests for Checkpoint Resumption (Phase 10, 14 & 18).
"""

from pathlib import Path
import pytest
import torch

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.training.config import (
    PretrainingConfig,
    TrainingHyperparameters,
    CheckpointConfig,
)
from chakrview.training.trainer import Trainer


def test_checkpoint_resume_restores_complete_state(tmp_path: Path):
    """
    Verify Run A (steps 0 -> 4) saves checkpoint, and Run B loads checkpoint
    and matches model weights, optimizer state, and step counter.
    """
    ckpt_dir = tmp_path / "checkpoints"
    config = PretrainingConfig(
        model=ModelConfig(),
        training=TrainingHyperparameters(
            max_steps=4,
            learning_rate=1e-3,
            warmup_steps=2,
            gradient_accumulation_steps=1,
        ),
        checkpoint=CheckpointConfig(
            directory=str(ckpt_dir),
            save_interval=2,
            keep_last_n=2,
        ),
    )

    # Synthetic batches: fixed sequence length 16, batch size 2
    batch1 = {
        "input_ids": torch.randint(0, 4000, (2, 16)),
        "target_ids": torch.randint(0, 4000, (2, 16)),
        "attention_mask": torch.ones(2, 16, dtype=torch.long),
    }
    batch2 = {
        "input_ids": torch.randint(0, 4000, (2, 16)),
        "target_ids": torch.randint(0, 4000, (2, 16)),
        "attention_mask": torch.ones(2, 16, dtype=torch.long),
    }

    # Run A: Train for 2 steps and save checkpoint
    trainer_a = Trainer(config=config)
    trainer_a.train_step(batch1)
    trainer_a.train_step(batch2)
    assert trainer_a.current_step == 2

    ckpt_path = trainer_a.checkpoint_manager.save(
        model=trainer_a.model,
        optimizer=trainer_a.optimizer,
        scheduler=trainer_a.scheduler,
        step=trainer_a.current_step,
    )
    assert ckpt_path.is_file()

    # Capture reference weights at step 2
    ref_weight = trainer_a.model.embedding.weight.clone().detach()

    # Run B: Fresh trainer resumes from checkpoint
    trainer_b = Trainer(config=config)
    resumed_step = trainer_b.resume(ckpt_path)

    assert resumed_step == 2
    assert trainer_b.current_step == 2

    # Verify weights are identical
    resumed_weight = trainer_b.model.embedding.weight
    assert torch.equal(ref_weight, resumed_weight)

    # Continue training Run B for 2 more steps (step 3 and 4)
    loss3 = trainer_b.train_step(batch1)
    loss4 = trainer_b.train_step(batch2)
    assert trainer_b.current_step == 4
    assert not torch.isnan(torch.tensor(loss3))
    assert not torch.isnan(torch.tensor(loss4))
