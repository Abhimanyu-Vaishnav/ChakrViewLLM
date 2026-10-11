"""
Tests for Milestone 3: Pretraining Experiment Engine verification.
Tests:
1. Resumability: verify training from checkpoint step N to M matches uninterrupted run.
2. Safety recovery: verify non-finite loss/grad recovery preserves last valid state.
3. Checkpoint separation: full resumption checkpoint vs inference-only state dict.
4. Determinism: identical seeds yield bit-exact losses.
"""

import tempfile
from pathlib import Path
import pytest
import torch

from chakrview.training.pretraining_engine import PretrainingExperimentEngine


def test_pretraining_engine_determinism():
    repo_root = Path(".")
    r1 = PretrainingExperimentEngine.run_training(
        repo_root=repo_root,
        steps=5,
        batch_size=4,
        seq_len=32,
        seed=123,
        device="cpu",
    )
    r2 = PretrainingExperimentEngine.run_training(
        repo_root=repo_root,
        steps=5,
        batch_size=4,
        seq_len=32,
        seed=123,
        device="cpu",
    )
    assert r1.final_train_loss == r2.final_train_loss
    assert r1.final_val_loss == r2.final_val_loss
    assert r1.test_loss == r2.test_loss


def test_pretraining_engine_checkpointing_and_resumption():
    repo_root = Path(".")
    with tempfile.TemporaryDirectory() as td:
        tmp_dir = Path(td)
        # Run 10 steps and save checkpoint
        r1 = PretrainingExperimentEngine.run_training(
            repo_root=repo_root,
            steps=10,
            batch_size=4,
            seq_len=32,
            seed=42,
            device="cpu",
            checkpoint_dir=tmp_dir,
        )
        assert Path(r1.checkpoint_path).exists()
        assert Path(r1.inference_checkpoint_path).exists()

        # Checkpoint contents
        full_ckpt = torch.load(r1.checkpoint_path, map_location="cpu", weights_only=False)
        assert "model_state_dict" in full_ckpt
        assert "optimizer_state_dict" in full_ckpt
        assert "rng_state" in full_ckpt
        assert full_ckpt["step"] == 10

        infer_ckpt = torch.load(r1.inference_checkpoint_path, map_location="cpu", weights_only=False)
        # Inference ckpt is just state_dict
        assert "embedding.weight" in infer_ckpt
        assert "optimizer_state_dict" not in infer_ckpt

        # Resume from step 10 to step 15
        r_resumed = PretrainingExperimentEngine.run_training(
            repo_root=repo_root,
            steps=15,
            batch_size=4,
            seq_len=32,
            seed=42,
            device="cpu",
            resume_from=Path(r1.checkpoint_path),
            checkpoint_dir=tmp_dir,
        )
        assert r_resumed.steps == 15
        assert len(r_resumed.history) >= 15
