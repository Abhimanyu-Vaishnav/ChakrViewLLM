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


def test_multi_shard_synthetic_coverage_and_boundaries():
    """Verify MultiShardDataset correctly reads across shards, handles boundaries, and validates schema."""
    import numpy as np
    from chakrview.training.pretraining_engine import MultiShardDataset

    with tempfile.TemporaryDirectory() as td:
        tmp_path = Path(td)
        # Create 2 synthetic shards with 100 tokens each
        s0_tokens = np.arange(0, 100, dtype=np.uint16)
        s1_tokens = np.arange(100, 200, dtype=np.uint16)

        s0_file = tmp_path / "shard_00000.bin"
        s1_file = tmp_path / "shard_00001.bin"
        s0_tokens.tofile(s0_file)
        s1_tokens.tofile(s1_file)

        with MultiShardDataset([s0_file, s1_file]) as ds:
            assert len(ds) == 200

            # Slice completely within shard 0
            sl0 = ds.get_slice(10, 5)
            assert np.array_equal(sl0, np.arange(10, 15))

            # Slice completely within shard 1
            sl1 = ds.get_slice(110, 5)
            assert np.array_equal(sl1, np.arange(110, 115))

            # Slice crossing shard boundary (from token 95 to 105)
            sl_cross = ds.get_slice(95, 10)
            assert np.array_equal(sl_cross, np.arange(95, 105))

            # Sampling batch across shards
            rng = np.random.RandomState(42)
            bx, by = ds.get_batch(batch_size=4, seq_len=16, rng=rng)
            assert bx.shape == (4, 16)
            assert by.shape == (4, 16)
            # Verify next-token relationship
            assert torch.equal(bx[:, 1:], by[:, :-1])

        # Error cases: empty shard
        empty_file = tmp_path / "shard_empty.bin"
        empty_file.touch()
        with pytest.raises(ValueError, match="empty"):
            MultiShardDataset([empty_file])

        # Missing shard
        with pytest.raises(FileNotFoundError):
            MultiShardDataset([tmp_path / "nonexistent.bin"])


def test_continuous_vs_resumed_exact_reproducibility():
    """
    Verify bit-exact reproducibility between an uninterrupted 8-step training run
    and an interrupted 4-step + resumed 4-step run under identical configuration.
    """
    repo_root = Path(".")
    with tempfile.TemporaryDirectory() as td:
        tmp_dir = Path(td)

        # 1. Uninterrupted continuous run for 8 steps
        continuous_res = PretrainingExperimentEngine.run_training(
            repo_root=repo_root,
            steps=8,
            batch_size=4,
            seq_len=32,
            lr=5e-4,
            seed=777,
            device="cpu",
            val_interval=4,
            checkpoint_dir=tmp_dir / "cont",
            use_multi_shard=False,  # Single shard slice for speed
        )

        # 2. Part 1: Run 4 steps and save checkpoint
        part1_res = PretrainingExperimentEngine.run_training(
            repo_root=repo_root,
            steps=4,
            batch_size=4,
            seq_len=32,
            lr=5e-4,
            seed=777,
            device="cpu",
            val_interval=4,
            checkpoint_dir=tmp_dir / "interrupted",
            use_multi_shard=False,
        )

        # 3. Part 2: Resume from step 4 up to step 8
        resumed_res = PretrainingExperimentEngine.run_training(
            repo_root=repo_root,
            steps=8,
            batch_size=4,
            seq_len=32,
            lr=5e-4,
            seed=777,
            device="cpu",
            val_interval=4,
            resume_from=Path(part1_res.checkpoint_path),
            checkpoint_dir=tmp_dir / "resumed",
            use_multi_shard=False,
        )

        # Assert loss progression matches bit-for-bit
        cont_train_losses = [h["train_loss"] for h in continuous_res.history]
        resumed_train_losses = [h["train_loss"] for h in resumed_res.history]
        assert cont_train_losses == resumed_train_losses, f"Loss mismatch: continuous {cont_train_losses} vs resumed {resumed_train_losses}"

        assert continuous_res.final_train_loss == resumed_res.final_train_loss
        assert continuous_res.final_val_loss == resumed_res.final_val_loss
        assert continuous_res.test_loss == resumed_res.test_loss

        # Compare final model weights bit-for-bit
        cont_ckpt = torch.load(continuous_res.checkpoint_path, map_location="cpu", weights_only=False)
        resumed_ckpt = torch.load(resumed_res.checkpoint_path, map_location="cpu", weights_only=False)
        for k in cont_ckpt["model_state_dict"]:
            diff = (cont_ckpt["model_state_dict"][k] - resumed_ckpt["model_state_dict"][k]).abs().max().item()
            assert diff == 0.0, f"Weight divergence in layer {k}: max diff = {diff}"


def test_legacy_checkpoint_backward_compatibility():
    """Verify that a legacy checkpoint lacking numpy_rng_state resumes cleanly without crashing."""
    repo_root = Path(".")
    with tempfile.TemporaryDirectory() as td:
        tmp_dir = Path(td)
        # Create a mock legacy checkpoint without numpy_rng_state or cuda_rng_state
        from chakrview.brain.config import ModelConfig
        from chakrview.brain.model import ChakrMicro
        model = ChakrMicro(ModelConfig(max_seq_len=512))
        legacy_ckpt = tmp_dir / "legacy_step_2.pt"
        torch.save({
            "step": 2,
            "seed": 42,
            "config": ModelConfig(max_seq_len=512),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": {},
            "rng_state": torch.get_rng_state(),
            "history": [],
        }, legacy_ckpt)

        # Must resume cleanly through step 4
        res = PretrainingExperimentEngine.run_training(
            repo_root=repo_root,
            steps=4,
            batch_size=2,
            seq_len=16,
            device="cpu",
            resume_from=legacy_ckpt,
            use_multi_shard=False,
        )
        assert res.steps == 4
        assert len(res.history) == 2


def test_evaluate_split_representative_and_weighted():
    """Verify that evaluate_split calculates weighted loss and perplexity across multiple batches."""
    import numpy as np
    from chakrview.brain.config import ModelConfig
    from chakrview.brain.model import ChakrMicro
    from chakrview.training.pretraining_engine import MultiShardDataset

    with tempfile.TemporaryDirectory() as td:
        tmp_path = Path(td)
        s0 = np.arange(0, 100, dtype=np.uint16)
        s1 = np.arange(100, 200, dtype=np.uint16)
        (tmp_path / "s0.bin").write_bytes(s0.tobytes())
        (tmp_path / "s1.bin").write_bytes(s1.tobytes())

        with MultiShardDataset([tmp_path / "s0.bin", tmp_path / "s1.bin"]) as ds:
            model = ChakrMicro(ModelConfig(max_seq_len=64))
            metrics = PretrainingExperimentEngine.evaluate_split(
                model=model,
                dataset=ds,
                batch_size=2,
                seq_len=16,
                max_batches=3,
                seed=42,
            )
            assert metrics["batches_evaluated"] == 3
            assert metrics["tokens_evaluated"] == 3 * 2 * 16  # 96 tokens
            assert metrics["loss"] > 0.0
            assert metrics["perplexity"] > 1.0
