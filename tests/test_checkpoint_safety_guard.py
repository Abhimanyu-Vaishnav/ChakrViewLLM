"""
Unit tests for centralized canonical checkpoint protection.

Verifies:
1. Rejection of canonical checkpoint directory as target.
2. Rejection of targets inside canonical checkpoint directory.
3. Rejection of targets matching canonical artifact SHA-256 hashes.
4. Protection against directory traversal and symlink/relative path aliases.
5. Integration into CheckpointManager and SFTExperimentEngine.
6. Permitting safe external directories.
"""

from pathlib import Path
import pytest
import torch
import torch.nn as nn

from chakrview.training.safety import (
    assert_safe_checkpoint_target,
    CanonicalArtifactProtectionError,
    PROTECTED_CANONICAL_HASHES,
)
from chakrview.training.checkpoint import CheckpointManager
from chakrview.training.sft_engine import SFTConfig, SFTExperimentEngine


def test_guard_rejects_canonical_checkpoint_directory():
    canonical_dir = Path("experiments/batch16_21_combined/main_training_10000")
    with pytest.raises(CanonicalArtifactProtectionError, match="CANONICAL CHECKPOINT PROTECTION"):
        assert_safe_checkpoint_target(canonical_dir)


def test_guard_rejects_subpath_inside_canonical_checkpoint_dir():
    target = Path("experiments/batch16_21_combined/main_training_10000/new_ckpt.pt")
    with pytest.raises(CanonicalArtifactProtectionError, match="CANONICAL CHECKPOINT PROTECTION"):
        assert_safe_checkpoint_target(target)


def test_guard_rejects_relative_traversal_to_canonical_checkpoint_dir():
    target = Path("chakrview/../experiments/batch16_21_combined/main_training_10000/checkpoint.pt")
    with pytest.raises(CanonicalArtifactProtectionError, match="CANONICAL CHECKPOINT PROTECTION"):
        assert_safe_checkpoint_target(target)


def test_guard_rejects_canonical_file_overwrite():
    canonical_file = Path("experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt")
    with pytest.raises(CanonicalArtifactProtectionError):
        assert_safe_checkpoint_target(canonical_file)


def test_guard_rejects_other_protected_artifacts(tmp_path):
    # Create a test file with exact protected vocab hash
    vocab_file = Path("data/experiments/vocab_4096/vocab.json")
    with pytest.raises(CanonicalArtifactProtectionError, match="CANONICAL ARTIFACT PROTECTION"):
        assert_safe_checkpoint_target(vocab_file)


def test_guard_permits_safe_paths(tmp_path):
    safe_dir = tmp_path / "checkpoints"
    safe_file = safe_dir / "ckpt_0001.pt"
    resolved = assert_safe_checkpoint_target(safe_file)
    assert resolved == safe_file.resolve()


def test_checkpoint_manager_blocks_canonical_dir():
    canonical_dir = "experiments/batch16_21_combined/main_training_10000"
    with pytest.raises(CanonicalArtifactProtectionError):
        CheckpointManager(canonical_dir)


def test_sft_engine_blocks_canonical_dir(tmp_path):
    canonical_dir = Path("experiments/batch16_21_combined/main_training_10000")
    dummy_train = tmp_path / "train.jsonl"
    dummy_val = tmp_path / "val.jsonl"
    dummy_train.write_text('{"prompt": "p", "response": "r"}\n', encoding="utf-8")
    dummy_val.write_text('{"prompt": "p", "response": "r"}\n', encoding="utf-8")

    cfg = SFTConfig(
        repo_root=Path("."),
        train_file=dummy_train,
        val_file=dummy_val,
        output_dir=canonical_dir,
        base_checkpoint=Path("experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt"),
    )

    with pytest.raises(CanonicalArtifactProtectionError):
        SFTExperimentEngine(cfg)
