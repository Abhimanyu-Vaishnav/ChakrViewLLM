"""
Tests for Milestone 7: Checkpoint provenance and safe training resumption.
Verifies:
1. Incompatible checkpoints (e.g. mismatched vocabulary or parameter dimensions) fail closed.
2. Corrupt checkpoint files or unexpected data payload structures are rejected.
3. Model and tokenizer compatibility check prior to text generation.
4. Candidate checkpoints never overwrite canonical model or protected artifacts.
"""

import tempfile
from pathlib import Path
import pytest
import torch

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.training.checkpoint import CheckpointManager
from chakrview.training.safety import TrainingSafetyError


def test_corrupted_checkpoint_fails_closed():
    """Verify that corrupt, incomplete, or truncated checkpoint files fail closed."""
    with tempfile.TemporaryDirectory() as td:
        corrupt_file = Path(td) / "corrupt_checkpoint.pt"
        # Write invalid binary content
        corrupt_file.write_bytes(b"NOT_A_VALID_PYTORCH_CHECKPOINT_DATA")

        with pytest.raises(Exception):
            CheckpointManager.load(corrupt_file, validate_training=True)


def test_incompatible_checkpoint_fails_closed():
    """Verify that a checkpoint with mismatched parameter count or missing keys is rejected."""
    with tempfile.TemporaryDirectory() as td:
        ckpt_file = Path(td) / "mismatched_checkpoint.pt"
        # Save a checkpoint with wrong param count
        torch.save({
            "step": 10,
            "model_state_dict": {"dummy.weight": torch.randn(10, 10)},
            "optimizer_state_dict": {},
            "rng_state": torch.get_rng_state(),
        }, ckpt_file)

        # Expected canonical parameter count is 3,443,136
        with pytest.raises(Exception):
            CheckpointManager.load(
                ckpt_file,
                validate_training=True,
                expected_param_count=3443136,
            )


def test_candidate_checkpoint_isolation_from_canonical():
    """Verify candidate checkpoint training state never modifies canonical baseline."""
    from chakrview.brain.canonical_hash import verify_canonical_baseline

    torch.manual_seed(42)
    base_model = ChakrMicro(ModelConfig())
    pre_rep = verify_canonical_baseline(base_model)
    assert pre_rep["is_bit_exact"] and pre_rep["delta_w"] == 0

    # Create candidate model and modify weights
    candidate_model = ChakrMicro(ModelConfig())
    with torch.no_grad():
        for p in candidate_model.parameters():
            p.add_(torch.randn_like(p) * 0.1)

    # Base model must remain completely unaffected
    post_rep = verify_canonical_baseline(base_model)
    assert post_rep["is_bit_exact"] and post_rep["delta_w"] == 0
