"""
Tests for Advanced Relational Representation & Inductive Generalization Suite.

Location: tests/test_inductive_representation_suite.py
"""

import tempfile
from pathlib import Path
import pytest
import torch

from chakrview.brain.canonical_hash import verify_canonical_baseline
from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.cognition.v050_candidate.inductive_representation_suite import (
    InductiveRepresentationSuite,
    InductiveBenchmarkMetrics,
)


def test_canonical_baseline_immutability():
    """Ensure tests do not mutate the canonical seed-42 baseline."""
    torch.manual_seed(42)
    m = ChakrMicro(ModelConfig())
    res = verify_canonical_baseline(m)
    assert res["is_bit_exact"] is True
    assert res["delta_w"] == 0
    assert res["total_params"] == 3443136


def test_data_splits_isolation_and_no_leakage():
    """Verify train, validation, and test sets have disjoint entity token IDs."""
    (tx, ty), (vx, vy), (tex, tey) = InductiveRepresentationSuite.generate_data(mode="categorical", seed=42)

    train_entities = set(tx[:, 1].tolist())
    val_entities = set(vx[:, 1].tolist())
    test_entities = set(tex[:, 1].tolist())

    # Strict disjointness
    assert train_entities.isdisjoint(val_entities), "Train and Val entities overlap!"
    assert train_entities.isdisjoint(test_entities), "Train and Test entities overlap!"
    assert val_entities.isdisjoint(test_entities), "Val and Test entities overlap!"


def test_naive_atomic_retains_negative_gap():
    """Confirms diagnostic Mode 1 reproduces the known generalization gap on atomic IDs."""
    metrics = InductiveRepresentationSuite.run_experiment(mode="naive_atomic", steps=25, seed=42)
    assert metrics.train_accuracy == 1.0
    assert metrics.test_accuracy == 0.0
    assert metrics.generalization_gap == 1.0
    assert metrics.is_generalizing is False


def test_categorical_grounded_achieves_generalization():
    """Confirms diagnostic Mode 3 demonstrates zero-shot transfer on held-out entities."""
    metrics = InductiveRepresentationSuite.run_experiment(mode="categorical", steps=35, seed=42)
    assert metrics.train_accuracy == 1.0
    assert metrics.val_accuracy == 1.0
    assert metrics.test_accuracy == 1.0
    assert metrics.test_loss < 0.20
    assert metrics.is_generalizing is True


def test_checkpoint_isolation_and_reload_consistency():
    """Verifies candidate checkpoint serialization and deterministic round-trip reload."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        metrics = InductiveRepresentationSuite.run_experiment(
            mode="categorical",
            steps=20,
            seed=42,
            checkpoint_dir=tmp_path,
        )
        assert metrics.checkpoint_hash is not None

        ckpt_file = tmp_path / "candidate_categorical_seed_42.pt"
        assert ckpt_file.exists()

        loaded_data = torch.load(ckpt_file, map_location="cpu")
        assert "state_dict" in loaded_data
        assert "metrics" in loaded_data
        assert loaded_data["mode"] == "categorical"

        # Reload candidate model and verify identical test predictions
        cand = ChakrMicro(ModelConfig())
        cand.load_state_dict(loaded_data["state_dict"])
        cand.eval()

        _, _, (tex, tey) = InductiveRepresentationSuite.generate_data(mode="categorical", seed=42)
        with torch.no_grad():
            preds = cand(tex)[:, -1, :].argmax(dim=-1)
            acc = (preds == tey).float().mean().item()
        assert acc == 1.0


def test_negative_invalid_mode_raises():
    """Negative test: ensure invalid mode fails closed."""
    with pytest.raises(ValueError, match="Unknown mode"):
        InductiveRepresentationSuite.generate_data(mode="non_existent_mode")
