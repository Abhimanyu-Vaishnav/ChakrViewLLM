"""
Tests for Training Configuration System (Phase 2 & 18).
"""

import pytest
from pathlib import Path
import json

from chakrview.training.config import (
    PretrainingConfig,
    TrainingHyperparameters,
    DataConfig,
    CheckpointConfig,
    EvaluationConfig,
)
from configs.pretraining_config import get_default_pretraining_config, get_micro_test_config


def test_training_config_defaults_and_validation():
    """Verify default pretraining config adheres to frozen architectural bounds."""
    cfg = get_default_pretraining_config()
    
    # Model checks
    assert cfg.model.vocab_size == 4096
    assert cfg.model.d_model == 192
    assert cfg.model.n_layers == 6
    assert cfg.model.n_heads == 6
    assert cfg.model.max_seq_len == 512
    assert cfg.model.tie_embeddings is True

    # Training checks
    assert cfg.training.seed == 42
    assert cfg.training.batch_size == 2
    assert cfg.training.learning_rate == 5e-4
    assert cfg.training.optimizer == "adamw"

    # Data checks
    assert cfg.data.sequence_length == 512

    # Negative bounds checks
    with pytest.raises(ValueError, match="sequence_length.*exceeds max context"):
        DataConfig(sequence_length=513)

    with pytest.raises(ValueError, match="batch_size must be positive"):
        TrainingHyperparameters(batch_size=0)

    with pytest.raises(ValueError, match="save_interval must be positive"):
        CheckpointConfig(save_interval=-1)

    with pytest.raises(ValueError, match="eval_batches must be positive"):
        EvaluationConfig(eval_batches=0)


def test_training_config_json_serialization(tmp_path: Path):
    """Verify JSON round-trip serialization and deserialization."""
    cfg = get_micro_test_config()
    json_path = tmp_path / "test_cfg.json"
    
    cfg.save(json_path)
    assert json_path.is_file()

    loaded_cfg = PretrainingConfig.load(json_path)
    assert loaded_cfg.training.max_steps == cfg.training.max_steps
    assert loaded_cfg.model.d_model == 192
    assert loaded_cfg.data.sequence_length == 64
