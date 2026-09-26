"""
Tests for ModelConfig in chakrview/brain/config.py.
"""

import pytest
from chakrview.brain.config import ModelConfig


def test_model_config_defaults():
    """Verify default hyperparameters match frozen Chakr-Micro v0.1 spec."""
    cfg = ModelConfig()
    assert cfg.vocab_size == 4096
    assert cfg.d_model == 192
    assert cfg.n_layers == 6
    assert cfg.n_heads == 6
    assert cfg.head_dim == 32
    assert cfg.max_seq_len == 512
    assert cfg.rms_norm_eps == 1e-5
    assert cfg.rope_theta == 10000.0
    assert cfg.hidden_dim == 512
    assert cfg.dropout == 0.0
    assert cfg.initializer_range == 0.02
    assert cfg.dtype == "float32"
    assert cfg.use_bias is False
    assert cfg.tie_embeddings is True
    assert cfg.bos_token_id == 0
    assert cfg.eos_token_id == 1
    assert cfg.pad_token_id == 2


def test_model_config_divisibility():
    """Reject configurations where d_model is not divisible by n_heads."""
    with pytest.raises(ValueError, match="must be divisible"):
        ModelConfig(d_model=192, n_heads=5)


def test_model_config_head_dim_parity():
    """Reject odd head dimensions (incompatible with pairwise RoPE)."""
    with pytest.raises(ValueError, match="must be even"):
        ModelConfig(d_model=190, n_heads=10)


def test_model_config_positive_bounds():
    """Reject non-positive hyperparameters."""
    with pytest.raises(ValueError):
        ModelConfig(vocab_size=0)
    with pytest.raises(ValueError):
        ModelConfig(d_model=-192)
    with pytest.raises(ValueError):
        ModelConfig(n_layers=0)
    with pytest.raises(ValueError):
        ModelConfig(hidden_dim=0)
    with pytest.raises(ValueError):
        ModelConfig(max_seq_len=-512)
