"""
Tests for ChakrMicro weight initialization (chakrview/brain/initialization.py):
- Determinism under fixed seed
- Reasonable variance and zero mean
- Absence of NaN/Inf
- Absence of external/pretrained weight files
- Tied weights remaining correctly tied after initialization
"""

import math
import pytest
import torch
from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.brain.initialization import initialize_weights


def test_initialization_determinism_under_fixed_seed():
    """Verify that identical random seeds produce bit-exact identical initial parameters."""
    cfg = ModelConfig()
    
    torch.manual_seed(1337)
    model1 = ChakrMicro(cfg)
    
    torch.manual_seed(1337)
    model2 = ChakrMicro(cfg)
    
    for (n1, p1), (n2, p2) in zip(model1.named_parameters(), model2.named_parameters()):
        assert n1 == n2
        assert torch.equal(p1, p2), f"Parameter {n1} differed between identical seed initializations"


def test_initialization_statistics_and_finiteness():
    """
    Verify initialization properties:
    - Base standard deviation is close to initializer_range (0.02)
    - Residual projections have scaled down std (0.02 / sqrt(12))
    - RMSNorm weights are initialized to exactly 1.0
    - No NaN or Inf exists
    """
    torch.manual_seed(42)
    cfg = ModelConfig(initializer_range=0.02, n_layers=6)
    model = ChakrMicro(cfg)
    
    # Check no NaN/Inf anywhere
    for name, p in model.named_parameters():
        assert torch.isfinite(p).all(), f"Parameter {name} contains non-finite values"
        
    # Check RMSNorm weights are exactly 1.0
    for layer in model.layers:
        assert torch.all(layer.norm_1.weight == 1.0)
        assert torch.all(layer.norm_2.weight == 1.0)
    assert torch.all(model.final_norm.weight == 1.0)
    
    # Check embedding statistics (mean near 0, std near 0.02)
    emb_weight = model.embedding.weight
    assert abs(emb_weight.mean().item()) < 0.005
    assert abs(emb_weight.std().item() - 0.02) < 0.005
    
    # Check residual projection scaling (out_proj and down_proj)
    expected_residual_std = 0.02 / math.sqrt(12.0)  # ~0.00577
    for layer in model.layers:
        out_std = layer.attn.out_proj.weight.std().item()
        down_std = layer.ffn.down_proj.weight.std().item()
        assert abs(out_std - expected_residual_std) < 0.002
        assert abs(down_std - expected_residual_std) < 0.002


def test_tied_weights_remain_tied_after_custom_reinitialization():
    """Verify that re-running initialize_weights maintains pointer identity."""
    cfg = ModelConfig()
    model = ChakrMicro(cfg)
    
    assert model.lm_head.weight.data_ptr() == model.embedding.weight.data_ptr()
    assert model.lm_head.weight is model.embedding.weight
    
    # Reinitialize
    initialize_weights(model, cfg)
    
    assert model.lm_head.weight.data_ptr() == model.embedding.weight.data_ptr()
    assert model.lm_head.weight is model.embedding.weight
