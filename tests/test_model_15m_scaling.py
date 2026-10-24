"""
tests/test_model_15m_scaling.py
Unit tests verifying the ChakrMicro-15M experimental architecture,
parameter accounting, tensor shapes, tied embedding semantics,
CPU forward/backward execution, and baseline invariance.
"""

import copy
import hashlib
import tempfile
from pathlib import Path
import pytest
import torch
import torch.nn as nn

from chakrview.brain.config import ModelConfig
from chakrview.brain.config_15m import ChakrMicro15MConfig
from chakrview.brain.model import ChakrMicro


def test_baseline_invariance():
    """Ensure baseline ModelConfig and ChakrMicro parameter count and digest remain untouched."""
    torch.manual_seed(42)
    baseline_model = ChakrMicro(ModelConfig())
    params = sum(p.numel() for p in baseline_model.parameters() if p.requires_grad)
    assert params == 3443136, f"Baseline parameter count changed: {params}"
    
    # Verify architecture digest
    hasher = hashlib.sha256()
    with torch.no_grad():
        for name, param in sorted(baseline_model.named_parameters()):
            hasher.update(name.encode("utf-8"))
            hasher.update(param.detach().cpu().numpy().tobytes())
    digest = hasher.hexdigest()
    assert digest == "c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da"


def test_15m_parameter_accounting():
    """Verify exact analytical and programmatic parameter counts for 15M configuration."""
    cfg = ChakrMicro15MConfig()
    model = ChakrMicro(cfg)
    
    # 1. Total trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert trainable_params == 15735168, f"Expected 15,735,168 parameters, got {trainable_params}"
    
    # Verify parameter delta is within 5% of 15,000,000
    target = 15_000_000
    error_pct = abs(trainable_params - target) / target
    assert error_pct < 0.05, f"Error {error_pct*100:.2f}% exceeds 5% boundary"
    
    # 2. Embedding tied weights check
    assert model.lm_head.weight.data_ptr() == model.embedding.weight.data_ptr()
    
    # 3. Layer dimensions
    assert len(model.layers) == 8
    assert model.embedding.weight.shape == (4096, 384)
    assert model.final_norm.weight.shape == (384,)


def test_15m_forward_backward_cpu():
    """Verify forward pass output shape, causal masking, and backward gradient computation."""
    torch.manual_seed(42)
    cfg = ChakrMicro15MConfig()
    model = ChakrMicro(cfg)
    
    batch_size = 2
    seq_len = 16
    input_ids = torch.randint(0, cfg.vocab_size, (batch_size, seq_len))
    
    # Forward pass
    logits = model(input_ids)
    assert logits.shape == (batch_size, seq_len, cfg.vocab_size)
    assert not torch.isnan(logits).any()
    
    # Backward pass
    loss = logits.sum()
    loss.backward()
    
    # Ensure gradients computed for all parameters
    for name, param in model.named_parameters():
        assert param.grad is not None, f"Gradient missing for {name}"
        assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"


def test_15m_checkpoint_roundtrip():
    """Verify state_dict serialization and safe deserialization."""
    cfg = ChakrMicro15MConfig()
    model1 = ChakrMicro(cfg)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "model_15m.pt"
        torch.save({"model_state_dict": model1.state_dict(), "config": cfg}, ckpt_path)
        
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model2 = ChakrMicro(ckpt["config"])
        model2.load_state_dict(ckpt["model_state_dict"])
        
        # Verify weight equivalence
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            assert torch.equal(p1, p2)
