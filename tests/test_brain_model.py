"""
Tests for ChakrMicro complete model (chakrview/brain/model.py):
- Forward pass shape
- Weight tying identity
- Exact parameter accounting
- Determinism
- Sequence length bounds
- Numerical stability (no NaN/Inf)
"""

import pytest
import torch
from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro


def test_chakr_micro_forward_shape_and_numerical_stability():
    """Verify ChakrMicro forward pass [B=2, T=16] -> [2, 16, 4096] with finite values."""
    cfg = ModelConfig()
    model = ChakrMicro(cfg)
    model.eval()
    
    input_ids = torch.randint(0, 4096, (2, 16), dtype=torch.long)
    with torch.no_grad():
        logits = model(input_ids)
        
    assert logits.shape == (2, 16, 4096)
    assert not torch.isnan(logits).any()
    assert not torch.isinf(logits).any()


def test_chakr_micro_weight_tying_identity():
    """Verify that LMHead and TokenEmbedding physically share the identical weight tensor."""
    cfg = ModelConfig()
    model = ChakrMicro(cfg)
    
    # Must share the exact data pointer
    assert model.lm_head.weight.data_ptr() == model.embedding.weight.data_ptr()


def test_chakr_micro_exact_parameter_count():
    """
    Verify programmatic parameter count equals exact analytical expectation:
    Total unique parameters: 3,443,136.
    """
    cfg = ModelConfig()
    model = ChakrMicro(cfg)
    counts = model.count_parameters()
    
    assert counts["embedding"] == 786432
    assert counts["attention_per_layer"] == 147456
    assert counts["attention_total"] == 147456 * 6
    assert counts["ffn_per_layer"] == 294912
    assert counts["ffn_total"] == 294912 * 6
    assert counts["normalization_total"] == (2 * 192 * 6) + 192  # 2,496
    assert counts["total_parameters"] == 3443136
    assert counts["trainable_parameters"] == 3443136


def test_chakr_micro_eval_determinism():
    """Verify identical inputs yield bit-exact identical outputs in eval mode."""
    cfg = ModelConfig()
    model = ChakrMicro(cfg)
    model.eval()
    
    input_ids = torch.tensor([[10, 20, 30, 40], [50, 60, 70, 80]], dtype=torch.long)
    with torch.no_grad():
        out1 = model(input_ids)
        out2 = model(input_ids)
        
    assert torch.equal(out1, out2)


def test_chakr_micro_max_seq_len_exceeded_rejected():
    """Reject sequence length exceeding max_seq_len (512)."""
    cfg = ModelConfig(max_seq_len=512)
    model = ChakrMicro(cfg)
    model.eval()
    
    long_input = torch.randint(0, 4096, (1, 513), dtype=torch.long)
    with pytest.raises(ValueError, match="exceeds maximum context window"):
        model(long_input)
