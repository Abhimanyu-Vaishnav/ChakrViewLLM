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


def test_chakr_micro_multi_sequence_smoke_test():
    """Verify forward pass at multiple context lengths: T in {1, 8, 32, 128, 512} with B=1."""
    cfg = ModelConfig()
    model = ChakrMicro(cfg)
    model.eval()
    
    test_lengths = [1, 8, 32, 128, 512]
    with torch.no_grad():
        for t in test_lengths:
            input_ids = torch.randint(0, 4096, (1, t), dtype=torch.long)
            logits = model(input_ids)
            assert logits.shape == (1, t, 4096)
            assert not torch.isnan(logits).any()
            assert not torch.isinf(logits).any()


def test_chakr_micro_serialization_round_trip(tmp_path):
    """
    Verify model serialization:
    1. Save state_dict to disk / buffer
    2. Instantiate fresh model
    3. Load state_dict
    4. Assert bit-exact identical forward pass outputs
    """
    cfg = ModelConfig()
    model1 = ChakrMicro(cfg)
    model1.eval()
    
    input_ids = torch.tensor([[5, 12, 100, 200, 300]], dtype=torch.long)
    with torch.no_grad():
        out1 = model1(input_ids)
        
    save_path = tmp_path / "chakr_micro_test.pt"
    torch.save(model1.state_dict(), save_path)
    
    model2 = ChakrMicro(cfg)
    model2.load_state_dict(torch.load(save_path, weights_only=True))
    model2.eval()
    
    with torch.no_grad():
        out2 = model2(input_ids)
        
    assert torch.allclose(out1, out2, atol=1e-6)
    # Also verify weight tying is maintained after deserialization
    assert model2.lm_head.weight.data_ptr() == model2.embedding.weight.data_ptr()


def test_chakr_micro_backpropagation_gradient_flow():
    """
    Verify that backpropagation computes finite gradients for all trainable parameters
    and that an optimizer step updates weights without NaN/Inf.
    """
    cfg = ModelConfig()
    model = ChakrMicro(cfg)
    model.train()
    
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    input_ids = torch.randint(0, 4096, (2, 8), dtype=torch.long)
    targets = torch.randint(0, 4096, (2, 8), dtype=torch.long)
    
    optimizer.zero_grad()
    logits = model(input_ids)
    loss = torch.nn.functional.cross_entropy(logits.view(-1, 4096), targets.view(-1))
    
    assert torch.isfinite(loss)
    loss.backward()
    
    # Check that all trainable parameters received finite gradients
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"Parameter {name} has None grad"
            assert not torch.isnan(p.grad).any(), f"Parameter {name} has NaN grad"
            assert not torch.isinf(p.grad).any(), f"Parameter {name} has Inf grad"
            
    # Step optimizer and verify finite weights
    optimizer.step()
    for name, p in model.named_parameters():
        assert not torch.isnan(p).any(), f"Parameter {name} has NaN weight after step"
        assert not torch.isinf(p).any(), f"Parameter {name} has Inf weight after step"
