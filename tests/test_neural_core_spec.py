"""
Tests for ChakrView Neural Core Architecture Specification & Tensor Contracts.

Verifies:
- Config validation and architectural constraints (divisibility, parity, bounds).
- Parameter accounting across multiple layer candidates (N=4, 6, 8, 10, 12).
- Static weight memory footprints across precisions (FP32, FP16, INT8, INT4).
- KV cache and dynamic activation memory scaling for T in {128, 256, 512}.
- FLOPs accounting across sequence lengths.
- Tensor shape contracts for every step of the causal transformer forward pass.
"""

import pytest
from chakrview.config import (
    ChakrConfig,
    calculate_parameter_breakdown,
    calculate_memory_budget,
    calculate_flops_breakdown,
    get_tensor_forward_contracts,
)


def test_default_chakr_micro_config():
    """Verify the default configuration matches frozen Step 4 decisions."""
    cfg = ChakrConfig()
    assert cfg.vocab_size == 4096
    assert cfg.d_model == 192
    assert cfg.num_layers == 6
    assert cfg.num_heads == 6
    assert cfg.num_kv_heads == 6
    assert cfg.head_dim == 32
    assert cfg.d_ff == 512
    assert cfg.max_seq_len == 512
    assert cfg.rope_theta == 10000.0
    assert cfg.rope_dim == 32
    assert cfg.rms_norm_eps == 1e-5
    assert cfg.bias is False
    assert cfg.tie_weights is True
    assert cfg.bos_token_id == 0
    assert cfg.eos_token_id == 1
    assert cfg.pad_token_id == 2


def test_config_divisibility_validation():
    """Reject configurations where d_model is not divisible by num_heads."""
    with pytest.raises(ValueError, match="must be divisible"):
        ChakrConfig(d_model=192, num_heads=5)


def test_config_simple_mha_validation():
    """Reject GQA configurations in v0.1 (num_kv_heads must equal num_heads)."""
    with pytest.raises(ValueError, match="Simple MHA"):
        ChakrConfig(num_heads=6, num_kv_heads=2)


def test_config_rope_parity_validation():
    """Reject odd head_dim or odd rope_dim."""
    # d_model=190, num_heads=10, num_kv_heads=10 -> head_dim=19 (odd)
    with pytest.raises(ValueError, match="must be even"):
        ChakrConfig(d_model=190, num_heads=10, num_kv_heads=10, rope_dim=19)


def test_config_bounds_validation():
    """Reject negative or zero dimensions."""
    with pytest.raises(ValueError):
        ChakrConfig(vocab_size=0)
    with pytest.raises(ValueError):
        ChakrConfig(d_model=-192)
    with pytest.raises(ValueError):
        ChakrConfig(num_layers=0)
    with pytest.raises(ValueError):
        ChakrConfig(d_ff=0)


def test_chakr_micro_exact_parameter_count():
    """
    Verify exact parameter accounting for Chakr-Micro v0.1:
    - Embedding: 4096 * 192 = 786,432
    - Per layer:
      - W_q, W_k, W_v, W_o: 4 * 192 * 192 = 147,456
      - RMSNorm 1: 192
      - W_gate, W_up, W_down: 3 * 192 * 512 = 294,912
      - RMSNorm 2: 192
      - Layer total: 147,456 + 192 + 294,912 + 192 = 442,752
    - 6 Layers: 6 * 442,752 = 2,656,512
    - Final RMSNorm: 192
    - LM Head (tied): 0
    - Total: 786,432 + 2,656,512 + 192 = 3,443,136
    """
    cfg = ChakrConfig()
    breakdown = calculate_parameter_breakdown(cfg)
    
    assert breakdown["embedding_params"] == 786432
    assert breakdown["per_layer"]["attn_total"] == 147456
    assert breakdown["per_layer"]["ffn_total"] == 294912
    assert breakdown["per_layer"]["rmsnorm_1"] == 192
    assert breakdown["per_layer"]["rmsnorm_2"] == 192
    assert breakdown["per_layer"]["total_layer_params"] == 442752
    assert breakdown["all_layers_params"] == 2656512
    assert breakdown["final_norm_params"] == 192
    assert breakdown["lm_head_params"] == 0
    assert breakdown["total_parameters"] == 3443136


def test_parameter_counts_across_layer_candidates():
    """Evaluate candidate layer counts N in {4, 6, 8, 10, 12}."""
    expected_totals = {
        4: 2557632,
        6: 3443136,
        8: 4328640,
        10: 5214144,
        12: 6099648,
    }
    for n_layers, expected_params in expected_totals.items():
        cfg = ChakrConfig(num_layers=n_layers)
        breakdown = calculate_parameter_breakdown(cfg)
        assert breakdown["total_parameters"] == expected_params
        assert breakdown["per_layer"]["total_layer_params"] == 442752


def test_static_weight_memory_footprint():
    """Verify static weight memory calculations in bytes and MiB."""
    cfg = ChakrConfig()
    breakdown = calculate_parameter_breakdown(cfg)
    mem = breakdown["static_memory"]
    
    assert mem["fp32_bytes"] == 3443136 * 4
    assert round(mem["fp32_mib"], 2) == 13.13
    assert mem["fp16_bytes"] == 3443136 * 2
    assert round(mem["fp16_mib"], 2) == 6.57
    assert mem["int8_bytes"] == 3443136 * 1
    assert round(mem["int8_mib"], 2) == 3.28
    assert mem["int4_bytes"] == 3443136 // 2
    assert round(mem["int4_mib"], 2) == 1.64


def test_kv_cache_scaling():
    """
    Verify KV cache elements:
    Total elements = 2 * B * N * T * d_model = 2 * 1 * 6 * T * 192 = 2304 * T
    """
    cfg = ChakrConfig()
    
    # T = 128
    b128 = calculate_memory_budget(cfg, batch_size=1, seq_len=128)
    assert b128["kv_cache"]["total_elements"] == 2304 * 128
    assert b128["kv_cache"]["fp32_bytes"] == 2304 * 128 * 4  # 1,179,648 bytes (1.125 MiB)
    assert b128["kv_cache"]["fp16_bytes"] == 2304 * 128 * 2  # 589,824 bytes (0.5625 MiB)
    assert b128["kv_cache"]["int8_bytes"] == 2304 * 128 * 1  # 294,912 bytes
    
    # T = 256
    b256 = calculate_memory_budget(cfg, batch_size=1, seq_len=256)
    assert b256["kv_cache"]["total_elements"] == 2304 * 256
    assert b256["kv_cache"]["fp32_bytes"] == 2304 * 256 * 4  # 2.25 MiB
    
    # T = 512
    b512 = calculate_memory_budget(cfg, batch_size=1, seq_len=512)
    assert b512["kv_cache"]["total_elements"] == 2304 * 512
    assert b512["kv_cache"]["fp32_bytes"] == 2304 * 512 * 4  # 4.50 MiB


def test_memory_budget_sequence_length_exceeded():
    """Reject memory budget queries where seq_len exceeds max_seq_len."""
    cfg = ChakrConfig(max_seq_len=512)
    with pytest.raises(ValueError, match="exceeds configured max_seq_len"):
        calculate_memory_budget(cfg, seq_len=513)


def test_flops_calculation_scaling():
    """Verify FLOP calculation scaling across sequence lengths."""
    cfg = ChakrConfig()
    
    f128 = calculate_flops_breakdown(cfg, seq_len=128)
    f256 = calculate_flops_breakdown(cfg, seq_len=256)
    f512 = calculate_flops_breakdown(cfg, seq_len=512)
    
    assert f128["total_mflops"] < f256["total_mflops"] < f512["total_mflops"]
    # Per-token FLOPs slightly increase with T due to O(T^2) attention scaling
    assert f128["flops_per_token"] < f256["flops_per_token"] < f512["flops_per_token"]
    
    # At T=512, total FLOPs should be ~4.77 GFLOPs
    assert 4700.0 <= f512["total_mflops"] <= 4850.0


def test_tensor_forward_contracts():
    """Verify all intermediate tensor shapes in the forward pipeline."""
    cfg = ChakrConfig()
    contracts = get_tensor_forward_contracts(cfg, batch_size=2, seq_len=256)
    
    assert contracts["input_ids"] == (2, 256)
    assert contracts["attention_mask"] == (2, 256)
    assert contracts["embedded_tokens"] == (2, 256, 192)
    assert contracts["layer_norm_1_output"] == (2, 256, 192)
    assert contracts["q_heads_rotated"] == (2, 6, 256, 32)
    assert contracts["k_heads_rotated"] == (2, 6, 256, 32)
    assert contracts["v_heads"] == (2, 6, 256, 32)
    assert contracts["causal_mask"] == (1, 1, 256, 256)
    assert contracts["attention_probabilities"] == (2, 6, 256, 256)
    assert contracts["attention_output_projected"] == (2, 256, 192)
    assert contracts["residual_after_attention"] == (2, 256, 192)
    assert contracts["layer_norm_2_output"] == (2, 256, 192)
    assert contracts["swiglu_gate_proj"] == (2, 256, 512)
    assert contracts["swiglu_down_proj"] == (2, 256, 192)
    assert contracts["layer_block_output"] == (2, 256, 192)
    assert contracts["final_norm_output"] == (2, 256, 192)
    assert contracts["output_logits"] == (2, 256, 4096)
