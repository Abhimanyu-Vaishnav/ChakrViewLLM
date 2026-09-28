"""
Unit tests for Step 10 KV-Cache Engine (chakrview.brain.cache).
"""

import pytest
import torch
from chakrview.brain.cache import (
    KVCache,
    LayerKVCache,
    KVCacheError,
    KVCacheOverflowError,
    KVCacheDeviceMismatchError,
    KVCacheDtypeMismatchError,
)


def test_kv_cache_initialization():
    cache = KVCache(num_layers=6, max_seq_len=512, device="cpu", dtype=torch.float32)
    assert cache.num_layers == 6
    assert cache.max_sequence_length == 512
    assert cache.sequence_length == 0
    assert cache.total_memory_bytes == 0

    with pytest.raises(ValueError, match="num_layers must be positive"):
        KVCache(num_layers=0)

    with pytest.raises(ValueError, match="max_seq_len must be in"):
        KVCache(max_seq_len=1024)


def test_kv_cache_append_and_get():
    cache = KVCache(num_layers=2, max_seq_len=128)
    B, H, T, D = 1, 6, 4, 32

    k0 = torch.randn(B, H, T, D)
    v0 = torch.randn(B, H, T, D)

    k_out, v_out = cache.append(0, k0, v0)
    assert cache.get_seq_len(0) == 4
    assert cache.sequence_length == 4
    assert k_out.shape == (B, H, 4, D)
    assert v_out.shape == (B, H, 4, D)
    assert cache.total_memory_bytes > 0

    # Append 1 more token
    k1 = torch.randn(B, H, 1, D)
    v1 = torch.randn(B, H, 1, D)
    cache.append(0, k1, v1)
    assert cache.get_seq_len(0) == 5

    k_ret, v_ret = cache.get(0)
    assert k_ret.shape == (B, H, 5, D)
    assert v_ret.shape == (B, H, 5, D)


def test_kv_cache_reset():
    cache = KVCache(num_layers=2, max_seq_len=128)
    k = torch.randn(1, 6, 8, 32)
    v = torch.randn(1, 6, 8, 32)

    cache.append(0, k, v)
    cache.append(1, k, v)
    assert cache.sequence_length == 8

    cache.reset()
    assert cache.sequence_length == 0
    assert cache.total_memory_bytes == 0
    k_ret, v_ret = cache.get(0)
    assert k_ret is None and v_ret is None


def test_kv_cache_truncation():
    cache = KVCache(num_layers=2, max_seq_len=128)
    k = torch.randn(1, 6, 16, 32)
    v = torch.randn(1, 6, 16, 32)
    cache.append(0, k, v)
    assert cache.sequence_length == 16

    # Truncate to 10
    cache.truncate(10)
    assert cache.sequence_length == 10
    k_ret, _ = cache.get(0)
    assert k_ret.shape == (1, 6, 10, 32)

    # Truncate to 0 resets
    cache.truncate(0)
    assert cache.sequence_length == 0

    with pytest.raises(ValueError):
        cache.truncate(-1)


def test_kv_cache_overflow():
    cache = KVCache(num_layers=1, max_seq_len=10)
    k = torch.randn(1, 6, 8, 32)
    v = torch.randn(1, 6, 8, 32)
    cache.append(0, k, v)

    # Appending 4 more tokens would exceed max_seq_len=10 (8 + 4 = 12)
    k_overflow = torch.randn(1, 6, 4, 32)
    v_overflow = torch.randn(1, 6, 4, 32)

    with pytest.raises(KVCacheOverflowError, match="exceeds max context"):
        cache.append(0, k_overflow, v_overflow)


def test_kv_cache_device_and_dtype_mismatch():
    cache = KVCache(num_layers=1, max_seq_len=64, dtype=torch.float32)
    k = torch.randn(1, 6, 2, 32, dtype=torch.float64)  # Wrong dtype
    v = torch.randn(1, 6, 2, 32, dtype=torch.float64)

    with pytest.raises(KVCacheDtypeMismatchError):
        cache.append(0, k, v)


def test_kv_cache_shape_validation():
    cache = KVCache(num_layers=1, max_seq_len=64)
    # Non-4D tensor
    k_bad = torch.randn(1, 6, 32)
    v_bad = torch.randn(1, 6, 32)
    with pytest.raises(ValueError, match="must be 4D"):
        cache.append(0, k_bad, v_bad)

    # Mismatched key vs value shape
    k_ok = torch.randn(1, 6, 2, 32)
    v_mismatch = torch.randn(1, 6, 3, 32)
    with pytest.raises(ValueError, match="does not match Value shape"):
        cache.append(0, k_ok, v_mismatch)
