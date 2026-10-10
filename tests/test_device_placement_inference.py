"""
Focused Device Placement, CUDA/CPU Consistency, and KV-Cache Tests.

Verifies:
1. Model parameters, KV-cache, and inputs reside on compatible devices.
2. Prefill and autoregressive decode operate without RuntimeError on CPU.
3. Prefill and autoregressive decode operate on CUDA when available, skipping gracefully when unavailable.
4. CPU fallback works cleanly when an invalid CUDA device is requested.
5. Model weights remain bit-exact (ΔW = 0) before and after inference.
"""

from __future__ import annotations

import pytest
import torch

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.brain.cache import KVCache, LayerKVCache, KVCacheDeviceMismatchError
from chakrview.runtime.hardware import ModelExecutionPlan, PrecisionType
from chakrview.runtime.inference import GenerationConfig, InferenceSession
from chakrview.runtime.local_runtime import LocalModelRuntime
from chakrview.runtime.pipeline import InferenceEngine, InferenceRequest
from chakrview.tokenizer.tokenizer import BPETokenizer


def test_cpu_inference_device_consistency():
    """Verify that on CPU, all model parameters, KV cache, and input tensors align on CPU."""
    plan = ModelExecutionPlan(
        device="cpu",
        precision=PrecisionType.FP32,
        max_context_len=256,
        batch_size=1,
        thread_count=1,
    )
    engine = InferenceEngine(execution_plan=plan)

    param_device = next(engine.model.parameters()).device
    assert param_device.type == "cpu"
    assert engine.device.type == "cpu"
    assert engine.kv_cache.device.type == "cpu"

    req = InferenceRequest(
        prompt="ChakrView native CPU device verification",
        generation_config=GenerationConfig(max_new_tokens=8),
    )
    res = engine.execute(req)
    assert res.output_token_count == 8
    assert res.weight_hash_verified is True


def test_kv_cache_devices_compatible_helper():
    """Test LayerKVCache._devices_compatible handles cuda vs cuda:0 and rejects cpu vs cuda."""
    assert LayerKVCache._devices_compatible(torch.device("cpu"), torch.device("cpu"))
    assert not LayerKVCache._devices_compatible(torch.device("cpu"), torch.device("cuda"))
    assert not LayerKVCache._devices_compatible(torch.device("cuda:0"), torch.device("cpu"))
    assert LayerKVCache._devices_compatible(torch.device("cuda"), torch.device("cuda:0"))
    assert LayerKVCache._devices_compatible(torch.device("cuda:0"), torch.device("cuda:0"))


def test_cuda_inference_device_consistency():
    """Verify CUDA execution when CUDA is available; skip cleanly if no CUDA device is present."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA device unavailable in current environment")

    plan = ModelExecutionPlan(
        device="cuda",
        precision=PrecisionType.FP32,
        max_context_len=256,
        batch_size=1,
        thread_count=2,
    )
    engine = InferenceEngine(execution_plan=plan)

    param_device = next(engine.model.parameters()).device
    assert param_device.type == "cuda"
    assert engine.device.type == "cuda"

    req = InferenceRequest(
        prompt="ChakrView CUDA inference verification",
        generation_config=GenerationConfig(max_new_tokens=16),
    )
    res = engine.execute(req)
    assert res.output_token_count == 16
    assert res.weight_hash_verified is True


def test_cuda_prefill_and_decode_kv_cache_integration():
    """Verify prefill and single-token decode with KV cache on CUDA."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA device unavailable in current environment")

    dev = torch.device("cuda")
    model = ChakrMicro(ModelConfig())
    model.to(dev)

    cache = KVCache(num_layers=model.config.n_layers, max_seq_len=256, device=dev)
    prompt = torch.tensor([[10, 20, 30]], dtype=torch.long, device=dev)

    logits, _ = model.prefill(prompt, kv_cache=cache)
    assert logits.shape == (1, 3, 4096)
    assert logits.device.type == "cuda"
    assert cache.sequence_length == 3

    step_tok = torch.tensor([[40]], dtype=torch.long, device=dev)
    step_logits = model.decode_next(step_tok, kv_cache=cache)
    assert step_logits.shape == (1, 1, 4096)
    assert step_logits.device.type == "cuda"
    assert cache.sequence_length == 4


def test_cpu_fallback_on_invalid_cuda_device():
    """Verify honest fallback to CPU when CUDA device initialization fails."""
    plan = ModelExecutionPlan(
        device="cuda:9999",  # Non-existent CUDA index
        precision=PrecisionType.FP32,
        max_context_len=128,
        batch_size=1,
        thread_count=1,
    )
    engine = InferenceEngine(execution_plan=plan)
    assert engine.device.type == "cpu"
    assert next(engine.model.parameters()).device.type == "cpu"

    req = InferenceRequest(
        prompt="Testing fallback execution",
        generation_config=GenerationConfig(max_new_tokens=4),
    )
    res = engine.execute(req)
    assert res.output_token_count == 4
    assert res.weight_hash_verified is True


def test_weight_immutability_preserved_across_devices():
    """Verify that moving model across devices does not mutate parameter weights (ΔW = 0)."""
    torch.manual_seed(42)
    model = ChakrMicro(ModelConfig())
    engine = InferenceEngine(model=model)
    initial_hash = engine.compute_weight_hash()

    if torch.cuda.is_available():
        engine.model.to("cuda")
        cuda_hash = engine.compute_weight_hash()
        assert cuda_hash == initial_hash, "Weight hash mutated after moving to CUDA"

        engine.model.to("cpu")
        cpu_hash = engine.compute_weight_hash()
        assert cpu_hash == initial_hash, "Weight hash mutated after moving back to CPU"
    else:
        assert initial_hash == engine.expected_weight_hash


def test_inference_session_cuda_support():
    """Verify InferenceSession in inference.py respects device placement."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA device unavailable in current environment")

    plan = ModelExecutionPlan(
        device="cuda",
        precision=PrecisionType.FP32,
        max_context_len=256,
        batch_size=1,
        thread_count=2,
    )
    model = ChakrMicro(ModelConfig())
    sess = InferenceSession(model=model, tokenizer=BPETokenizer(), execution_plan=plan)

    assert sess.device.type == "cuda"
    assert next(sess.model.parameters()).device.type == "cuda"

    tokens, latency = sess.prefill("Testing session prefill")
    assert len(tokens) > 0
    assert latency > 0.0
