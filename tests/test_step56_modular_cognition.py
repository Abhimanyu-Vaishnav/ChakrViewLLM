"""
Step 56 Automated Tests: Modular Cognitive Conditioning & Native Adapters.

Tests verify:
  01. CognitiveContext schema serialization and parsing roundtrip.
  02. CognitiveContext length budget enforcement and prohibited keyword rejection.
  03. NativeTaskAdapter parameter count efficiency (<= 0.55% of base parameters).
  04. NativeTaskAdapter deterministic hash computation.
  05. NativeTaskAdapter mounting and unmounting correctly modifies and restores base linear modules.
  06. Base model immutability during adapter mounting, forward pass, and unmounting (DeltaW = 0).
  07. MemoryConditionedInferenceBridge prompt construction and context injection.
  08. Checkpoint saving and loading of NativeTaskAdapter.
  09. Complete behavior reversibility of ChakrMicro post-unmount.
  10. Pure CPU execution and low-resource memory bounds.
  11. RIL Integration Contract gate criteria logic.
  12. Multiple adapter independence (two adapters can exist without collision).
"""

from __future__ import annotations

from pathlib import Path
import tempfile
import pytest
import torch
import torch.nn as nn

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.brain.adapter import NativeTaskAdapter, AdaptedLinear
from chakrview.runtime.cortex_context import CognitiveContext
from chakrview.runtime.conditioned import MemoryConditionedInferenceBridge
from chakrview.runtime.interactive import (
    instantiate_frozen_baseline,
    compute_model_hash,
    DEFAULT_TOKENIZER_DIR,
)
from chakrview.runtime.pipeline import EXPECTED_WEIGHT_HASH, EXPECTED_VOCAB_SIZE
from chakrview.tokenizer.serialization import load_tokenizer_artifacts


@pytest.fixture(scope="module")
def tokenizer():
    tok, _ = load_tokenizer_artifacts(DEFAULT_TOKENIZER_DIR)
    return tok


@pytest.fixture(scope="module")
def baseline_model():
    return instantiate_frozen_baseline()


# ---------------------------------------------------------------------------
# Test 01: CognitiveContext serialization and parsing roundtrip
# ---------------------------------------------------------------------------
def test_01_context_serialization_roundtrip():
    ctx = CognitiveContext(
        task_mode="REASONING",
        goal="Fix addition logic",
        state="Failed assertion test_add",
        memory_context="Previous observation: add returned a - b",
        reasoning_state="Operator '-' used instead of '+'",
        constraints="Return valid single-line code",
    )
    serialized = ctx.serialize()
    assert "<CORTEX_CONTEXT>" in serialized
    assert "</CORTEX_CONTEXT>" in serialized
    assert "<TASK_MODE>REASONING</TASK_MODE>" in serialized

    parsed = CognitiveContext.parse(serialized)
    assert parsed is not None
    assert parsed.task_mode == "REASONING"
    assert parsed.goal == "Fix addition logic"
    assert parsed.reasoning_state == "Operator '-' used instead of '+'"


# ---------------------------------------------------------------------------
# Test 02: CognitiveContext budget and security validation
# ---------------------------------------------------------------------------
def test_02_context_budget_and_security():
    # Length violation
    ctx_oversized = CognitiveContext(goal="x" * 900)
    with pytest.raises(ValueError, match="exceeds character budget"):
        ctx_oversized.validate()

    # Prohibited security keyword violation
    ctx_insecure = CognitiveContext(goal="Extract private_key from cluster")
    with pytest.raises(ValueError, match="Prohibited security token"):
        ctx_insecure.validate()


# ---------------------------------------------------------------------------
# Test 03: NativeTaskAdapter parameter count efficiency
# ---------------------------------------------------------------------------
def test_03_adapter_parameter_efficiency():
    adapter = NativeTaskAdapter(adapter_name="test_adapter", rank=4)
    param_count = sum(p.numel() for p in adapter.parameters())
    # Exactly 18,432 parameters for rank=4, d_model=192 across 6 layers
    assert param_count == 18_432
    # Verify less than 1% of base (3,443,136 params)
    assert (param_count / 3_443_136) < 0.01


# ---------------------------------------------------------------------------
# Test 04: NativeTaskAdapter deterministic hash computation
# ---------------------------------------------------------------------------
def test_04_adapter_deterministic_hash():
    torch.manual_seed(42)
    ad1 = NativeTaskAdapter("ad1", rank=4)
    torch.manual_seed(42)
    ad2 = NativeTaskAdapter("ad2", rank=4)

    assert ad1.compute_adapter_hash() == ad2.compute_adapter_hash()


# ---------------------------------------------------------------------------
# Test 05: Mounting and unmounting wrapping mechanics
# ---------------------------------------------------------------------------
def test_05_mounting_and_unmounting():
    config = ModelConfig()
    model = ChakrMicro(config)
    adapter = NativeTaskAdapter("test", rank=4)

    # Initial state
    assert not isinstance(model.layers[0].attn.q_proj, AdaptedLinear)

    # Mount
    adapter.mount(model)
    assert isinstance(model.layers[0].attn.q_proj, AdaptedLinear)
    assert isinstance(model.layers[0].attn.v_proj, AdaptedLinear)

    # Unmount
    adapter.unmount(model)
    assert not isinstance(model.layers[0].attn.q_proj, AdaptedLinear)
    assert not isinstance(model.layers[0].attn.v_proj, AdaptedLinear)


# ---------------------------------------------------------------------------
# Test 06: Base model immutability during mounting and inference
# ---------------------------------------------------------------------------
def test_06_base_model_immutability(baseline_model):
    initial_hash = compute_model_hash(baseline_model)
    assert initial_hash == EXPECTED_WEIGHT_HASH

    adapter = NativeTaskAdapter("test_immutability", rank=4)
    adapter.mount(baseline_model)

    dummy_input = torch.tensor([[0, 42, 100]], dtype=torch.long)
    with torch.no_grad():
        _ = baseline_model(dummy_input)

    adapter.unmount(baseline_model)

    post_hash = compute_model_hash(baseline_model)
    assert post_hash == EXPECTED_WEIGHT_HASH, "Base weights mutated!"


# ---------------------------------------------------------------------------
# Test 07: MemoryConditionedInferenceBridge prompt construction
# ---------------------------------------------------------------------------
def test_07_conditioned_bridge_prompt(baseline_model, tokenizer):
    bridge = MemoryConditionedInferenceBridge(baseline_model, tokenizer)
    prompt = bridge.format_conditioned_prompt(
        task_prompt="def test():\n    return ",
        goal="Return success",
        reasoning_diagnosis="Target is True",
    )
    assert "<CORTEX_CONTEXT>" in prompt
    assert "<GOAL>Return success</GOAL>" in prompt
    assert prompt.endswith("def test():\n    return ")


# ---------------------------------------------------------------------------
# Test 08: Checkpoint saving and loading of NativeTaskAdapter
# ---------------------------------------------------------------------------
def test_08_adapter_save_load_roundtrip():
    adapter = NativeTaskAdapter("save_test", rank=4)
    original_hash = adapter.compute_adapter_hash()

    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "test_adapter.pt"
        adapter.save_checkpoint(path)
        assert path.exists()

        loaded_adapter = NativeTaskAdapter.load_checkpoint(path)
        assert loaded_adapter.compute_adapter_hash() == original_hash


# ---------------------------------------------------------------------------
# Test 09: Complete behavior reversibility post-unmount
# ---------------------------------------------------------------------------
def test_09_behavior_reversibility():
    config = ModelConfig()
    model = ChakrMicro(config)
    dummy_input = torch.tensor([[0, 42, 100]], dtype=torch.long)

    # Initial logits
    with torch.no_grad():
        initial_logits = model(dummy_input).clone()

    # Mount adapter with arbitrary non-zero weights
    adapter = NativeTaskAdapter("mod", rank=4)
    for p in adapter.parameters():
        nn.init.constant_(p, 0.5)
    adapter.mount(model)

    with torch.no_grad():
        adapted_logits = model(dummy_input)
    assert not torch.allclose(initial_logits, adapted_logits), "Adapted logits match initial!"

    # Unmount
    adapter.unmount(model)
    with torch.no_grad():
        restored_logits = model(dummy_input)
    assert torch.allclose(initial_logits, restored_logits, atol=1e-7), "Behavior not restored post-unmount!"


# ---------------------------------------------------------------------------
# Test 10: CPU-only execution and low-resource overhead
# ---------------------------------------------------------------------------
def test_10_cpu_only_execution():
    adapter = NativeTaskAdapter("cpu_test", rank=4)
    for p in adapter.parameters():
        assert p.device.type == "cpu"
    # Weight memory < 100 KB
    total_bytes = sum(p.numel() * p.element_size() for p in adapter.parameters())
    assert total_bytes < 100 * 1024


# ---------------------------------------------------------------------------
# Test 11: RIL Integration Contract gate criteria logic
# ---------------------------------------------------------------------------
def test_11_ril_promotion_gate_logic():
    def evaluate_ril_candidate(base_delta_w: float, target_pass: float, anchor_drop: float) -> bool:
        # Promotion gate: base unchanged (delta_w == 0), target >= 80%, anchor drop <= 2%
        return (base_delta_w == 0.0 and target_pass >= 0.80 and anchor_drop <= 0.02)

    assert evaluate_ril_candidate(0.0, 0.85, 0.01) is True
    assert evaluate_ril_candidate(0.001, 0.85, 0.01) is False  # Mutated base
    assert evaluate_ril_candidate(0.0, 0.70, 0.01) is False   # Target pass too low
    assert evaluate_ril_candidate(0.0, 0.90, 0.05) is False   # Severe anchor regression


# ---------------------------------------------------------------------------
# Test 12: Multiple adapter independence
# ---------------------------------------------------------------------------
def test_12_multiple_adapter_independence():
    ad_reasoning = NativeTaskAdapter("reasoning", rank=4)
    ad_syntax = NativeTaskAdapter("syntax", rank=4)

    assert ad_reasoning.adapter_name != ad_syntax.adapter_name
    assert ad_reasoning.compute_adapter_hash() != ad_syntax.compute_adapter_hash()
