"""
Unit and integration tests for Unified Plan Representation & Span-to-Slot Fusion.
"""
import pytest
import torch

from chakrview.brain.config import ModelConfig
from chakrview.brain.span_grounding import ChakrSpanGroundingModel, ExtractedSpan
from chakrview.cognition.unified_plan_fusion import (
    AbstractPlanStep,
    NeuralSpanSlotFusionPipeline,
    SlotBinding,
    UnifiedPlan,
)


class DummyTokenizer:
    def encode(self, text):
        return [ord(c) % 100 for c in text]

    def decode(self, ids, **kwargs):
        return "".join(chr(i + 65) for i in ids[:4])


def test_unified_plan_materialization():
    steps = [
        AbstractPlanStep(0, "tool_call", "calculator", "expr=@slot0*@slot1", [], ["@slot0", "@slot1"]),
        AbstractPlanStep(1, "tool_call", "calculator", "expr=$0+@slot2", [0], ["@slot2"]),
    ]
    bindings = {
        "@slot0": SlotBinding("@slot0", "a", 0, "expr", ExtractedSpan(0, 1, "NUMBER", 0.9, "600")),
        "@slot1": SlotBinding("@slot1", "b", 0, "expr", ExtractedSpan(2, 3, "NUMBER", 0.9, "5")),
        "@slot2": SlotBinding("@slot2", "c", 1, "expr", ExtractedSpan(4, 5, "NUMBER", 0.9, "14")),
    }
    plan = UnifiedPlan(goal="compound_arithmetic", steps=steps, bindings=bindings)

    dsl, is_valid, err = plan.materialize_dsl()
    assert is_valid, f"Validation failed: {err}"
    assert "expr=600*5" in dsl
    assert "expr=$0+14" in dsl
    assert "<END_PLAN>" in dsl


def test_missing_slot_clarification():
    steps = [
        AbstractPlanStep(0, "tool_call", "calculator", "expr=@slot0*@slot1", [], ["@slot0", "@slot1"]),
    ]
    bindings = {
        "@slot0": SlotBinding("@slot0", "a", 0, "expr", ExtractedSpan(0, 1, "NUMBER", 0.9, "600")),
        "@slot1": SlotBinding("@slot1", "b", 0, "expr", None, is_required=True),
    }
    plan = UnifiedPlan(goal="compound_arithmetic", steps=steps, bindings=bindings)

    dsl, is_valid, err = plan.materialize_dsl()
    assert is_valid
    assert "CLARIFICATION: NEEDED" in dsl
    assert "missing=@slot1" in dsl


def test_fusion_pipeline_e2e_structure():
    cfg = ModelConfig(vocab_size=100, d_model=64, n_layers=2, n_heads=2, max_seq_len=64)
    model = ChakrSpanGroundingModel(cfg, max_slots=4)
    tok = DummyTokenizer()
    pipe = NeuralSpanSlotFusionPipeline(model, tok, torch.device("cpu"))

    prompt = "User: Calculate 600 * 5 and then add 14 to the result\nPlan:\n"
    plan = pipe.predict_plan_topology(prompt)
    assert plan.goal == "compound_arithmetic"
    assert len(plan.steps) == 2
    assert "@slot0" in plan.bindings
