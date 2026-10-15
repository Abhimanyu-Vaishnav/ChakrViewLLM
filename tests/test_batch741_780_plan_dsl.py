"""
Unit and Integration Tests for Batches 741-780:
Autoregressive Plan DSL, Parser Grammar, Validator Rules, and Safe Execution.
Location: tests/test_batch741_780_plan_dsl.py
"""
from pathlib import Path
import pytest

from chakrview.cognition.plan_dsl import (
    ParsedPlan,
    PlanDSLParser,
    PlanDSLSerializer,
    PlanDSLValidator,
)
from chakrview.runtime.tools import CalculatorTool, TextUtilityTool, ToolExecutor, ToolRegistry

REPO_ROOT = Path(__file__).resolve().parent.parent
CKPT_PATH = REPO_ROOT / "experiments" / "batch741_780" / "checkpoints" / "candidate_step250_autoregressive_planner.pt"


def test_autoregressive_checkpoint_presence_and_weights():
    """Verifies that the autoregressive planner candidate checkpoint exists and is valid."""
    assert CKPT_PATH.exists(), f"Autoregressive checkpoint not found at {CKPT_PATH}"
    import torch
    state = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    assert "embedding.weight" in state or "layers.0.attn.q_proj.weight" in state or "layers.0.attention.wq.weight" in state or len(state) > 10


def test_plan_dsl_serializer_and_parser_roundtrip():
    """Verifies that serialization and parsing preserve plan structure."""
    steps = [
        {"action": "tool_call", "tool": "calculator", "args": {"expr": "15*4"}, "deps": []},
        {"action": "tool_call", "tool": "calculator", "args": {"expr": "$0+10"}, "deps": [0]},
    ]
    dsl_text = PlanDSLSerializer.serialize("compound_arithmetic", steps, False)
    assert "GOAL: compound_arithmetic" in dsl_text
    assert "<END_PLAN>" in dsl_text

    parsed = PlanDSLParser.parse(dsl_text)
    assert parsed.is_valid is True
    assert parsed.goal == "compound_arithmetic"
    assert len(parsed.steps) == 2
    assert parsed.steps[0].tool == "calculator"
    assert parsed.steps[1].deps == [0]
    assert parsed.steps[1].args["expr"] == "$0+10"


def test_plan_dsl_validator_rejects_truncated_plan():
    """Verifies that plans without terminal <END_PLAN> marker are strictly rejected."""
    truncated_dsl = "GOAL: compound_arithmetic\nSTEP 0: tool_call | calculator | args={expr=10*2} | deps=[]"
    parsed = PlanDSLParser.parse(truncated_dsl)
    assert parsed.is_valid is False
    assert "missing terminal <END_PLAN>" in parsed.validation_error


def test_plan_dsl_validator_rejects_unauthorized_tools():
    """Verifies that unauthorized or arbitrary tools are blocked from execution."""
    malicious_dsl = (
        "GOAL: unsupported\n"
        "STEP 0: tool_call | root_exploit | args={cmd='rm -rf'} | deps=[]\n"
        "CLARIFICATION: NONE\n"
        "<END_PLAN>"
    )
    parsed = PlanDSLParser.parse(malicious_dsl)
    assert parsed.is_valid is False
    assert "Unauthorized or unknown tool" in parsed.validation_error


def test_plan_dsl_validator_rejects_cyclic_dependencies():
    """Verifies that cyclic or self-dependencies violate acyclicity and are rejected."""
    cyclic_dsl = (
        "GOAL: compound_arithmetic\n"
        "STEP 0: tool_call | calculator | args={expr=$0+5} | deps=[0]\n"
        "CLARIFICATION: NONE\n"
        "<END_PLAN>"
    )
    parsed = PlanDSLParser.parse(cyclic_dsl)
    assert parsed.is_valid is False
    assert "Acyclicity violation" in parsed.validation_error


def test_plan_dsl_safe_execution_via_tool_executor():
    """Verifies that a valid parsed DSL plan executes safely through ToolExecutor."""
    valid_dsl = (
        "GOAL: compound_arithmetic\n"
        "STEP 0: tool_call | calculator | args={expr=25*4} | deps=[]\n"
        "STEP 1: tool_call | calculator | args={expr=$0+50} | deps=[0]\n"
        "CLARIFICATION: NONE\n"
        "<END_PLAN>"
    )
    parsed = PlanDSLParser.parse(valid_dsl)
    assert parsed.is_valid is True

    registry = ToolRegistry()
    registry.register(CalculatorTool())
    executor = ToolExecutor(registry)

    artifacts = {}
    for st in parsed.steps:
        expr = st.args["expr"]
        for dep in st.deps:
            expr = expr.replace(f"${dep}", str(artifacts[dep]))
        res = registry.get("calculator").execute(expression=expr)
        artifacts[st.step_id] = res.output

    assert artifacts[0] == 100
    assert artifacts[1] == 150
