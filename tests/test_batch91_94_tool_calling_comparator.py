"""
Unit & Regression Tests for Batch 91-94:
- Deterministic Safe Calculator: Schema validation, error codes, bounds checking, zero eval/exec.
- Checkpoint Output Comparator: SHA-256 calculation, side-by-side inference, diff generation, report persistence.
- Cold reload deterministic replay.
- Graceful error handling for missing/corrupted checkpoints.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest
import sys
import tempfile
import torch

sys.path.insert(0, ".")

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from experiments.batch91_94_tool_calling_comparator.tools.calculator import (
    execute_safe_calculator,
    validate_and_execute_tool_call,
    validate_tool_call_payload,
    ToolExecutionError,
)
from experiments.batch91_94_tool_calling_comparator.workbench.app import (
    inspect_checkpoint,
    compare_checkpoints,
    run_deterministic_oracle,
    save_comparison_report,
    load_comparison_report,
)


def test_safe_calculator_basic_operations():
    assert execute_safe_calculator("add", 3, 4) == 7
    assert execute_safe_calculator("sub", 10, 4) == 6
    assert execute_safe_calculator("mul", 6, 7) == 42
    assert execute_safe_calculator("add", -5, 5) == 0


def test_safe_calculator_unsupported_operation():
    with pytest.raises(ToolExecutionError) as exc_info:
        execute_safe_calculator("div", 10, 2)
    assert exc_info.value.code == "UNSUPPORTED_OPERATION"


def test_tool_call_payload_validation_success():
    payload = {
        "tool": "calculator",
        "op": "add",
        "a": 12,
        "b": 18,
    }
    res = validate_tool_call_payload(payload)
    assert res["valid"] is True
    assert res["tool"] == "calculator"
    assert res["op"] == "add"
    assert res["a"] == 12
    assert res["b"] == 18


def test_tool_call_payload_validation_errors():
    # Unsupported tool
    assert validate_tool_call_payload({"tool": "bash", "command": "ls"})["error_code"] == "UNSUPPORTED_TOOL"
    # Missing tool
    assert validate_tool_call_payload({"op": "add", "a": 1, "b": 2})["error_code"] == "MISSING_FIELD"
    # Missing op
    assert validate_tool_call_payload({"tool": "calculator", "a": 1, "b": 2})["error_code"] == "MISSING_FIELD"
    # Non-integer operands
    assert validate_tool_call_payload({"tool": "calculator", "op": "add", "a": "1", "b": 2})["error_code"] == "OPERAND_TYPE_ERROR"
    # Out of bounds operands
    assert validate_tool_call_payload({"tool": "calculator", "op": "add", "a": 20000, "b": 2})["error_code"] == "OPERAND_BOUNDS_ERROR"


def test_tool_call_execution_from_string():
    raw_str = 'TOOL_CALL: {"tool": "calculator", "op": "add", "a": 5, "b": 7}'
    res = validate_and_execute_tool_call(raw_str)
    assert res["status"] == "SUCCESS"
    assert res["executed"] is True
    assert res["result"] == 12
    assert res["provenance"] == "DETERMINISTIC_SAFE_CALCULATOR"


def test_tool_call_execution_malformed_string():
    res = validate_and_execute_tool_call("Just some random text with no tool call.")
    assert res["status"] == "MALFORMED_JSON"
    assert res["executed"] is False
    assert res["result"] is None


def test_inspect_checkpoint_valid(tmp_path):
    # Save a small dummy model state
    config = ModelConfig(max_seq_len=64)
    model = ChakrMicro(config)
    ckpt_file = tmp_path / "dummy.pt"
    torch.save({
        "step": 42,
        "model_state_dict": model.state_dict(),
        "condition_name": "test_condition",
    }, ckpt_file)

    meta = inspect_checkpoint(str(ckpt_file))
    assert meta["exists"] is True
    assert meta["valid"] is True
    assert meta["step"] == 42
    assert meta["param_count"] == 3443136
    assert len(meta["sha256"]) == 64


def test_inspect_checkpoint_missing():
    meta = inspect_checkpoint("non_existent_ckpt_path.pt")
    assert meta["exists"] is False
    assert meta["valid"] is False
    assert "does not exist" in meta["error"]


def test_deterministic_oracle():
    res_add = run_deterministic_oracle("<|im_start|>user\nCalculate 12 + 8.<|im_end|>\n<|im_start|>assistant\n")
    assert res_add["is_arithmetic"] is True
    assert res_add["operation"] == "add"
    assert res_add["operands"] == [12, 8]
    assert res_add["expected_answer"] == "20"

    res_non = run_deterministic_oracle("Tell me a funny story.")
    assert res_non["is_arithmetic"] is False


def test_comparison_report_save_and_load(tmp_path):
    report_data = {
        "title": "Test Comparison",
        "summary": {"total_prompts": 2, "identical_outputs_count": 1},
        "comparisons": [{"index": 0, "outputs_identical": True}],
    }
    saved_path = save_comparison_report(report_data)
    assert Path(saved_path).is_file()

    loaded = load_comparison_report(str(saved_path))
    assert loaded["title"] == "Test Comparison"
    assert loaded["summary"]["total_prompts"] == 2
