"""
Regression tests for Batches 115-118:
Tool-Gating Calibration, Strict Execution Metrics, and Reliable Argument Binding.
"""
from pathlib import Path
import json
import pytest

from experiments.batch103_106_slot_weighted_sft.slot_weighted_training import (
    BASE_CKPT, compute_file_sha256
)
from experiments.batch115_118_tool_gating_curriculum.metrics import (
    parse_tool_grammar,
    validate_schema,
    evaluate_single_prediction,
    compute_aggregate_metrics,
)

def test_canonical_checkpoint_hash_invariant_b118():
    expected_sha = "96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee"
    actual_sha = compute_file_sha256(str(BASE_CKPT))
    assert actual_sha == expected_sha

def test_balanced_curriculum_checkpoints_exist_and_hashes_stable():
    ckpt_dir = Path("experiments/batch115_118_tool_gating_curriculum/checkpoints")
    ctrl_ckpt = ckpt_dir / "balanced_control_base_step150.pt"
    bridge_ckpt = ckpt_dir / "balanced_curriculum_bridge_step150.pt"
    assert ctrl_ckpt.exists()
    assert bridge_ckpt.exists()
    sha_ctrl = compute_file_sha256(str(ctrl_ckpt))
    sha_bridge = compute_file_sha256(str(bridge_ckpt))
    assert sha_ctrl == "6905fa8b759f3730926665395e54453a779fb0767fb574e115163b3fc512a427"
    assert sha_bridge == "ff606e5a8c25272347d1e127005255080eb82192a7b2d6eb8136e9a13134a2f7"

def test_balanced_curriculum_dataset_counts():
    data_dir = Path("experiments/batch115_118_tool_gating_curriculum/data")
    with open(data_dir / "manifest.json", "r", encoding="utf-8") as f:
        m = json.load(f)
    assert m["counts"]["train"]["tool_examples"] == 64
    assert m["counts"]["train"]["non_tool_controls"] == 16
    assert m["counts"]["val"]["tool_examples"] == 16
    assert m["counts"]["val"]["non_tool_controls"] == 4
    assert m["counts"]["test"]["tool_examples"] == 20
    assert m["counts"]["test"]["non_tool_controls"] == 8

def test_evaluation_metric_matrix_synthetic():
    # 1. Perfectly correct tool call
    rec_tool = {"requires_tool": True, "operands": [4, 7]}
    pred_ok = "TOOL_CALL: {\"tool\": \"calculator\", \"op\": \"add\", \"a\": 4, \"b\": 7}<|im_end|>"
    e_ok = evaluate_single_prediction(pred_ok, rec_tool)
    assert e_ok["e2e_correct"] is True
    
    # 2. Syntax ok, but operands swapped: (7, 4) instead of (4, 7)
    pred_swap = "TOOL_CALL: {\"tool\": \"calculator\", \"op\": \"add\", \"a\": 7, \"b\": 4}<|im_end|>"
    e_swap = evaluate_single_prediction(pred_swap, rec_tool)
    assert e_swap["a_correct"] is False
    assert e_swap["exact_call"] is False
    assert e_swap["e2e_correct"] is False
    
    # 3. Clean refusal on non-tool prompt
    rec_nontool = {"requires_tool": False}
    pred_refuse = "I am a calculator assistant and only support arithmetic calculations.<|im_end|>"
    e_refuse = evaluate_single_prediction(pred_refuse, rec_nontool)
    assert e_refuse["false_positive"] is False
    assert e_refuse["correct"] is True
    
    # 4. Spurious tool call on non-tool prompt
    pred_spurious = "TOOL_CALL: {\"tool\": \"calculator\", \"op\": \"add\", \"a\": 0, \"b\": 0}<|im_end|>"
    e_spurious = evaluate_single_prediction(pred_spurious, rec_nontool)
    assert e_spurious["false_positive"] is True
    assert e_spurious["correct"] is False
