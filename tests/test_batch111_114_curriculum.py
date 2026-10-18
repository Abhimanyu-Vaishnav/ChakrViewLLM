"""
Regression tests for Batches 111-114:
Curriculum Transfer, Checkpoint Invariants, and Allowlisted Calculator Safety.
"""
from pathlib import Path
import json
import pytest

from experiments.batch103_106_slot_weighted_sft.slot_weighted_training import (
    BASE_CKPT, TOKENIZER_DIR, load_tokenizer_artifacts, compute_file_sha256
)
from experiments.batch111_114_curriculum_transfer.data.generate_curriculum_data import DATA_DIR
from experiments.batch111_114_curriculum_transfer.train_and_eval_curriculum import parse_tool_call
from experiments.batch91_94_tool_calling_comparator.tools.calculator import validate_and_execute_tool_call

def test_canonical_checkpoint_hash_invariant_b114():
    expected_sha = "96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee"
    actual_sha = compute_file_sha256(str(BASE_CKPT))
    assert actual_sha == expected_sha, f"Base checkpoint modified! {actual_sha} != {expected_sha}"

def test_curriculum_checkpoints_exist_and_hashes_stable():
    ckpt_dir = Path("experiments/batch111_114_curriculum_transfer/checkpoints")
    direct_ckpt = ckpt_dir / "direct_tool_control_step150.pt"
    curric_ckpt = ckpt_dir / "curriculum_bridge_transfer_step150.pt"
    assert direct_ckpt.exists(), "Direct control checkpoint missing"
    assert curric_ckpt.exists(), "Curriculum checkpoint missing"
    sha_direct = compute_file_sha256(str(direct_ckpt))
    sha_curric = compute_file_sha256(str(curric_ckpt))
    assert sha_direct == "b4b489fca356b5f12e19eb2b88d7d25a418bb9af543a5f7368f0e13a6f093521"
    assert sha_curric == "67c857666f082aa1057c0c8aef47959f68d661556ba522d4febdcbe5eff8c4a5"

def test_tool_call_parser_invariants():
    valid_raw = "TOOL_CALL: {\"tool\": \"calculator\", \"op\": \"add\", \"a\": 4, \"b\": 7}<|im_end|>"
    parsed = parse_tool_call(valid_raw)
    assert parsed == {"tool": "calculator", "op": "add", "a": 4, "b": 7}
    
    # Incomplete or non-tool call
    assert parse_tool_call("Just normal text") is None
    assert parse_tool_call("TOOL_CALL: broken json") is None

def test_allowlisted_calculator_safety():
    # Valid call
    v_res = validate_and_execute_tool_call("TOOL_CALL: {\"tool\": \"calculator\", \"op\": \"add\", \"a\": 4, \"b\": 7}")
    assert v_res["executed"] is True
    assert v_res["result"] == 11
    
    # Prohibited shell / arbitrary execution injection
    malicious = "TOOL_CALL: {\"tool\": \"os\", \"op\": \"system\", \"cmd\": \"dir\"}"
    m_res = validate_and_execute_tool_call(malicious)
    assert m_res["executed"] is False
    assert m_res["status"] == "UNSUPPORTED_TOOL"

def test_curriculum_split_isolation():
    p = DATA_DIR
    def load(path):
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    train_recs = load(p / "train.jsonl")
    val_recs = load(p / "val.jsonl")
    test_recs = load(p / "test.jsonl")
    
    assert len(train_recs) == 64
    assert len(val_recs) == 16
    assert len(test_recs) == 20
    
    train_pairs = set(tuple(r["operands"]) for r in train_recs)
    val_pairs = set(tuple(r["operands"]) for r in val_recs)
    test_pairs = set(tuple(r["operands"]) for r in test_recs)
    
    assert len(train_pairs & test_pairs) == 0, "Train-test leakage detected"
    assert len(val_pairs & test_pairs) == 0, "Val-test leakage detected"
    assert len(train_pairs & val_pairs) == 0, "Train-val leakage detected"
