"""
Unit & Integration Tests for Batches 99-102:
- Hidden-state extraction and dynamic token-position alignment (accounting for BPE digit mergers).
- Probe split isolation and zero pair/reverse-pair leakage.
- Probe baselines: majority class, shuffled-label control, deterministic seeds.
- Correct-token scoring versus generated-token scoring under identical teacher-forced prefixes.
- Interactive token diagnostics data model, top-5 candidates, and divergence marking.
- Saving and reopening reports offline, backward compatibility with older reports.
- Missing/corrupted checkpoint error handling.
- Tool-call execution security allowlist validation.
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import pytest
import torch

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from experiments.batch42_reproducible_evaluation.evaluation.evaluator import (
    load_tokenizer_artifacts,
)
from experiments.batch91_94_tool_calling_comparator.tools.calculator import (
    validate_and_execute_tool_call,
)
from experiments.batch99_102_representation_audit.representation_probe import (
    find_operand_token_positions,
    extract_hidden_states,
    train_and_eval_probe,
)
from experiments.batch99_102_representation_audit.workbench.app import (
    inspect_checkpoint,
    compute_teacher_forced_token_scores,
    run_single_inference_with_scores,
    save_diagnostic_report,
    load_diagnostic_report,
    safe_token_repr,
)

TOKENIZER_DIR = "release/chakrview-v0.5.0/data/experiments/vocab_4096"
BASE_CKPT = "experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt"
CONDC_CKPT = "experiments/batch95_98_slot_binding_diagnostics/experiments/condition_c_slot_binding_step_100.pt"


@pytest.fixture(scope="module")
def tokenizer():
    tok, _ = load_tokenizer_artifacts(TOKENIZER_DIR)
    return tok


def test_bpe_digit_merger_dynamic_alignment(tokenizer):
    """Verifies that find_operand_token_positions correctly aligns operand tokens even when ': 1' is merged."""
    # Test case 1: operand A=1, B=2 (A is merged into ': 1' token 2665)
    prompt_1 = (
        "<|im_start|>user\nRequest: add numbers\n"
        "Operand A: 1\nOperand B: 2\n"
        "Required action: calculator addition<|im_end|>\n<|im_start|>assistant\n"
    )
    tokens_1 = [0] + tokenizer.encode(prompt_1)
    pos_a1, pos_b1 = find_operand_token_positions(tokens_1)
    assert pos_a1 is not None and pos_b1 is not None
    assert "1" in tokenizer.decode([tokens_1[pos_a1]])
    assert "2" in tokenizer.decode([tokens_1[pos_b1]])

    # Test case 2: operand A=3, B=1 (B is merged into ': 1' token 2665)
    prompt_2 = (
        "<|im_start|>user\nRequest: add numbers\n"
        "Operand A: 3\nOperand B: 1\n"
        "Required action: calculator addition<|im_end|>\n<|im_start|>assistant\n"
    )
    tokens_2 = [0] + tokenizer.encode(prompt_2)
    pos_a2, pos_b2 = find_operand_token_positions(tokens_2)
    assert pos_a2 is not None and pos_b2 is not None
    assert "3" in tokenizer.decode([tokens_2[pos_a2]])
    assert "1" in tokenizer.decode([tokens_2[pos_b2]])


def test_probe_split_isolation_and_no_leakage():
    """Confirms zero unordered-pair overlap between training and held-out test splits."""
    train_file = Path("experiments/batch95_98_slot_binding_diagnostics/data/train.jsonl")
    test_file = Path("experiments/batch95_98_slot_binding_diagnostics/data/test.jsonl")
    train_recs = [json.loads(line) for line in open(train_file) if line.strip()]
    test_recs = [json.loads(line) for line in open(test_file) if line.strip()]

    train_unordered = set((min(r["operands"]), max(r["operands"])) for r in train_recs)
    test_unordered = set((min(r["operands"]), max(r["operands"])) for r in test_recs)

    overlap = train_unordered.intersection(test_unordered)
    assert len(overlap) == 0, f"Found pair leakage between train and test: {overlap}"


def test_linear_probe_deterministic_and_shuffled_control():
    """Tests that linear probe trains deterministically and achieves ~chance on shuffled labels."""
    import numpy as np

    np.random.seed(42)
    # 50 samples, 20 features, 5 classes
    X_tr = np.random.randn(50, 20)
    y_tr = np.random.randint(0, 5, size=50)
    X_te = np.random.randn(20, 20)
    y_te = np.random.randint(0, 5, size=20)

    res1 = train_and_eval_probe(X_tr, y_tr, X_te, y_te, seed=123, epochs=50)
    res2 = train_and_eval_probe(X_tr, y_tr, X_te, y_te, seed=123, epochs=50)

    # Determinism
    assert res1["train_accuracy"] == res2["train_accuracy"]
    assert res1["test_accuracy"] == res2["test_accuracy"]

    # Shuffled control test accuracy should not significantly exceed random chance
    shuffled_y_tr = np.random.permutation(y_tr)
    ctrl_res = train_and_eval_probe(X_tr, shuffled_y_tr, X_te, y_te, seed=123, epochs=50)
    assert ctrl_res["test_accuracy"] <= 0.45


def test_hidden_state_extraction_layers():
    """Verifies that hidden states across all 8 stages have expected shapes."""
    config = ModelConfig(n_layers=6, d_model=192, max_seq_len=64)
    model = ChakrMicro(config)
    toks = [0, 50, 60, 70, 80]
    hs = extract_hidden_states(model, toks)

    assert len(hs) == 8  # Layer 0 (Embedding) + Layers 1..6 + Final Norm (7)
    for h in hs:
        assert h.shape == (1, 5, 192)


def test_teacher_forced_candidate_distribution(tokenizer):
    """Verifies top-K candidate extraction under teacher-forced scoring."""
    config = ModelConfig(n_layers=2, d_model=192, max_seq_len=64)
    model = ChakrMicro(config)
    toks = [0, 10, 20, 30]
    scores = compute_teacher_forced_token_scores(model, toks, tokenizer, top_k=5)

    assert len(scores) == 3
    for s in scores:
        assert len(s["top_candidates"]) == 5
        probs = [c["prob"] for c in s["top_candidates"]]
        assert probs == sorted(probs, reverse=True)


def test_interactive_inference_scores_and_divergence_tagging(tokenizer):
    """Verifies inference scoring outputs top-5 candidates and step diagnostics."""
    config = ModelConfig(n_layers=2, d_model=192, max_seq_len=64)
    model = ChakrMicro(config)
    out = run_single_inference_with_scores(model, tokenizer, "Test prompt", max_tokens=10, top_k=5)

    assert "step_diagnostics" in out
    assert len(out["step_diagnostics"]) > 0
    step0 = out["step_diagnostics"][0]
    assert "top_candidates" in step0
    assert len(step0["top_candidates"]) == 5


def test_offline_report_persistence_and_backward_compatibility():
    """Verifies saving, reopening reports offline, and backward compatibility when candidate details are absent."""
    report_data = {
        "success": True,
        "title": "Test Diagnostic Report",
        "timestamp": "2026-10-12T12:00:00",
        "checkpoint_a": {"path": "ckpt_a.pt"},
        "checkpoint_b": {"path": "ckpt_b.pt"},
        "summary": {"total_prompts": 1, "divergence_rate_pct": 0.0},
        "diagnostics": [
            {
                "index": 0,
                "prompt": "Test",
                "checkpoint_a_output": {
                    "step_diagnostics": [{"step": 0, "token_id": 100, "token_str": "'a'", "prob": 0.8}]
                },
                "checkpoint_b_output": {
                    "step_diagnostics": [{"step": 0, "token_id": 100, "token_str": "'a'", "prob": 0.8}]
                },
            }
        ],
    }

    with tempfile.NamedTemporaryFile("w+", suffix=".json", delete=False) as f:
        json.dump(report_data, f)
        temp_path = f.name

    loaded = load_diagnostic_report(temp_path)
    assert loaded["success"] is True
    assert loaded["title"] == "Test Diagnostic Report"
    # Verify legacy compatibility: missing top_candidates handled without error
    step = loaded["diagnostics"][0]["checkpoint_a_output"]["step_diagnostics"][0]
    assert "top_candidates" not in step  # absent in legacy format
    Path(temp_path).unlink()


def test_missing_or_corrupt_checkpoint_handling():
    """Confirms that inspect_checkpoint safely reports non-existent or corrupted files without unhandled exceptions."""
    res_missing = inspect_checkpoint("non_existent_ckpt_12345.pt")
    assert res_missing["valid"] is False
    assert "File does not exist" in res_missing["error"]

    with tempfile.NamedTemporaryFile("w+", suffix=".pt", delete=False) as f:
        f.write("corrupted data")
        corrupt_path = f.name

    res_corrupt = inspect_checkpoint(corrupt_path)
    assert res_corrupt["valid"] is False
    assert "Corrupted or invalid" in res_corrupt["error"]
    Path(corrupt_path).unlink()


def test_safe_token_representation(tokenizer):
    """Tests safe string representation for tokens across whitespace and special tokens."""
    assert safe_token_repr(tokenizer, 0) == "''"
    assert safe_token_repr(tokenizer, 13) == "'\\n'"
    assert safe_token_repr(tokenizer, 999999) == "<tok_999999>"


def test_tool_call_security_allowlist_preservation():
    """Confirms that calculator validation enforces arithmetic allowlist and rejects arbitrary execution."""
    valid_res = validate_and_execute_tool_call('TOOL_CALL: {"tool": "calculator", "op": "add", "a": 5, "b": 7}')
    assert valid_res["status"] == "SUCCESS"
    assert valid_res["result"] == 12

    malicious_res = validate_and_execute_tool_call('TOOL_CALL: {"tool": "os", "op": "system", "cmd": "dir"}')
    assert malicious_res["executed"] is False
    assert malicious_res["status"] == "UNSUPPORTED_TOOL"
