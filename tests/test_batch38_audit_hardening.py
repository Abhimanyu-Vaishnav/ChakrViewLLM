"""
Regression and verification tests for Batch 38:
Audit Tool Hardening, Executable Code Validation, Dynamic Baseline Evaluation Harness,
Causal Shift & Masking Verification, and Fail-Closed SFT Readiness Gate.
"""

import json
import math
import os
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.runtime.local_runtime import DEFAULT_TOKENIZER_DIR
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

from experiments.batch38_audit_hardening.audits.validator import (
    SafeCodeExecutionHarness,
    compute_file_sha256,
)
from experiments.batch38_audit_hardening.evaluation.evaluator import (
    HardenedBaselineEvaluator,
    resolve_tokenizer_eos_id,
)
from experiments.batch38_audit_hardening.gates.sft_readiness_gate import (
    evaluate_sft_readiness_gate,
)


# =========================================================================
# 1. Tests for Code Execution Harness (Safe Behavioral Validation)
# =========================================================================

def test_mere_function_definition_is_not_behaviorally_validated():
    """Proves that compiling or exec'ing a function definition without invocation is not behavioral validation."""
    code_snippet = (
        "def broken_function(x):\n"
        "    return 1 / 0\n"  # Syntactically valid, but raises ZeroDivisionError when invoked!
    )
    # Merely compiling it or execing it succeeds without error!
    loc = {}
    exec(code_snippet, {}, loc)
    assert callable(loc["broken_function"])  # Definition succeeded!

    # But behavioral execution must fail!
    with pytest.raises(ZeroDivisionError):
        loc["broken_function"](10)


def test_code_execution_harness_behavioral_validation():
    """Tests SafeCodeExecutionHarness invoking compute_cross_entropy and binary_search."""
    # Test compute_cross_entropy
    cross_entropy_code = (
        "def compute_cross_entropy(logits: list[float], target: int) -> float:\n"
        "    import math\n"
        "    max_logit = max(logits)\n"
        "    sum_exp = sum(math.exp(x - max_logit) for x in logits)\n"
        "    log_sum_exp = max_logit + math.log(sum_exp)\n"
        "    return log_sum_exp - logits[target]\n"
    )
    res_ce = SafeCodeExecutionHarness.execute_and_verify("EXP-TR-CD-01", cross_entropy_code)
    assert res_ce["status"] == "PASS"
    assert res_ce["behaviorally_executed"] is True
    assert res_ce["test_cases_passed"] == 2

    # Test binary_search
    binary_search_code = (
        "def binary_search(arr, target):\n"
        "    low, high = 0, len(arr) - 1\n"
        "    while low <= high:\n"
        "        mid = (low + high) // 2\n"
        "        if arr[mid] == target: return mid\n"
        "        elif arr[mid] < target: low = mid + 1\n"
        "        else: high = mid - 1\n"
        "    return -1\n"
    )
    res_bs = SafeCodeExecutionHarness.execute_and_verify("EXP-TR-CD-02", binary_search_code)
    assert res_bs["status"] == "PASS"
    assert res_bs["behaviorally_executed"] is True
    assert res_bs["test_cases_passed"] == 9

    # Test unallowlisted code returns NOT_VERIFIED, not EXECUTION_PASS
    unknown_code = "def custom_func(x): return x * 2\n"
    res_unk = SafeCodeExecutionHarness.execute_and_verify("EXP-UNKNOWN-01", unknown_code)
    assert res_unk["status"] == "NOT_VERIFIED"
    assert res_unk["behaviorally_executed"] is False


# =========================================================================
# 2. Tests for Dynamic Baseline Evaluation Harness
# =========================================================================

def test_dynamic_hash_calculation():
    """Verifies that evaluator computes file bytes SHA-256 dynamically."""
    ckpt_path = "experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt"
    if not os.path.exists(ckpt_path):
        pytest.skip("Base checkpoint not found")

    computed_sha = compute_file_sha256(ckpt_path)
    assert computed_sha == "96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee"


def test_eos_id_resolution():
    """Verifies that EOS ID is dynamically resolved from tokenizer config."""
    eos_id = resolve_tokenizer_eos_id(DEFAULT_TOKENIZER_DIR, expected_eos_id=1)
    assert eos_id == 1

    # Should fail if expected EOS does not match
    with pytest.raises(ValueError):
        resolve_tokenizer_eos_id(DEFAULT_TOKENIZER_DIR, expected_eos_id=999)


def test_evaluation_mode_enforcement():
    """Verifies that HardenedBaselineEvaluator sets model to eval() mode explicitly."""
    ckpt_path = "experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt"
    if not os.path.exists(ckpt_path):
        pytest.skip("Base checkpoint not found")

    evaluator = HardenedBaselineEvaluator(checkpoint_path=ckpt_path)
    assert evaluator.model.training is False


def test_context_window_boundaries():
    """Verifies that evaluator enforces context window limit before model forward pass."""
    ckpt_path = "experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt"
    if not os.path.exists(ckpt_path):
        pytest.skip("Base checkpoint not found")

    # Set very small max_context_length (e.g. 5) to trigger boundary enforcement
    evaluator = HardenedBaselineEvaluator(checkpoint_path=ckpt_path, max_context_length=5)
    long_prompt = "This prompt contains many more tokens than five"
    res = evaluator.generate_greedy(long_prompt, max_new_tokens=10)
    assert res["stop_reason"] in ("context_limit_exceeded_at_prompt", "context_limit")


def test_missing_checkpoint_fails_loudly():
    """Verifies that evaluator raises FileNotFoundError if checkpoint file is missing."""
    with pytest.raises(FileNotFoundError):
        HardenedBaselineEvaluator(checkpoint_path="non_existent_checkpoint.pt")


def test_aggregate_metric_calculation_strictly_from_per_prompt():
    """Verifies that summary metrics match per_prompt_results.jsonl exactly."""
    summary_path = "experiments/batch38_audit_hardening/evaluation/summary.json"
    per_prompt_path = "experiments/batch38_audit_hardening/evaluation/per_prompt_results.jsonl"
    if not os.path.exists(summary_path) or not os.path.exists(per_prompt_path):
        pytest.skip("Evaluation results not found")

    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    recs = []
    with open(per_prompt_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                recs.append(json.loads(line))

    assert summary["total_evaluated"] == len(recs)
    calc_first_eos = sum(1 for r in recs if r["first_token_is_eos"])
    assert summary["first_token_eos_count"] == calc_first_eos
    calc_avg_tokens = sum(r["generated_token_count"] for r in recs) / len(recs)
    assert abs(summary["average_generated_tokens"] - calc_avg_tokens) < 1e-2


# =========================================================================
# 3. Tests for Actual SFT Training Behavior (Loss Masking, Causal Shift, Gradients)
# =========================================================================

def test_sft_causal_shift_and_target_alignment():
    """
    Demonstrates which token is predicted by each logit position,
    assistant-only target masking, causal shift, and EOS target inclusion.
    """
    p_toks = [10, 20]
    r_toks = [30, 40]
    bos_id = 0
    eos_id = 1
    ignore = -100

    input_seq = [bos_id] + p_toks + r_toks        # [0, 10, 20, 30, 40]
    target_seq = ([ignore] * len(p_toks)) + r_toks + [eos_id]  # [-100, -100, 30, 40, 1]

    # In causal language modeling:
    # Logit pos 0 (inp[0]=BOS) predicts inp[1]=10 -> target is masked (-100)
    # Logit pos 1 (inp[1]=10)  predicts inp[2]=20 -> target is masked (-100)
    # Logit pos 2 (inp[2]=20)  predicts inp[3]=30 -> target is 30 (first response token!)
    # Logit pos 3 (inp[3]=30)  predicts inp[4]=40 -> target is 40
    # Logit pos 4 (inp[4]=40)  predicts EOS=1     -> target is 1 (EOS included!)
    assert len(input_seq) == len(target_seq)
    assert target_seq[-1] == eos_id
    assert [t for t in target_seq if t != ignore] == [30, 40, 1]


def test_sft_loss_behavior_when_all_targets_masked():
    """Demonstrates that cross_entropy with ignore_index returns NaN when all targets are masked."""
    logits = torch.randn(1, 5, 4096)
    all_masked = torch.tensor([[-100, -100, -100, -100, -100]], dtype=torch.long)
    loss = F.cross_entropy(logits.view(-1, 4096), all_masked.view(-1), ignore_index=-100)
    assert torch.isnan(loss)


def test_sft_finite_loss_and_gradients_on_minimal_batch():
    """Verifies that model forward and backward pass produce finite loss and gradients."""
    cfg = ModelConfig(max_seq_len=512)
    model = ChakrMicro(cfg)
    model.train()

    inp = torch.tensor([[0, 10, 20, 30, 40]], dtype=torch.long)
    tgt = torch.tensor([[-100, -100, 30, 40, 1]], dtype=torch.long)

    logits = model(inp)
    loss = F.cross_entropy(logits.view(-1, 4096), tgt.view(-1), ignore_index=-100)
    assert math.isfinite(loss.item())

    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0
    assert all(torch.isfinite(g).all() for g in grads)


# =========================================================================
# 4. Tests for Enforced Fail-Closed SFT Readiness Gate
# =========================================================================

def test_sft_readiness_gate_fail_closed_logic():
    """Verifies that the readiness gate correctly reports independent statuses and fails closed."""
    gate = evaluate_sft_readiness_gate()
    assert gate["gate_decision"] == "BLOCKED"
    assert gate["independent_statuses"]["DATASET_READY"] is False
    assert gate["independent_statuses"]["BASELINE_READY"] is True
    assert gate["independent_statuses"]["PIPELINE_READY"] is True
    assert gate["independent_statuses"]["SFT_READY"] is False
    assert gate["blocking_reasons_count"] >= 1


def test_batch37_claims_audit_artifact_consistency():
    """Verifies that Batch 37 claims audit correctly identifies qualified and rejected claims."""
    audit_path = "experiments/batch38_audit_hardening/audits/batch37_claims_audit.json"
    if not os.path.exists(audit_path):
        pytest.skip("Batch 37 claims audit not found")

    with open(audit_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    claims = {c["claim_id"]: c for c in data["claims_evaluated"]}
    assert claims["CLM-B37-01"]["verdict"] == "QUALIFIED"
    assert claims["CLM-B37-02"]["verdict"] == "REJECTED"
    assert claims["CLM-B37-04"]["verdict"] == "CONFIRMED"
