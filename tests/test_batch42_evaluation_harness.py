"""
Unit and regression tests for Batch 42: Reproducible Evaluation Harness and Capability Measurement.
"""

import hashlib
import json
import math
from pathlib import Path
import pytest
import torch

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from experiments.batch42_reproducible_evaluation.evaluation.evaluator import (
    DeterministicEvaluator,
    compute_file_sha256,
)

EXPECTED_CANONICAL_HASH = "c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da"
EXPECTED_CANONICAL_PARAMS = 3_443_136
EXPECTED_STEP10000_HASH = "96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee"


def test_canonical_invariants_verification():
    """Confirms canonical fresh-init digest and step-10,000 checkpoint hash remain invariant."""
    torch.manual_seed(42)
    model = ChakrMicro(ModelConfig())
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params == EXPECTED_CANONICAL_PARAMS

    hasher = hashlib.sha256()
    with torch.no_grad():
        for name, param in sorted(model.named_parameters()):
            hasher.update(name.encode("utf-8"))
            hasher.update(param.detach().cpu().numpy().tobytes())
    assert hasher.hexdigest() == EXPECTED_CANONICAL_HASH

    base_ckpt = Path("experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt")
    assert base_ckpt.exists()
    assert compute_file_sha256(base_ckpt) == EXPECTED_STEP10000_HASH


def test_hand_computed_metric_aggregation(tmp_path):
    """Verifies that DeterministicEvaluator computes exact token-weighted cross-entropy loss against a hand-computed fixture."""
    base_ckpt = Path("experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt")
    if not base_ckpt.exists():
        pytest.skip("Base checkpoint missing")

    evaluator = DeterministicEvaluator(base_ckpt)

    # Create two synthetic records with known prompts & responses
    r1 = {
        "id": "rec_1",
        "prompt": "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n",
        "response": "World<|im_end|>\n",
    }
    r2 = {
        "id": "rec_2",
        "prompt": "<|im_start|>user\nTest<|im_end|>\n<|im_start|>assistant\n",
        "response": "Pass<|im_end|>\n",
    }

    test_ds = tmp_path / "test_fixture.jsonl"
    with open(test_ds, "w", encoding="utf-8") as f:
        f.write(json.dumps(r1) + "\n")
        f.write(json.dumps(r2) + "\n")

    res = evaluator.evaluate_dataset(test_ds)
    assert res["total_records"] == 2
    assert res["evaluated_records"] == 2
    assert res["skipped_records"] == 0

    # Verify manual aggregation
    r1_tokens = res["records"][0]["active_tokens"]
    r1_loss = res["records"][0]["loss"]
    r2_tokens = res["records"][1]["active_tokens"]
    r2_loss = res["records"][1]["loss"]

    expected_total_tokens = r1_tokens + r2_tokens
    assert res["total_active_tokens"] == expected_total_tokens

    expected_mean_loss = ((r1_loss * r1_tokens) + (r2_loss * r2_tokens)) / expected_total_tokens
    assert abs(res["mean_token_loss"] - expected_mean_loss) < 1e-3


def test_evaluator_handles_invalid_and_empty_records(tmp_path):
    """Verifies that invalid or empty records are recorded as skipped rather than crashing or distorting metrics."""
    base_ckpt = Path("experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt")
    if not base_ckpt.exists():
        pytest.skip("Base checkpoint missing")

    evaluator = DeterministicEvaluator(base_ckpt)

    records = [
        {"id": "valid_1", "prompt": "<|im_start|>user\nA<|im_end|>\n<|im_start|>assistant\n", "response": "B<|im_end|>\n"},
        {"id": "empty_resp", "prompt": "<|im_start|>user\nEmpty<|im_end|>\n<|im_start|>assistant\n", "response": ""},
    ]

    test_ds = tmp_path / "mixed_fixture.jsonl"
    with open(test_ds, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    res = evaluator.evaluate_dataset(test_ds)
    assert res["total_records"] == 2
    assert res["evaluated_records"] == 1
    assert res["skipped_records"] == 1
    assert res["records"][1]["status"] == "SKIPPED_EMPTY_RESPONSE"


def test_evaluator_reproducibility_on_aligned_validation_set():
    """Verifies that full dataset evaluation produces identical loss and perplexity across multiple runs."""
    ckpt_path = Path("experiments/batch41_evaluation_git/reproducibility/sft_aligned_checkpoint_step_20.pt")
    val_path = Path("experiments/batch41_evaluation_git/val_aligned.jsonl")
    if not ckpt_path.exists() or not val_path.exists():
        pytest.skip("Aligned checkpoint or validation set missing")

    evaluator_1 = DeterministicEvaluator(ckpt_path)
    res_1 = evaluator_1.evaluate_dataset(val_path)

    evaluator_2 = DeterministicEvaluator(ckpt_path)
    res_2 = evaluator_2.evaluate_dataset(val_path)

    assert res_1["mean_token_loss"] == res_2["mean_token_loss"]
    assert res_1["perplexity"] == res_2["perplexity"]
    assert res_1["total_active_tokens"] == res_2["total_active_tokens"] == 195
    assert res_1["mean_token_loss"] == 6.0563
    assert res_1["perplexity"] == 426.79
