"""
Regression test suite for Sprint Batches 75-78:
Controlled Synthetic Pretraining, Curriculum Learning, and Generalization Verification.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest
import torch

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from experiments.batch42_reproducible_evaluation.evaluation.evaluator import (
    compute_file_sha256,
    load_tokenizer_artifacts,
)

BASE_CKPT = Path("experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt")
EXPECTED_BASE_SHA = "96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee"
EXPECTED_INIT_SHA = "c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da"
EXP_DIR = Path("experiments/batch75_78_synthetic_pretraining")


def test_batch75_canonical_model_invariants():
    """Verify canonical ChakrMicro parameter count and initialization hash."""
    torch.manual_seed(42)
    model = ChakrMicro(ModelConfig())
    param_count = sum(p.numel() for p in model.parameters())
    assert param_count == 3443136, f"Parameter count mismatch: {param_count}"

    import hashlib
    hasher = hashlib.sha256()
    with torch.no_grad():
        for name, param in sorted(model.named_parameters()):
            hasher.update(name.encode("utf-8"))
            hasher.update(param.detach().cpu().numpy().tobytes())
    assert hasher.hexdigest() == EXPECTED_INIT_SHA, f"Init SHA mismatch: {hasher.hexdigest()}"


def test_batch75_checkpoint_10000_hash():
    """Verify bit-exact integrity of canonical step 10k checkpoint."""
    assert BASE_CKPT.exists(), "Base checkpoint missing"
    actual_sha = compute_file_sha256(str(BASE_CKPT))
    assert actual_sha == EXPECTED_BASE_SHA, f"Base checkpoint SHA mismatch: {actual_sha}"


def test_batch76_data_split_integrity_and_disjointness():
    """Verify that curriculum pretraining pairs do not leak into validation or held-out test."""
    manifest_path = EXP_DIR / "data" / "curriculum_manifest.json"
    assert manifest_path.exists(), "Curriculum manifest missing"
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    train_pairs = set(tuple(p) for p in manifest["allowed_train_pairs"])
    val_pairs = set(tuple(p) for p in manifest["isolated_val_pairs"])
    test_pairs = set(tuple(p) for p in manifest["isolated_test_pairs"])

    assert len(train_pairs & val_pairs) == 0, "Train-Val overlap detected!"
    assert len(train_pairs & test_pairs) == 0, "Train-Test overlap detected!"
    assert len(val_pairs & test_pairs) == 0, "Val-Test overlap detected!"
    assert manifest["total_tokens"] == 56231, "Token count mismatch in manifest"


def test_batch77_experiment_artifacts_and_metrics():
    """Verify that Batch 77 produced valid checkpoints, reproducible metrics, and diagnostic outputs."""
    summary_path = EXP_DIR / "experiments" / "batch77_experiment_summary.json"
    assert summary_path.exists(), "Batch 77 experiment summary missing"
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    # Condition B assertions
    cond_b = summary["condition_b_arithmetic_curriculum"]
    assert cond_b["metrics"]["test_held_out"]["modal_collapsed"] is True
    assert cond_b["metrics"]["test_held_out"]["eos_rate"] == 1.0
    assert "6" in cond_b["metrics"]["test_held_out"]["prediction_distribution"]

    # Checkpoint exists and hash matches
    ckpt_b_path = Path(cond_b["checkpoint_path"])
    assert ckpt_b_path.exists(), f"Condition B checkpoint missing: {ckpt_b_path}"
    sha_b = compute_file_sha256(str(ckpt_b_path))
    assert sha_b == cond_b["checkpoint_sha256"], "Condition B checkpoint SHA mismatch"


def test_batch78_final_report_validity():
    """Verify that final_report.json matches the documented conclusions."""
    report_json_path = EXP_DIR / "final_report.json"
    assert report_json_path.exists(), "final_report.json missing"
    with open(report_json_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    assert report["sprint"] == "BATCHES_75_78"
    assert report["conclusion"]["generalization_observed"] is False
    assert report["conclusion"]["modal_collapse_confirmed"] is True
    assert report["experiments"]["condition_b_arithmetic_curriculum"]["modal_attractor"] == "6"
