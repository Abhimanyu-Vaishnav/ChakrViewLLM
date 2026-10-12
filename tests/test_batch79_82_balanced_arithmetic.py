"""
Regression test suite for Sprint Batches 79-82:
Balanced Arithmetic Learning, Strict Generalization Tests, and Training Diagnosis.
"""

from __future__ import annotations

import hashlib
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
EXP_DIR = Path("experiments/batch79_82_balanced_arithmetic")


def test_batch79_canonical_invariants():
    """Verify parameter count and fresh initialization hash."""
    torch.manual_seed(42)
    model = ChakrMicro(ModelConfig())
    param_count = sum(p.numel() for p in model.parameters())
    assert param_count == 3443136

    hasher = hashlib.sha256()
    with torch.no_grad():
        for name, param in sorted(model.named_parameters()):
            hasher.update(name.encode("utf-8"))
            hasher.update(param.detach().cpu().numpy().tobytes())
    assert hasher.hexdigest() == EXPECTED_INIT_SHA
    assert compute_file_sha256(str(BASE_CKPT)) == EXPECTED_BASE_SHA


def test_batch79_benchmark_split_disjointness():
    """Verify unordered pair isolation and zero reversed-pair leakage across benchmark splits."""
    manifest_path = EXP_DIR / "data" / "benchmark_manifest.json"
    assert manifest_path.exists()
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    train_u = set(tuple(u) for u in manifest["splits"]["train"]["unordered_pairs"])
    val_u = set(tuple(u) for u in manifest["splits"]["validation"]["unordered_pairs"])
    test_u = set(tuple(u) for u in manifest["splits"]["test_held_out"]["unordered_pairs"])

    assert train_u.isdisjoint(val_u)
    assert train_u.isdisjoint(test_u)
    assert val_u.isdisjoint(test_u)
    assert len(train_u) + len(val_u) + len(test_u) == 55

    # Check sum class coverage in train
    train_sums = set(manifest["splits"]["train"]["sum_classes_covered"])
    assert train_sums == set(range(19))


def test_batch80_balanced_curriculum_manifest():
    """Verify that curriculum training data has uniform sum-class coverage."""
    curriculum_manifest = EXP_DIR / "data" / "curriculum_manifest.json"
    assert curriculum_manifest.exists()
    with open(curriculum_manifest, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["total_tokens"] == 63794
    assert len(data["sum_counts"]) == 19
    for s, count in data["sum_counts"].items():
        assert count == 150


def test_batch81_checkpoint_integrity_and_metrics():
    """Verify experiment checkpoints exist and match summary hashes."""
    summary_path = EXP_DIR / "experiments" / "batch81_experiment_summary.json"
    assert summary_path.exists()
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    cond_c = summary["condition_c_balanced_curriculum"]
    assert cond_c["metrics"]["test_held_out"]["eos_rate"] == 1.0

    ckpt_path = Path(cond_c["checkpoint_path"])
    assert ckpt_path.exists()
    assert compute_file_sha256(str(ckpt_path)) == cond_c["checkpoint_sha256"]


def test_batch82_final_report_consistency():
    """Verify final_report.json matches recorded observations."""
    report_path = EXP_DIR / "final_report.json"
    assert report_path.exists()
    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    assert report["sprint"] == "BATCHES_79_82"
    assert report["conclusion"]["generalization_supported"] is False
    assert report["benchmark"]["reversed_pair_leakage"] == 0
