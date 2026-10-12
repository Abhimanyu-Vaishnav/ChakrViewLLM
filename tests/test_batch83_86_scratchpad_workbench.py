"""
Focused test suite for Sprint Batches 83-86:
Scratchpad Learning Experiment and Usable Local Workbench.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import pytest
import torch

from experiments.batch83_86_scratchpad_workbench.data.generate_scratchpad_data import (
    build_counting_string,
    parse_final_answer,
)
from experiments.batch83_86_scratchpad_workbench.workbench.app import (
    inspect_checkpoint,
    run_deterministic_eval,
    save_workbench_report,
    list_saved_reports,
)
from experiments.batch42_reproducible_evaluation.evaluation.evaluator import (
    compute_file_sha256,
)

BASE_CKPT = Path("experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt")
EXPECTED_BASE_SHA = "96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee"
EXP_DIR = Path("experiments/batch83_86_scratchpad_workbench")


def test_batch83_scratchpad_construction_and_parsing():
    """Verify deterministic counting sequence construction and final answer extraction."""
    # Test counting string
    assert build_counting_string(2, 3) == "3, 4, 5"
    assert build_counting_string(0, 0) == "none"
    assert build_counting_string(7, 2) == "8, 9"

    # Test parser
    assert parse_final_answer("Start at 2; count 3 more: 3, 4, 5; answer: 5<|im_end|>") == "5"
    assert parse_final_answer("Start at 0; count 0 more: none; answer: 0") == "0"
    assert parse_final_answer("18<|im_end|>") == "18"
    assert parse_final_answer("answer: 15") == "15"
    assert parse_final_answer("malformed generation with no digits") is None


def test_batch83_dataset_split_integrity():
    """Verify scratchpad manifest disjointness and matched token budgets."""
    manifest_path = EXP_DIR / "data" / "scratchpad_manifest.json"
    assert manifest_path.exists()
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest["disjointness"]["reversed_pair_leakage"] == 0
    assert manifest["training_tokens"]["matched_token_budget"] == 99417


def test_batch84_experiment_metrics_and_checkpoints():
    """Verify that Batch 84 produced valid checkpoints and reproducible evaluation records."""
    summary_path = EXP_DIR / "experiments" / "batch84_experiment_summary.json"
    assert summary_path.exists()
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    cond_b = summary["condition_b_scratchpad_arithmetic"]
    assert cond_b["metrics"]["test_scratchpad"]["valid_format_rate"] == 1.0
    assert cond_b["metrics"]["test_scratchpad"]["eos_rate"] == 1.0

    ckpt_b_path = Path(cond_b["checkpoint_path"])
    assert ckpt_b_path.exists()
    assert compute_file_sha256(str(ckpt_b_path)) == cond_b["checkpoint_sha256"]


def test_batch85_workbench_smoke_and_provenance():
    """Verify workbench inspector, deterministic oracle, and report persistence."""
    # 1. Inspector
    info = inspect_checkpoint(str(BASE_CKPT))
    assert info["valid"] is True
    assert info["sha256"] == EXPECTED_BASE_SHA

    # 2. Oracle
    oracle = run_deterministic_eval(3, 4, mode="scratchpad")
    assert oracle["true_sum"] == 7
    assert oracle["parsed_answer"] == "7"
    assert "Start at 3; count 4 more: 4, 5, 6, 7; answer: 7" in oracle["formatted_output"]
    assert oracle["source"] == "DETERMINISTIC_ORACLE"

    # 3. Report persistence
    sample_report = {
        "title": "Smoke Test Run",
        "summary": {"status": "ok", "test_mode": "unit_test"},
    }
    report_file = save_workbench_report(sample_report)
    assert Path(report_file).exists()
    reports = list_saved_reports()
    assert any("smoke_test" in r["filename"] for r in reports)
