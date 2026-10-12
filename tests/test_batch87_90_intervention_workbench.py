"""
Focused regression tests for Sprint Batches 87-90:
Incremental Supervision Intervention and Workbench Training Monitor.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import pytest
import torch

from experiments.batch87_90_intervention_workbench.data.generate_incremental_data import (
    build_incremental_steps,
)
from experiments.batch87_90_intervention_workbench.workbench.app import (
    inspect_checkpoint,
    run_deterministic_eval,
    launch_training_job,
    save_workbench_report,
    list_saved_reports,
    ACTIVE_JOBS,
)
from experiments.batch42_reproducible_evaluation.evaluation.evaluator import (
    compute_file_sha256,
)

BASE_CKPT = Path("experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt")
EXPECTED_BASE_SHA = "96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee"
EXP_DIR = Path("experiments/batch87_90_intervention_workbench")


def test_batch87_incremental_step_construction():
    """Verify that incremental steps decompose a + b into b transitions."""
    steps_0 = build_incremental_steps(4, 0)
    assert len(steps_0) == 1
    assert steps_0[0]["is_final"] is True
    assert steps_0[0]["expected_next"] == 4

    steps_3 = build_incremental_steps(2, 3)
    assert len(steps_3) == 3
    assert [s["expected_next"] for s in steps_3] == [3, 4, 5]
    assert steps_3[-1]["is_final"] is True


def test_batch87_dataset_split_integrity():
    """Verify dataset manifest disjointness and matched token budgets."""
    manifest_path = EXP_DIR / "data" / "dataset_manifest.json"
    assert manifest_path.exists()
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest["disjointness"]["reversed_pair_leakage"] == 0
    assert manifest["training_tokens"]["matched_token_budget"] == 124289


def test_batch88_checkpoints_and_evaluation_integrity():
    """Verify that Batch 88 produced valid checkpoints and reproducible evaluation records."""
    summary_path = EXP_DIR / "experiments" / "batch88_experiment_summary.json"
    assert summary_path.exists()
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    cond_b = summary["condition_b_incremental"]
    ckpt_path = Path(cond_b["checkpoint_path"])
    assert ckpt_path.exists()
    assert compute_file_sha256(str(ckpt_path)) == cond_b["checkpoint_sha256"]
    assert cond_b["direct_eval"]["test"]["eos_rate"] == 1.0


def test_batch89_workbench_training_monitor_smoke():
    """Verify asynchronous training job launch, progress tracking, and report persistence."""
    job_id = launch_training_job("Test Smoke Job", total_steps=10)
    assert job_id in ACTIVE_JOBS
    assert ACTIVE_JOBS[job_id]["status"] in ["RUNNING", "COMPLETED"]

    # Wait briefly for completion
    import time
    for _ in range(50):
        if ACTIVE_JOBS[job_id]["status"] == "COMPLETED":
            break
        time.sleep(0.05)

    assert ACTIVE_JOBS[job_id]["status"] == "COMPLETED"
    assert ACTIVE_JOBS[job_id]["exit_code"] == 0
    assert len(ACTIVE_JOBS[job_id]["loss_history"]) == 10

    # Verify report was persisted
    reports = list_saved_reports()
    assert any("test_smoke_job" in r["filename"] for r in reports)
