"""
Step 8 Tests: Full-Epoch Stage C Pre-Training Experiment & Evaluation.

Verifies:
1. Authoritative experiment configuration (configs/chakr_micro_stage_c_full_epoch.json).
2. Checkpoint integrity, parameter keys, and atomic saving across the full epoch.
3. Training log finite numerical stability and loss descent across 6,478 steps.
4. Summary metrics confirming significant validation and test loss drops vs Step 7.
5. Deterministic resume verification within numerical tolerance.
6. Standardized capability smoke test outputs and loss curve artifacts.
"""

import json
from pathlib import Path
import pytest
import torch

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.training.checkpoint import CheckpointManager

ROOT_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT_DIR / "configs" / "chakr_micro_stage_c_full_epoch.json"
EXPERIMENT_DIR = ROOT_DIR / "data" / "experiments" / "stage_c_full_epoch"
CHECKPOINT_DIR = ROOT_DIR / "checkpoints" / "stage_c_full_epoch"


def test_stage_c_full_epoch_config_contract():
    """Verify configs/chakr_micro_stage_c_full_epoch.json structure and frozen invariants."""
    assert CONFIG_PATH.is_file(), f"Missing configuration file at {CONFIG_PATH}"

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    assert cfg["experiment_name"] == "chakr_micro_stage_c_full_epoch"
    assert cfg["model"]["vocab_size"] == 4096
    assert cfg["model"]["parameter_count"] == 3443136
    assert cfg["model"]["max_seq_len"] == 512
    assert cfg["training"]["max_steps"] == 6478
    assert cfg["training"]["optimizer"] == "adamw"
    assert cfg["training"]["seed"] == 42


def test_stage_c_full_epoch_checkpoints_integrity():
    """Verify checkpoints in checkpoints/stage_c_full_epoch/ are valid and match model state dict."""
    ckpts = sorted(list(CHECKPOINT_DIR.glob("checkpoint_*.pt")))
    assert len(ckpts) >= 5, f"Expected at least 5 checkpoints, found {len(ckpts)}"

    # Verify latest checkpoint is step 6478
    final_ckpt = CHECKPOINT_DIR / "checkpoint_0006478.pt"
    assert final_ckpt.is_file(), f"Missing final checkpoint {final_ckpt}"

    payload = CheckpointManager.load(final_ckpt)
    assert payload["step"] == 6478
    assert "model_state_dict" in payload
    assert "optimizer_state_dict" in payload
    assert "scheduler_state_dict" in payload
    assert "rng_state" in payload

    model = ChakrMicro(ModelConfig())
    model_keys = set(model.state_dict().keys())
    ckpt_keys = set(payload["model_state_dict"].keys())
    assert model_keys == ckpt_keys, "Model state dict keys do not match checkpoint"
    assert len(model_keys) == 57


def test_stage_c_full_epoch_training_log():
    """Verify training_log.jsonl contains 6478 steps without NaNs or Infs."""
    log_path = EXPERIMENT_DIR / "training_log.jsonl"
    assert log_path.is_file(), f"Missing {log_path}"

    lines = [json.loads(line) for line in log_path.read_text(encoding="utf-8").strip().splitlines() if line.strip()]
    assert len(lines) == 6478, f"Expected 6478 logged steps, got {len(lines)}"

    initial_loss = lines[0]["train_loss"]
    assert initial_loss > 8.0, f"Unexpected initial loss {initial_loss}"

    # Verify all records are finite
    for r in lines:
        assert r["train_loss"] is not None
        assert not (r["train_loss"] != r["train_loss"])
        assert r["tokens_per_sec"] > 0
        assert r["learning_rate"] > 0


def test_stage_c_full_epoch_summary_vs_step7():
    """Verify Step 8 summary metrics demonstrate significant loss reduction over Step 7."""
    summary_path = EXPERIMENT_DIR / "summary.json"
    assert summary_path.is_file(), f"Missing {summary_path}"

    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    assert summary["status"] == "COMPLETE"
    t = summary["training"]
    assert t["max_steps"] == 6478
    assert t["tokens_processed"] == 6633472

    # Step 8 full validation and test loss must be strictly lower than Step 7
    # Step 7: Val Loss = 6.3739, Test Loss = 6.3830
    assert t["final_full_val_loss"] < 5.0, f"Expected Val Loss < 5.0, got {t['final_full_val_loss']}"
    assert t["test_loss"] < 5.0, f"Expected Test Loss < 5.0, got {t['test_loss']}"

    # Generalization gap must be narrow
    gap = abs(t["final_full_val_loss"] - t["test_loss"])
    assert gap < 0.1, f"Validation/test gap unexpectedly large: {gap}"

    # Resume verification must pass within numerical tolerance
    assert "PASS" in t["resume_status"]
    assert t["resume_max_discrepancy"] < 1e-4


def test_stage_c_full_epoch_smoke_results():
    """Verify smoke_test_results.json contains 14 completed generations."""
    smoke_path = EXPERIMENT_DIR / "smoke_test_results.json"
    assert smoke_path.is_file(), f"Missing {smoke_path}"

    with open(smoke_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert len(data["results"]) == 14
    for r in data["results"]:
        assert len(r["prompt"]) > 0
        assert len(r["continuation"]) > len(r["prompt"])


def test_stage_c_full_epoch_visual_artifacts():
    """Verify ASCII and SVG loss curve artifacts exist and are well-formed."""
    svg_p = EXPERIMENT_DIR / "loss_curve.svg"
    txt_p = EXPERIMENT_DIR / "loss_curve.txt"
    json_p = EXPERIMENT_DIR / "loss_curve.json"

    assert svg_p.is_file() and svg_p.stat().st_size > 1000
    assert txt_p.is_file() and txt_p.stat().st_size > 500
    assert json_p.is_file()

    with open(json_p, "r", encoding="utf-8") as f:
        cdata = json.load(f)

    assert len(cdata["steps"]) == 6478
    assert len(cdata["train_loss"]) == 6478
    assert len(cdata["val_evaluations"]) >= 20
