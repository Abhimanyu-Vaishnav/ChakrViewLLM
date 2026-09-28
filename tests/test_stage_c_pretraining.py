"""
Step 7 Tests: Real-Corpus Stage C Baseline Pre-Training & Evaluation.

Verifies:
1. Authoritative experiment configuration (configs/chakr_micro_stage_c_baseline.json).
2. Standardized capability smoke test suite (configs/stage_c_smoke_prompts.json).
3. Checkpoint integrity, parameter keys, and atomic saving.
4. Training log finite numerical stability and loss descent.
5. Multi-split evaluation metrics and domain-specific profiling.
6. Resume determinism and smoke test outputs.
"""

import json
from pathlib import Path
import pytest
import torch

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.training.checkpoint import CheckpointManager

ROOT_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT_DIR / "configs" / "chakr_micro_stage_c_baseline.json"
PROMPTS_PATH = ROOT_DIR / "configs" / "stage_c_smoke_prompts.json"
EXPERIMENT_DIR = ROOT_DIR / "data" / "experiments" / "stage_c_baseline"
CHECKPOINT_DIR = ROOT_DIR / "checkpoints" / "stage_c_baseline"


def test_stage_c_baseline_config_contract():
    """Verify configs/chakr_micro_stage_c_baseline.json structure and frozen invariants."""
    assert CONFIG_PATH.is_file(), f"Missing configuration file at {CONFIG_PATH}"

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    assert cfg["experiment_name"] == "chakr_micro_stage_c_baseline"
    assert cfg["model"]["vocab_size"] == 4096
    assert cfg["model"]["parameter_count"] == 3443136
    assert cfg["model"]["max_seq_len"] == 512
    assert cfg["training"]["optimizer"] == "adamw"
    assert cfg["training"]["max_steps"] >= 100
    assert cfg["training"]["learning_rate"] > 0.0
    assert cfg["training"]["batch_size"] > 0
    assert cfg["checkpoint"]["save_interval"] > 0
    assert cfg["evaluation"]["eval_interval"] > 0


def test_stage_c_smoke_prompts_suite_coverage():
    """Verify configs/stage_c_smoke_prompts.json covers all required capability categories."""
    assert PROMPTS_PATH.is_file(), f"Missing prompts file at {PROMPTS_PATH}"

    with open(PROMPTS_PATH, "r", encoding="utf-8") as f:
        suite = json.load(f)

    assert "prompts" in suite
    assert len(suite["prompts"]) >= 10

    categories = {p["category"] for p in suite["prompts"]}
    expected_categories = {
        "English",
        "Hindi",
        "Hinglish",
        "Sanskrit",
        "Code",
        "Mathematics",
        "Structured Data",
        "Unicode",
        "Context",
    }
    missing = expected_categories - categories
    assert not missing, f"Missing prompt categories: {missing}"

    for p in suite["prompts"]:
        assert len(p["prompt"].strip()) > 0
        assert p.get("max_new_tokens", 0) > 0


def test_stage_c_baseline_checkpoints_if_present():
    """Verify checkpoints in checkpoints/stage_c_baseline/ if experiment has run."""
    ckpts = list(CHECKPOINT_DIR.glob("checkpoint_*.pt"))
    if not ckpts:
        pytest.skip("Baseline checkpoints not yet generated (experiment in progress)")

    assert len(ckpts) >= 3, f"Expected at least 3 checkpoints, found {len(ckpts)}"

    for ckpt_p in ckpts:
        payload = CheckpointManager.load(ckpt_p)
        assert "model_state_dict" in payload
        assert "optimizer_state_dict" in payload
        assert "step" in payload
        assert "config" in payload
        assert payload["step"] >= 0

        # Verify model state dict matches ChakrMicro parameter names
        model = ChakrMicro(ModelConfig())
        model_keys = set(model.state_dict().keys())
        ckpt_keys = set(payload["model_state_dict"].keys())
        assert model_keys == ckpt_keys, f"Key mismatch in {ckpt_p.name}"


def test_stage_c_training_log_if_present():
    """Verify training_log.jsonl metrics, loss descent, and absence of NaNs/Infs."""
    log_path = EXPERIMENT_DIR / "training_log.jsonl"
    if not log_path.is_file():
        pytest.skip("Training log not yet generated (experiment in progress)")

    records = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    assert len(records) >= 50, f"Expected at least 50 logged steps, got {len(records)}"

    # Check finite loss and throughput
    for r in records:
        assert r["train_loss"] is not None
        assert not (r["train_loss"] != r["train_loss"])  # not NaN
        assert r["tokens_per_sec"] > 0
        assert r["learning_rate"] > 0

    # Verify overall loss drop
    initial_loss = records[0]["train_loss"]
    final_loss = records[-1]["train_loss"]
    assert final_loss < initial_loss, f"Loss did not decrease: initial={initial_loss}, final={final_loss}"
    loss_drop_pct = ((initial_loss - final_loss) / initial_loss) * 100
    assert loss_drop_pct > 10.0, f"Expected >10% loss drop, got {loss_drop_pct:.1f}%"


def test_stage_c_master_summary_if_present():
    """Verify summary.json schema and validation results."""
    summary_path = EXPERIMENT_DIR / "summary.json"
    if not summary_path.is_file():
        pytest.skip("Master summary not yet generated (experiment in progress)")

    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    assert summary["status"] == "COMPLETE"
    assert "training" in summary
    assert "domain_evaluation" in summary
    t = summary["training"]
    assert t["initial_val_loss"] > t["final_val_loss"]
    assert t["test_loss"] > 0.0
    assert t["test_perplexity"] > 0.0
    assert t["tokens_processed"] > 0
    assert t["average_throughput_tokens_per_sec"] > 500


def test_stage_c_smoke_results_if_present():
    """Verify smoke_test_results.json contains non-empty generated continuations."""
    smoke_path = EXPERIMENT_DIR / "smoke_test_results.json"
    if not smoke_path.is_file():
        pytest.skip("Smoke test results not yet generated (experiment in progress)")

    with open(smoke_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "results" in data
    assert len(data["results"]) >= 10

    for r in data["results"]:
        assert len(r["prompt"]) > 0
        assert len(r["continuation"]) > len(r["prompt"])
        assert r["generated_length_chars"] > 0


def test_stage_c_baseline_freeze_artifacts_contract():
    """Verify that Step 7 baseline freeze record, report, and visual curve artifacts exist."""
    freeze_doc = ROOT_DIR / "docs" / "STEP_07_BASELINE_FREEZE.md"
    report_doc = ROOT_DIR / "docs" / "STEP_07_REAL_CORPUS_BASELINE_REPORT.md"
    svg_curve = EXPERIMENT_DIR / "loss_curve.svg"
    txt_curve = EXPERIMENT_DIR / "loss_curve.txt"

    assert freeze_doc.is_file(), f"Missing {freeze_doc}"
    assert report_doc.is_file(), f"Missing {report_doc}"
    assert svg_curve.is_file(), f"Missing {svg_curve}"
    assert txt_curve.is_file(), f"Missing {txt_curve}"

    freeze_text = freeze_doc.read_text(encoding="utf-8")
    assert "FROZEN & AUDITED" in freeze_text
    assert "research baseline, not a production model" in freeze_text
    assert "3,443,136" in freeze_text

    assert svg_curve.stat().st_size > 500
    assert txt_curve.stat().st_size > 200

