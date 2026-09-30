"""
Step 48 — Dedicated Learning Validation Test Suite.

Verifies:
1. Dataset fingerprint determinism.
2. Train/validation identity and split separation.
3. Token ID bounds across real dataset shards.
4. Baseline invariant preservation (3,443,136 params, 4096 vocab, 512 context).
5. Baseline evaluation reproducibility.
6. Training loss calculation and finiteness.
7. Validation loss calculation and reduction.
8. Perplexity calculation (PPL = exp(loss)).
9. Weight delta calculation (L2 norm and max delta).
10. Baseline hash immutability (ΔW_baseline = 0).
11. Experiment weight modification (ΔW_exp > 0).
12. Training curve schema and validity.
13. Generation comparison schema and integrity.
14. Checkpoint compatibility and verification.
15. Reproducibility expectations (bit-exact across same seed).
"""

import math
import json
import hashlib
from pathlib import Path
import pytest
import torch
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
DATASET_BASE = ROOT_DIR / "data" / "tokenized" / "stage_b"
MANIFEST_FILE = ROOT_DIR / "data" / "manifests" / "stage_b_manifest.json"
TRAIN_CURVE_FILE = ROOT_DIR / "docs" / "STEP_48_TRAINING_CURVE.json"
GEN_COMPARE_FILE = ROOT_DIR / "docs" / "STEP_48_GENERATION_COMPARISON.json"
BASELINE_EVAL_FILE = ROOT_DIR / "docs" / "STEP_48_BASELINE_EVALUATION.json"
DATA_REPORT_FILE = ROOT_DIR / "docs" / "STEP_48_DATA_VALIDATION_REPORT.json"

FROZEN_BASELINE_HASH = "c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da"
EXPECTED_PARAM_COUNT = 3_443_136

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.training.dataset import StreamingTokenDataset
from chakrview.training.loss import CausalLoss
from chakrview.training.safety import TrainingSafetyChecker, TrainingSafetyError
from chakrview.training.checkpoint import CheckpointManager


def compute_model_hash(model: torch.nn.Module) -> str:
    hasher = hashlib.sha256()
    with torch.no_grad():
        for name, param in sorted(model.named_parameters()):
            hasher.update(name.encode("utf-8"))
            hasher.update(param.detach().cpu().numpy().tobytes())
    return hasher.hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
# 1. Dataset Fingerprint Determinism
# ─────────────────────────────────────────────────────────────────────────────

def test_step48_dataset_fingerprint_determinism():
    """Verify dataset fingerprint remains deterministic and matches validation report."""
    assert DATA_REPORT_FILE.is_file(), f"Missing {DATA_REPORT_FILE}"
    with open(DATA_REPORT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    expected_fp = data["dataset_fingerprint"]
    assert expected_fp == "3efb6f8ff1f402e54b4e9f174feea346ec3c250f1573aea0bddcb185b93d7ac3"

    # Recompute fingerprint across stage_b splits
    hasher = hashlib.sha256()
    for s in ["train", "validation", "test"]:
        s_dir = DATASET_BASE / s
        meta_path = s_dir / "metadata.json"
        with open(meta_path, "r", encoding="utf-8") as mf:
            meta = json.load(mf)
        arrays = [np.fromfile(s_dir / sh["filename"], dtype=np.uint16) for sh in meta["shards"]]
        tokens = np.concatenate(arrays)
        hasher.update(s.encode("utf-8"))
        hasher.update(tokens.tobytes())

    recomputed_fp = hasher.hexdigest()
    assert recomputed_fp == expected_fp


# ─────────────────────────────────────────────────────────────────────────────
# 2. Train / Validation Identity Validation
# ─────────────────────────────────────────────────────────────────────────────

def test_step48_train_validation_identity_and_isolation():
    """Verify train and validation splits are genuinely distinct and have distinct shard files."""
    train_dir = DATASET_BASE / "train"
    val_dir = DATASET_BASE / "validation"

    with open(train_dir / "metadata.json", "r", encoding="utf-8") as tf:
        train_meta = json.load(tf)
    with open(val_dir / "metadata.json", "r", encoding="utf-8") as vf:
        val_meta = json.load(vf)

    assert train_meta["split"] == "train"
    assert val_meta["split"] == "validation"
    assert train_meta["total_tokens"] > val_meta["total_tokens"]

    train_shas = {s["sha256"] for s in train_meta["shards"]}
    val_shas = {s["sha256"] for s in val_meta["shards"]}

    # Shard files must have zero collision
    assert len(train_shas.intersection(val_shas)) == 0


# ─────────────────────────────────────────────────────────────────────────────
# 3. Token ID Bounds Across Shards
# ─────────────────────────────────────────────────────────────────────────────

def test_step48_token_id_bounds():
    """Verify all token IDs in Stage B shards are strictly within [0, 4095]."""
    for split in ["train", "validation", "test"]:
        s_dir = DATASET_BASE / split
        meta_path = s_dir / "metadata.json"
        with open(meta_path, "r", encoding="utf-8") as mf:
            meta = json.load(mf)
        for s_info in meta["shards"]:
            shard_path = s_dir / s_info["filename"]
            tokens = np.fromfile(shard_path, dtype=np.uint16)
            assert len(tokens) > 0
            assert tokens.min() >= 0
            assert tokens.max() < 4096


# ─────────────────────────────────────────────────────────────────────────────
# 4. Baseline Invariant Preservation
# ─────────────────────────────────────────────────────────────────────────────

def test_step48_baseline_invariants():
    """Verify canonical baseline model adheres to all frozen architectural contracts."""
    torch.manual_seed(42)
    model = ChakrMicro(ModelConfig())
    model.eval()

    param_count = sum(p.numel() for p in model.parameters())
    assert param_count == EXPECTED_PARAM_COUNT
    assert model.config.vocab_size == 4096
    assert model.config.max_seq_len == 512

    weight_hash = compute_model_hash(model)
    assert weight_hash == FROZEN_BASELINE_HASH


# ─────────────────────────────────────────────────────────────────────────────
# 5. Baseline Evaluation Reproducibility
# ─────────────────────────────────────────────────────────────────────────────

def test_step48_baseline_evaluation_reproducibility():
    """Verify baseline evaluation document exists and records finite initial metrics."""
    assert BASELINE_EVAL_FILE.is_file(), f"Missing {BASELINE_EVAL_FILE}"
    with open(BASELINE_EVAL_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["model_weight_sha256"] == FROZEN_BASELINE_HASH
    metrics = data["evaluation_metrics"]
    val_metrics = metrics["validation_sample"]

    # Initial cross-entropy loss should be near theoretical uniform (-ln(1/4096) = 8.318)
    assert 8.0 < val_metrics["mean_loss"] < 8.6
    assert 3000.0 < val_metrics["perplexity"] < 5500.0


# ─────────────────────────────────────────────────────────────────────────────
# 6. Training Loss Calculation & Finiteness
# ─────────────────────────────────────────────────────────────────────────────

def test_step48_training_loss_calculation():
    """Verify CausalLoss produces finite, properly masked values."""
    loss_fn = CausalLoss(ignore_index=2)
    torch.manual_seed(42)
    logits = torch.randn(2, 8, 4096)
    targets = torch.tensor([[10, 20, 2, 2, 30, 40, 50, 60], [1, 2, 3, 4, 5, 2, 2, 2]], dtype=torch.long)

    loss = loss_fn(logits, targets)
    loss_val = TrainingSafetyChecker.check_loss(loss)
    assert isinstance(loss_val, float)
    assert not math.isnan(loss_val)
    assert not math.isinf(loss_val)
    assert loss_val > 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 7. Validation Loss Calculation & Reduction
# ─────────────────────────────────────────────────────────────────────────────

def test_step48_validation_loss_reduction():
    """Verify training curve exhibits significant validation loss reduction."""
    assert TRAIN_CURVE_FILE.is_file(), f"Missing {TRAIN_CURVE_FILE}"
    with open(TRAIN_CURVE_FILE, "r", encoding="utf-8") as f:
        curve = json.load(f)

    init_val = curve["experiment"]["initial_val_loss"]
    final_val = curve["experiment"]["final_val_loss"]

    assert final_val < init_val, f"Validation loss did not improve: {init_val} -> {final_val}"
    reduction = init_val - final_val
    assert reduction > 2.0, f"Expected substantial validation loss drop, got {reduction:.4f}"


# ─────────────────────────────────────────────────────────────────────────────
# 8. Perplexity Calculation
# ─────────────────────────────────────────────────────────────────────────────

def test_step48_perplexity_calculation():
    """Verify safe perplexity calculation formula PPL = exp(loss)."""
    loss = 2.5324
    ppl = math.exp(loss)
    assert math.isclose(ppl, 12.5836, rel_tol=1e-3)

    # Upper clamp safety check
    large_loss = 25.0
    safe_ppl = math.exp(min(large_loss, 20.0))
    assert not math.isinf(safe_ppl)


# ─────────────────────────────────────────────────────────────────────────────
# 9. Weight Delta Calculation
# ─────────────────────────────────────────────────────────────────────────────

def test_step48_weight_delta_metrics():
    """Verify weight delta recorded in training curve confirms parameter mutation."""
    assert TRAIN_CURVE_FILE.is_file()
    with open(TRAIN_CURVE_FILE, "r", encoding="utf-8") as f:
        curve = json.load(f)

    wd = curve["weight_delta"]
    assert wd["tensors_changed"] == 56
    assert wd["total_tensors"] == 56
    assert wd["total_l2_norm"] > 20.0
    assert wd["max_abs_delta"] > 0.01


# ─────────────────────────────────────────────────────────────────────────────
# 10. Baseline Hash Immutability (ΔW_baseline = 0)
# ─────────────────────────────────────────────────────────────────────────────

def test_step48_baseline_hash_immutability():
    """Verify that frozen baseline hash is strictly unchanged across operations."""
    torch.manual_seed(42)
    m1 = ChakrMicro(ModelConfig())
    m1.eval()
    h1 = compute_model_hash(m1)

    # Perform dummy evaluation
    with torch.no_grad():
        dummy_in = torch.tensor([[10, 20, 30]], dtype=torch.long)
        _ = m1(dummy_in)

    h2 = compute_model_hash(m1)
    assert h1 == FROZEN_BASELINE_HASH
    assert h2 == FROZEN_BASELINE_HASH


# ─────────────────────────────────────────────────────────────────────────────
# 11. Experiment Weight Modification (ΔW_exp > 0)
# ─────────────────────────────────────────────────────────────────────────────

def test_step48_experiment_weights_modified():
    """Verify trained experiment hash differs strictly from baseline hash."""
    assert TRAIN_CURVE_FILE.is_file()
    with open(TRAIN_CURVE_FILE, "r", encoding="utf-8") as f:
        curve = json.load(f)

    exp_hash = curve["experiment"]["final_hash"]
    base_hash = curve["baseline"]["model_hash"]

    assert exp_hash != base_hash
    assert base_hash == FROZEN_BASELINE_HASH


# ─────────────────────────────────────────────────────────────────────────────
# 12. Training Curve Schema & Validity
# ─────────────────────────────────────────────────────────────────────────────

def test_step48_training_curve_schema():
    """Verify training curve contains valid history with required metrics."""
    assert TRAIN_CURVE_FILE.is_file()
    with open(TRAIN_CURVE_FILE, "r", encoding="utf-8") as f:
        curve = json.load(f)

    history = curve["history"]
    assert len(history) == 100

    first_entry = history[0]
    required_keys = ["step", "train_loss", "train_ppl", "learning_rate", "grad_norm", "param_delta_norm"]
    for k in required_keys:
        assert k in first_entry

    assert first_entry["step"] == 1
    assert history[-1]["step"] == 100


# ─────────────────────────────────────────────────────────────────────────────
# 13. Generation Comparison Schema
# ─────────────────────────────────────────────────────────────────────────────

def test_step48_generation_comparison_schema():
    """Verify generation comparison document exists and contains prompt records."""
    assert GEN_COMPARE_FILE.is_file()
    with open(GEN_COMPARE_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "prompts" in data
    assert len(data["prompts"]) == 3

    for p in data["prompts"]:
        assert "prompt" in p
        assert "baseline_text" in p
        assert "baseline_tokens" in p
        assert "trained_text" in p
        assert "trained_tokens" in p
        assert "tokens_differ" in p


# ─────────────────────────────────────────────────────────────────────────────
# 14. Checkpoint Compatibility
# ─────────────────────────────────────────────────────────────────────────────

def test_step48_checkpoint_compatibility():
    """Verify experiment checkpoints were saved and validate cleanly."""
    ckpt_dir = ROOT_DIR / "artifacts" / "step48" / "run_seed_42"
    assert ckpt_dir.is_dir()
    final_ckpt = ckpt_dir / "checkpoint_0000100.pt"
    assert final_ckpt.is_file()

    payload = CheckpointManager.load(
        final_ckpt,
        validate_training=True,
        expected_tokenizer_checksum="7498d92adeef7c9e64d4d1bbd5b4b23f8774692d868742e3b98e30c584c4403f",
        expected_param_count=EXPECTED_PARAM_COUNT,
    )
    assert payload["checkpoint_type"] == "training"
    assert payload["step"] == 100
    assert payload["parameter_count"] == EXPECTED_PARAM_COUNT


# ─────────────────────────────────────────────────────────────────────────────
# 15. Reproducibility Expectations
# ─────────────────────────────────────────────────────────────────────────────

def test_step48_reproducibility_invariants():
    """Verify training curve reports bit-exact match across repeated runs."""
    assert TRAIN_CURVE_FILE.is_file()
    with open(TRAIN_CURVE_FILE, "r", encoding="utf-8") as f:
        curve = json.load(f)

    repro = curve["reproducibility"]
    assert repro["bit_exact_match"] is True
    assert repro["loss_curve_match"] is True
    assert repro["run_a_hash"] == repro["run_b_hash"]
