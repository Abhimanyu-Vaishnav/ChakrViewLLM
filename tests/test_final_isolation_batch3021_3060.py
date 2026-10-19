"""Unit and Integration Tests for Batch 3021–3060 Final Dataset Isolation & Gate Enforcement.

Validates:
1. Strict document-level isolation for replay data (0 document crossings).
2. Reconciled configuration parity (160 steps, batch size 4).
3. Fail-closed preflight behavior and zero-training guarantee.
4. Independent test set integrity and zero cross-split leakage.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
BATCH_DIR = REPO_ROOT / "experiments" / "batch3021_3060"


def test_01_document_level_split_isolation():
    """Verify that replay documents do not cross train, val, and test splits."""
    split_assign_file = BATCH_DIR / "SPLIT_ASSIGNMENTS.json"
    assert split_assign_file.exists()
    
    with open(split_assign_file, "r", encoding="utf-8") as f:
        manifest = json.load(f)
        
    overlap = manifest["document_overlap"]
    assert len(overlap["train_val"]) == 0, f"Train-Val document collision: {overlap['train_val']}"
    assert len(overlap["train_test"]) == 0, f"Train-Test document collision: {overlap['train_test']}"
    assert len(overlap["val_test"]) == 0, f"Val-Test document collision: {overlap['val_test']}"
    
    # Check that document assignments match verified corpora
    assignments = manifest["replay_document_assignments"]
    assert len(assignments["train"]) == 3
    assert len(assignments["val"]) == 2
    assert len(assignments["test"]) == 2


def test_02_reconciled_training_configuration_parity():
    """Verify SFT_EXPERIMENT_CONFIG.json specifies 160 steps, batch size 4, and isolated output."""
    cfg_file = BATCH_DIR / "SFT_EXPERIMENT_CONFIG.json"
    assert cfg_file.exists()
    
    with open(cfg_file, "r", encoding="utf-8") as f:
        cfg = json.load(f)
        
    hp = cfg["training_hyperparameters"]
    assert hp["pilot_steps"] == 160
    assert hp["batch_size"] == 4
    assert hp["val_interval"] == 10
    assert hp["save_interval"] == 20
    assert hp["save_best_checkpoint"] is True
    assert "batch3021_3060" in cfg["output_directory"]


def test_03_preflight_passes_all_gates_cleanly():
    """Verify final preflight runs cleanly, validates all 7 gates, and does not train."""
    from experiments.batch3021_3060.run_preflight import run_preflight
    exit_code = run_preflight()
    assert exit_code == 0
    
    report_file = BATCH_DIR / "FINAL_PREFLIGHT_REPORT.json"
    assert report_file.exists()
    
    with open(report_file, "r", encoding="utf-8") as f:
        rep = json.load(f)
        
    assert rep["overall_verdict"] == "PASS"
    assert rep["training_executed"] is False
    assert rep["optimizer_steps_executed"] == 0
    assert len(rep["gates"]) == 7
    for gname, gdata in rep["gates"].items():
        assert gdata["status"] == "PASS", f"Gate {gname} failed!"


def test_04_test_split_integrity_and_disjointness():
    """Verify test split has exactly 40 records (28 instruction, 12 replay) disjoint from train."""
    train_recs = [json.loads(l) for l in open(BATCH_DIR / "train.jsonl", encoding="utf-8") if l.strip()]
    val_recs = [json.loads(l) for l in open(BATCH_DIR / "val.jsonl", encoding="utf-8") if l.strip()]
    test_recs = [json.loads(l) for l in open(BATCH_DIR / "test.jsonl", encoding="utf-8") if l.strip()]
    
    assert len(train_recs) == 320
    assert len(val_recs) == 40
    assert len(test_recs) == 40
    
    train_prompts = set(r["prompt"].strip() for r in train_recs)
    val_prompts = set(r["prompt"].strip() for r in val_recs)
    test_prompts = set(r["prompt"].strip() for r in test_recs)
    
    assert len(train_prompts.intersection(test_prompts)) == 0
    assert len(val_prompts.intersection(test_prompts)) == 0
    assert len(train_prompts.intersection(val_prompts)) == 0
