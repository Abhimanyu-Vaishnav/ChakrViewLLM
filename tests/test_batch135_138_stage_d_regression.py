"""
Tests for Batch 135–138 Stage D Pretraining Revival & Engine Integrity.
Verifies:
1. Canonical 10,000-step checkpoint file SHA-256 and baseline parameter invariants.
2. Stage C/D corpus split hash disjointness (zero train-val-test overlap).
3. Pretraining engine smoke checkpoint saving, step tracking, and resumability.
4. Stage D 100-step pretraining experiment artifact integrity and finite metrics.
"""
import json
import hashlib
from pathlib import Path
import pytest
import torch

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro

REPO_ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = REPO_ROOT / "experiments" / "batch135_138_stage_d_pretraining"
CKPT_DIR = EXP_DIR / "checkpoints"
REPORTS_DIR = EXP_DIR / "reports"
STAGE_C_MANIFEST = REPO_ROOT / "data" / "manifests" / "stage_c_manifest.json"
CANONICAL_BASE_CKPT = REPO_ROOT / "experiments" / "batch16_21_combined" / "main_training_10000" / "training_checkpoint_step_10000.pt"

def compute_file_sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def test_canonical_checkpoint_and_parameter_invariants():
    assert CANONICAL_BASE_CKPT.exists(), "Canonical Step 10,000 checkpoint file missing!"
    sha = compute_file_sha256(CANONICAL_BASE_CKPT)
    expected_sha = "96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee"
    assert sha == expected_sha, f"Canonical checkpoint hash mutated! Expected {expected_sha}, got {sha}"
    
    cfg = ModelConfig(max_seq_len=512)
    model = ChakrMicro(cfg)
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params == 3443136, f"Expected 3,443,136 parameters, got {total_params}"

def test_corpus_manifest_and_split_isolation():
    assert STAGE_C_MANIFEST.exists(), "Stage C manifest file missing!"
    with open(STAGE_C_MANIFEST, "r", encoding="utf-8") as f:
        man = json.load(f)
        
    docs = man.get("documents", [])
    assert len(docs) == 5534, f"Expected 5,534 documents, got {len(docs)}"
    
    split_hashes = {"train": set(), "validation": set(), "test": set()}
    for d in docs:
        split_hashes[d["split"]].add(d["sha256"])
        
    # Strictly zero overlap
    assert len(split_hashes["train"] & split_hashes["validation"]) == 0, "Train-Validation hash overlap detected!"
    assert len(split_hashes["train"] & split_hashes["test"]) == 0, "Train-Test hash overlap detected!"
    assert len(split_hashes["validation"] & split_hashes["test"]) == 0, "Validation-Test hash overlap detected!"

def test_smoke_test_and_resumption_integrity():
    smoke_report_p = REPORTS_DIR / "smoke_test_report.json"
    assert smoke_report_p.exists(), "Smoke test report missing!"
    with open(smoke_report_p, "r", encoding="utf-8") as f:
        rep = json.load(f)
        
    assert rep["status"] == "PASSED"
    assert rep["canonical_baseline_intact"] is True
    
    ckpt15 = CKPT_DIR / "smoke_test" / "training_checkpoint_step_15.pt"
    ckpt25 = CKPT_DIR / "smoke_test" / "training_checkpoint_step_25.pt"
    assert ckpt15.exists(), "Step 15 checkpoint missing"
    assert ckpt25.exists(), "Step 25 checkpoint missing"
    
    # Check that Step 25 contains step=25
    data25 = torch.load(ckpt25, map_location="cpu", weights_only=False)
    assert data25["step"] == 25
    assert "optimizer_state_dict" in data25
    assert "rng_state" in data25

def test_stage_d_experiment_results_integrity():
    exp_report_p = REPORTS_DIR / "stage_d_experiment_results.json"
    assert exp_report_p.exists(), "Stage D experiment report missing!"
    with open(exp_report_p, "r", encoding="utf-8") as f:
        rep = json.load(f)
        
    assert rep["steps"] == 100
    assert rep["tokens_processed"] == 51200
    assert "canonical_baseline" in rep
    assert "stage_d_pretrained" in rep
    
    stage_d_ckpt = Path(rep["stage_d_checkpoint"])
    assert stage_d_ckpt.exists(), "Stage D pilot checkpoint missing"
    
    # Verify baseline metrics are finite and realistic
    base_val_ppl = rep["canonical_baseline"]["validation"]["perplexity"]
    base_test_ppl = rep["canonical_baseline"]["test"]["perplexity"]
    assert 20.0 < base_val_ppl < 35.0, f"Unexpected baseline val ppl: {base_val_ppl}"
    assert 20.0 < base_test_ppl < 35.0, f"Unexpected baseline test ppl: {base_test_ppl}"
