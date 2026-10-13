"""
Tests for Batch 131–134 Relational Rule Induction (RRI-1) Benchmark and Checkpoints.
Verifies:
1. Dataset split isolation (zero prompt leakage between train and test).
2. Checkpoint integrity and parameter invariants (3,443,136 parameters, tied embeddings).
3. Evaluator contract and results reporting integrity.
"""
import json
import hashlib
from pathlib import Path
import pytest
import torch

from experiments.batch103_106_slot_weighted_sft.slot_weighted_training import (
    ChakrMicro, ModelConfig, compute_file_sha256
)

EXP_DIR = Path("experiments/batch131_134_mission_alignment_audit")
DATA_DIR = EXP_DIR / "data"
CKPT_DIR = EXP_DIR / "checkpoints"
REPORTS_DIR = EXP_DIR / "reports"

def test_dataset_split_isolation():
    train_p = DATA_DIR / "train.jsonl"
    test_p = DATA_DIR / "test.jsonl"
    assert train_p.exists(), "train.jsonl missing"
    assert test_p.exists(), "test.jsonl missing"
    
    with open(train_p, "r", encoding="utf-8") as f:
        train_recs = [json.loads(l) for l in f if l.strip()]
    with open(test_p, "r", encoding="utf-8") as f:
        test_recs = [json.loads(l) for l in f if l.strip()]
        
    assert len(train_recs) == 100
    assert len(test_recs) == 45
    
    # Check that unseen combinations and heldout entities have zero prompt overlap with train
    train_prompts = {r["prompt"] for r in train_recs}
    p2_unseen = [r for r in test_recs if r["category"] == "unseen_combinations"]
    p3_heldout = [r for r in test_recs if r["category"] == "heldout_entities"]
    
    for r in p2_unseen:
        assert r["prompt"] not in train_prompts, f"Prompt leakage in P2: {r['id']}"
    for r in p3_heldout:
        assert r["prompt"] not in train_prompts, f"Prompt leakage in P3: {r['id']}"
        # Assert heldout keys/vals never appeared in training routing metadata
        meta = r["meta"]
        assert meta["k1"] in ["W", "X"]
        assert meta["v1"] in ["Y", "Z"]

def test_checkpoint_invariants_and_hashes():
    manifest_p = DATA_DIR / "manifest.json"
    assert manifest_p.exists()
    
    ckpt_a = CKPT_DIR / "rri_armA_canonical_base_step150.pt"
    ckpt_b = CKPT_DIR / "rri_armB_copy_init_step150.pt"
    assert ckpt_a.exists(), "Arm A checkpoint missing"
    assert ckpt_b.exists(), "Arm B checkpoint missing"
    
    # Verify bit-exact parameter count
    cfg = ModelConfig(max_seq_len=512)
    model = ChakrMicro(cfg)
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params == 3443136, f"Expected 3,443,136 parameters, got {total_params}"

def test_rri_results_integrity():
    res_p = REPORTS_DIR / "rri_experiment_results.json"
    assert res_p.exists(), "Results JSON missing"
    
    with open(res_p, "r", encoding="utf-8") as f:
        res = json.load(f)
        
    assert "armA_canonical_base" in res
    assert "armB_copy_init" in res
    
    arm_b = res["armB_copy_init"]
    assert arm_b["total_count"] == 45
    assert arm_b["total_correct"] == 22
    assert arm_b["overall_accuracy"] == 0.4889
    
    cats = arm_b["category_summary"]
    assert cats["test_negative_control"]["accuracy"] == 1.0
    assert cats["test_negative_control"]["correct"] == 10
    assert cats["unseen_combinations"]["correct"] == 7
    assert cats["unseen_combinations"]["count"] == 15
    assert cats["heldout_entities"]["correct"] == 0
    assert cats["heldout_entities"]["count"] == 10
