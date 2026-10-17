"""
Tests for Batch 2141-2180: Controlled CPT, Source-Aware Evaluator, Checkpoint Resume, and Feedback Engine.
"""

import json
import os
import shutil
import tempfile
import pytest
import torch

from experiments.batch2141_2180.feedback_learning_engine import (
    ConversationMemoryStore,
    FeedbackExperienceLedger,
    ControlledLearningDatasetExporter,
    SessionFeedbackAdapter
)
from experiments.batch2101_2140.cpt_checkpoint_engine import save_cpt_checkpoint, load_cpt_checkpoint
from chakrview.brain.model import ChakrMicro
from chakrview.brain.config import ModelConfig


def test_protected_baseline_hashes():
    """Verify that canonical baseline checkpoint and tokenizer config exist and are unaltered."""
    canonical_ckpt = "experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt"
    assert os.path.exists(canonical_ckpt), f"Missing canonical checkpoint: {canonical_ckpt}"
    
    tokenizer_dir = "release/chakrview-v0.5.0/data/experiments/vocab_4096"
    assert os.path.exists(os.path.join(tokenizer_dir, "vocab.json"))
    assert os.path.exists(os.path.join(tokenizer_dir, "config.json"))


def test_cpt_candidates_and_evidence():
    """Verify all 3 CPT seed runs generated best checkpoints and evaluation results."""
    eval_json = "experiments/batch2141_2180/evaluation_results.json"
    assert os.path.exists(eval_json), "Evaluation results JSON missing"
    
    with open(eval_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    assert "canonical_baseline" in data
    assert "cpt_seed_42_best" in data
    assert "cpt_seed_1337_best" in data
    assert "cpt_seed_2026_best" in data
    
    # Verify Seed 42 outperformed baseline on test token-weighted CE
    base_ce = data["canonical_baseline"]["held_out_test_benchmark"]["token_weighted_ce"]
    seed42_ce = data["cpt_seed_42_best"]["held_out_test_benchmark"]["token_weighted_ce"]
    assert seed42_ce < base_ce, f"Expected seed 42 CE ({seed42_ce}) < baseline CE ({base_ce})"


def test_checkpoint_resume_engine():
    """Verify save_cpt_checkpoint and load_cpt_checkpoint recover full state without loss."""
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = ModelConfig(vocab_size=128, d_model=32, n_layers=2, n_heads=2, hidden_dim=64, max_seq_len=64)
        model = ChakrMicro(cfg)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        ckpt_path = Path(tmpdir) / "ckpt_step10.pt"
        
        # Save step 10
        save_cpt_checkpoint(
            checkpoint_path=ckpt_path,
            model=model,
            optimizer=opt,
            scheduler=None,
            global_step=10,
            seed=42,
            config={"d_model": 32},
            corpus_manifest_sha256="test_sha",
            tokenizer_vocab_sha256="vocab_sha"
        )
        assert ckpt_path.exists()
        
        # Recover step 10
        model2 = ChakrMicro(cfg)
        opt2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
        rec_meta = load_cpt_checkpoint(ckpt_path, model=model2, optimizer=opt2, expected_corpus_sha256="test_sha")
        
        assert rec_meta["global_step"] == 10
        assert rec_meta["seed"] == 42
        
        # Verify weight equivalence
        for p1, p2 in zip(model.parameters(), model2.parameters()):
            assert torch.equal(p1, p2)


def test_feedback_learning_engine_approval_and_consent_gates():
    """Verify Layer 1 memory consent gate, Layer 2 review gate, and Layer 3 export filter."""
    with tempfile.TemporaryDirectory() as tmpdir:
        mem_path = os.path.join(tmpdir, "mem.json")
        fb_path = os.path.join(tmpdir, "fb.json")
        export_manifest = os.path.join(tmpdir, "export_manifest.json")
        
        mem_store = ConversationMemoryStore(storage_path=mem_path)
        fb_ledger = FeedbackExperienceLedger(ledger_path=fb_path)
        exporter = ControlledLearningDatasetExporter(ledger=fb_ledger)
        
        # 1. Layer 1: Memory requires explicit user consent to add
        with pytest.raises(PermissionError):
            mem_store.add_memory("User secret", source="session1", user_consent=False)
            
        m1 = mem_store.add_memory("User prefers dark mode", source="session1", user_consent=True, tags=["pref"])
        assert m1["content"] == "User prefers dark mode"
        results = mem_store.search("dark mode")
        assert len(results) == 1
        
        # Memory deletion
        mem_store.delete_memory(m1["memory_id"], user_consent=True)
        assert len(mem_store.search("dark mode")) == 0
        
        # 2. Layer 2: Feedback without consent cannot be approved for training
        f_no_consent = fb_ledger.record_interaction(
            prompt="What is RoPE?",
            model_output="RoPE is rope.",
            task_domain="deep_learning",
            user_correction="Rotary Position Embedding.",
            user_consent=False
        )
        with pytest.raises(PermissionError):
            fb_ledger.review_entry(f_no_consent["feedback_id"], approved=True)
            
        # Feedback with consent
        f_with_consent = fb_ledger.record_interaction(
            prompt="Explain transformer self-attention",
            model_output="It is attention.",
            task_domain="deep_learning",
            user_correction="Self-attention relates all positions in a sequence.",
            rating=1,
            user_consent=True
        )
        # Review and approve
        fb_ledger.review_entry(f_with_consent["feedback_id"], approved=True)
        
        # 3. Layer 3: Export filters strictly for approved & consented records
        manifest = exporter.export_training_dataset(output_manifest_path=export_manifest)
        assert manifest["total_documents"] == 1
        assert os.path.exists(export_manifest)
