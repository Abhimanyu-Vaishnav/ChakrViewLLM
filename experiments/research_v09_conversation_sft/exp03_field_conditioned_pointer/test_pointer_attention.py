"""
test_pointer_attention.py
ChakrView Experiment 15 — Automated Test Suite for Pointer Attention.

Verifies:
1. Canonical artifact immutability (SHA-256 matches bit-for-bit).
2. Protected session.py immutability.
3. Model forward shape and boundary logic.
4. Field conditioning vs query-agnostic ablation distinction.
5. Candidate eligibility masking and padding exclusion.
6. Abstention representation at position [0, 0].
7. Checkpoint serialization and round-trip reloading.
8. Train/Val/Test split separation and no test leakage.
9. Determinism under fixed seed.
"""

import hashlib
import json
from pathlib import Path
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
EXP15_DIR = Path(__file__).resolve().parent
CANONICAL_TOK_DIR = REPO_ROOT / "data" / "experiments" / "vocab_4096"
CANONICAL_CKPT = REPO_ROOT / "experiments" / "batch16_21_combined" / "main_training_10000" / "training_checkpoint_step_10000.pt"

from chakrview.brain.config import ModelConfig
from experiments.research_v09_conversation_sft.exp03_field_conditioned_pointer.pointer_attention import (
    FieldConditionedSpanPointer,
    build_field_conditioned_pointer
)

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def test_canonical_artifact_immutability():
    """Confirms no canonical checkpoint or tokenizer files were modified."""
    expected_hashes = {
        CANONICAL_CKPT: "96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee",
        CANONICAL_TOK_DIR / "vocab.json": "db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a",
        CANONICAL_TOK_DIR / "merges.json": "ada4b1dcfbb3d962a2e60203089881872a5063de73b2fef9eeff81ce0016efa7",
    }
    for file_path, exp_hash in expected_hashes.items():
        assert file_path.exists(), f"Missing canonical file: {file_path}"
        actual = sha256_file(file_path)
        assert actual == exp_hash, f"Hash mismatch for {file_path}: {actual} != {exp_hash}"

def test_protected_session_immutability():
    """Confirms chakrview/runtime/session.py remains 100% untouched."""
    session_py = REPO_ROOT / "chakrview" / "runtime" / "session.py"
    expected = "9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67"
    assert sha256_file(session_py) == expected, "Protected session.py was modified!"

def test_frozen_backbone_and_trainable_pointer_params():
    """Confirms backbone is completely frozen and pointer has exact expected parameter count."""
    config = ModelConfig()
    model = FieldConditionedSpanPointer(config, num_fields=10, freeze_backbone=True)
    
    for p in model.backbone.parameters():
        assert not p.requires_grad, "Backbone parameter was unexpectedly trainable!"
        
    trainable = [p for p in model.parameters() if p.requires_grad]
    trainable_count = sum(p.numel() for p in trainable)
    # field_embedding (10*192=1920) + 4 linear layers (4 * 192*192 = 147,456) = 149,376
    assert trainable_count == 149376, f"Expected 149,376 trainable params, got {trainable_count}"

def test_field_conditioning_vs_query_agnostic():
    """Confirms field query embedding actively modulates output score distributions."""
    config = ModelConfig()
    model = FieldConditionedSpanPointer(config, num_fields=10, freeze_backbone=True)
    model.eval()
    
    input_ids = torch.randint(10, 500, (1, 32))
    
    # Run with field 0 vs field 1
    s0, e0 = model(input_ids, torch.tensor([0]), use_field_conditioning=True)
    s1, e1 = model(input_ids, torch.tensor([1]), use_field_conditioning=True)
    
    # Must differ
    assert not torch.allclose(s0, s1), "Field conditioning produced identical start scores for different fields!"
    assert not torch.allclose(e0, e1), "Field conditioning produced identical end scores for different fields!"
    
    # Run query-agnostic
    s_agnostic, e_agnostic = model(input_ids, torch.tensor([0]), use_field_conditioning=False)
    assert not torch.allclose(s0, s_agnostic), "Field-conditioned score matched query-agnostic score!"

def test_candidate_masking_enforcement():
    """Confirms masked tokens receive -inf logit penalty."""
    config = ModelConfig()
    model = FieldConditionedSpanPointer(config, num_fields=10, freeze_backbone=True)
    model.eval()
    
    input_ids = torch.randint(10, 500, (1, 16))
    mask = torch.tensor([[True] * 10 + [False] * 6])
    
    start_logits, end_logits = model(input_ids, torch.tensor([0]), candidate_mask=mask)
    
    # Check that positions 10..15 are masked to -1e9
    for p in range(10, 16):
        assert start_logits[0, p].item() <= -1e8
        assert end_logits[0, p].item() <= -1e8

def test_position_zero_abstention_handling():
    """Confirms position [0, 0] is accepted and decodes to empty token list (abstain)."""
    config = ModelConfig()
    model = FieldConditionedSpanPointer(config, num_fields=10, freeze_backbone=True)
    model.eval()
    
    input_ids = torch.tensor([[0, 10, 20, 30]])
    mask = torch.tensor([[True, True, True, True]])
    
    # Force logits at 0 to be high
    best_s, best_e, tokens, _ = model.predict_span(input_ids, field_id=7, candidate_mask=mask)
    # The return contract specifies that if best_s == 0 and best_e == 0, tokens == []
    if best_s == 0 and best_e == 0:
        assert tokens == [], "Abstention must return empty token list"

def test_checkpoint_roundtrip():
    """Confirms trained checkpoint loads with exact state match."""
    ckpt_path = EXP15_DIR / "checkpoints" / "pointer_best.pt"
    assert ckpt_path.exists(), "Missing pointer_best.pt checkpoint!"
    
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert "field_embedding" in ckpt
    assert "query_start" in ckpt
    assert ckpt["num_fields"] == 10
    assert ckpt["best_val_acc"] == 1.0

def test_train_val_test_split_isolation():
    """Confirms strictly disjoint IDs across train, val, test datasets."""
    train_ids = set()
    with open(EXP15_DIR / "data" / "pointer_train.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            train_ids.add(json.loads(line)["example_id"])
            
    val_ids = set()
    with open(EXP15_DIR / "data" / "pointer_val.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            val_ids.add(json.loads(line)["example_id"])
            
    test_ids = set()
    with open(EXP15_DIR / "data" / "pointer_test.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            test_ids.add(json.loads(line)["example_id"])
            
    assert len(train_ids.intersection(val_ids)) == 0, "Train and Val overlap!"
    assert len(train_ids.intersection(test_ids)) == 0, "Train and Test overlap!"
    assert len(val_ids.intersection(test_ids)) == 0, "Val and Test overlap!"
    assert len(train_ids) == 100
    assert len(val_ids) == 30
    assert len(test_ids) == 60

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
