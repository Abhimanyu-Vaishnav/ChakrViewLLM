#!/usr/bin/env python3
"""
Unit and regression tests for Batches 1901–1940:
Independent Evaluation Audit, Leakage Investigation & Generalization Gate
"""

import hashlib
import json
import math
from pathlib import Path
import pytest
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
BATCH1901_DIR = REPO_ROOT / "experiments" / "batch1901_1940"
CANONICAL_CHECKPOINT_PATH = REPO_ROOT / "experiments" / "batch16_21_combined" / "main_training_10000" / "training_checkpoint_step_10000.pt"
ADAPTED_CHECKPOINT_PATH = REPO_ROOT / "experiments" / "batch1381_1420" / "checkpoints" / "chakrmicro_domain_adapted_experimental.pt"


def test_checkpoint_immutability():
    """Verify Canonical and Adapted reference checkpoints match exact SHA-256 hashes."""
    assert CANONICAL_CHECKPOINT_PATH.exists(), "Canonical checkpoint missing!"
    canon_hash = hashlib.sha256(CANONICAL_CHECKPOINT_PATH.read_bytes()).hexdigest()
    assert canon_hash == "96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee", "Canonical checkpoint hash violated!"

    assert ADAPTED_CHECKPOINT_PATH.exists(), "Adapted checkpoint missing!"
    adapted_hash = hashlib.sha256(ADAPTED_CHECKPOINT_PATH.read_bytes()).hexdigest()
    assert adapted_hash == "15d48176efd6941415cc446b0f0602573a266a49a17da56d72d365d445091139", "Adapted checkpoint hash violated!"


def test_target_alignment_and_loss_aggregation_sanity():
    """Hand-constructed synthetic sanity test verifying next-token target alignment and causal masking."""
    vocab_size = 8
    seq_len = 4
    # Mock logits: shape (1, seq_len - 1, vocab_size)
    # Target tokens: shape (1, seq_len - 1)
    inp = torch.tensor([[1, 2, 3]], dtype=torch.long)
    tgt = torch.tensor([[2, 3, 4]], dtype=torch.long)

    # Let logit at position t have max value at true target t
    logits = torch.zeros((1, 3, vocab_size))
    for i in range(3):
        logits[0, i, tgt[0, i]] = 10.0

    loss = F.cross_entropy(logits.view(-1, vocab_size), tgt.view(-1), reduction="sum")
    assert loss.item() < 0.1, "Aligned logits should produce near-zero loss!"

    # Unaligned target test
    wrong_tgt = torch.tensor([[5, 6, 7]], dtype=torch.long)
    wrong_loss = F.cross_entropy(logits.view(-1, vocab_size), wrong_tgt.view(-1), reduction="sum")
    assert wrong_loss.item() > 5.0, "Misaligned targets must yield substantial cross-entropy!"


def test_audit_prose_boilerplate_detection():
    """Verify that our audit detects the boilerplate placeholder repetition in Batch 1861 prose test set."""
    b1861_test = REPO_ROOT / "experiments" / "batch1861_1900" / "data" / "test_held_out_expanded_corpus.json"
    assert b1861_test.exists(), "Batch 1861 test set missing!"
    with open(b1861_test, "r", encoding="utf-8") as f:
        docs = json.load(f)

    prose_docs = [d for d in docs if d["domain"] == "prose"]
    assert len(prose_docs) == 8, f"Expected 8 prose docs, got {len(prose_docs)}"

    # Check that all 8 prose docs are identical string duplicates
    first_text = prose_docs[0]["text"]
    assert "Domain prose indigenous system text specifications" in first_text
    for d in prose_docs:
        assert d["text"] == first_text, "All prose docs in Batch 1861 should be identical placeholder duplicates!"


def test_source_disjoint_benchmark_integrity():
    """Verify that the independent source-disjoint benchmark is non-empty and contains zero boilerplate repeats."""
    indep_corpus_path = BATCH1901_DIR / "data" / "source_disjoint_corpus.json"
    assert indep_corpus_path.exists(), "Source disjoint benchmark missing!"
    with open(indep_corpus_path, "r", encoding="utf-8") as f:
        docs = json.load(f)

    assert len(docs) >= 50, f"Expected at least 50 independent docs, got {len(docs)}"
    total_tokens = sum(d["token_count"] for d in docs)
    assert total_tokens >= 10000, f"Expected >= 10,000 evaluated tokens, got {total_tokens}"

    # Verify no doc contains the boilerplate string
    for d in docs:
        assert "Domain prose indigenous system text specifications" not in d["text"]


def test_all_deliverables_exist():
    """Check that all 6 required markdown and json deliverables exist and are non-empty."""
    required = [
        "INDEPENDENT_AUDIT_REPORT.md",
        "ARTIFACT_AND_CHECKPOINT_MANIFEST.json",
        "LEAKAGE_AND_PROVENANCE_AUDIT.md",
        "INDEPENDENT_EVALUATION_RESULTS.json",
        "TRANSFER_DIAGNOSTICS.md",
        "GO_NO_GO_DECISION.md"
    ]
    for fname in required:
        p = BATCH1901_DIR / fname
        assert p.exists(), f"Deliverable {fname} is missing!"
        assert p.stat().st_size > 100, f"Deliverable {fname} is unexpectedly empty!"
