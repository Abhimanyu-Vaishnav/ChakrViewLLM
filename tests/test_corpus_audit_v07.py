"""
Unit tests for Milestone 2: Corpus quality and leakage analysis audit logic.
"""

from pathlib import Path
import pytest
from scripts.audit_corpus_quality import run_v07_corpus_audit


def test_v07_corpus_audit_document_disjointness():
    repo_root = Path(".")
    report = run_v07_corpus_audit(repo_root)

    # Document disjointness verification
    doc_leakage = report["document_level_leakage"]
    assert doc_leakage["is_document_disjoint"] is True
    assert doc_leakage["train_val_id_overlap"] == 0
    assert doc_leakage["train_test_id_overlap"] == 0
    assert doc_leakage["train_val_content_hash_overlap"] == 0
    assert doc_leakage["train_test_content_hash_overlap"] == 0


def test_v07_corpus_audit_shard_integrity_and_bounds():
    repo_root = Path(".")
    report = run_v07_corpus_audit(repo_root)

    # All shards match expected totals and hashes
    for split in ["train", "validation", "test"]:
        assert report["shard_integrity"][split]["match"] is True
        assert report["shard_integrity"][split]["hash_match_all"] is True

    # Vocabulary bounds strictly in [0, 4095]
    bounds = report["token_bounds_audit"]
    assert bounds["shard_errors"] == []
    assert bounds["all_within_bounds"] is True
    assert bounds["train_min_token"] >= 0
    assert bounds["train_max_token"] < 4096
    assert bounds["test_max_token"] < 4096
