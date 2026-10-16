#!/usr/bin/env python3
"""
Unit and regression tests for Batches 1941–1980:
Statistical Audit, Natural Corpus Construction & Training Readiness
"""

import hashlib
import json
import random
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
BATCH1941_DIR = REPO_ROOT / "experiments" / "batch1941_1980"
CANONICAL_CHECKPOINT_PATH = REPO_ROOT / "experiments" / "batch16_21_combined" / "main_training_10000" / "training_checkpoint_step_10000.pt"
ADAPTED_CHECKPOINT_PATH = REPO_ROOT / "experiments" / "batch1381_1420" / "checkpoints" / "chakrmicro_domain_adapted_experimental.pt"


def test_checkpoint_immutability():
    """Verify Canonical and Adapted reference checkpoints remain strictly immutable."""
    assert CANONICAL_CHECKPOINT_PATH.exists(), "Canonical checkpoint missing!"
    canon_hash = hashlib.sha256(CANONICAL_CHECKPOINT_PATH.read_bytes()).hexdigest()
    assert canon_hash == "96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee", "Canonical checkpoint hash violated!"

    assert ADAPTED_CHECKPOINT_PATH.exists(), "Adapted checkpoint missing!"
    adapted_hash = hashlib.sha256(ADAPTED_CHECKPOINT_PATH.read_bytes()).hexdigest()
    assert adapted_hash == "15d48176efd6941415cc446b0f0602573a266a49a17da56d72d365d445091139", "Adapted checkpoint hash violated!"


def test_bootstrap_toy_dataset_sanity():
    """Deterministic synthetic test verifying that point estimate lies inside bootstrap CI."""
    rng = random.Random(1337)
    # 50 paired documents with known candidate improvement
    # candidate loss = canon loss - 0.25 + small noise
    canon_losses = [5.0 + rng.random() for _ in range(50)]
    cand_losses = [l - 0.25 + (rng.random() - 0.5) * 0.1 for l in canon_losses]

    deltas = [c - b for c, b in zip(cand_losses, canon_losses)]
    point_estimate = sum(deltas) / len(deltas)

    # 1,000 bootstrap resamples
    boot_means = []
    n = len(deltas)
    for _ in range(1000):
        sample = [rng.choice(deltas) for _ in range(n)]
        boot_means.append(sum(sample) / n)

    boot_means.sort()
    ci_lower = boot_means[int(0.025 * 1000)]
    ci_upper = boot_means[int(0.975 * 1000)]

    assert ci_lower <= point_estimate <= ci_upper, (
        f"Point estimate {point_estimate} must lie inside 95% CI [{ci_lower}, {ci_upper}]"
    )
    assert ci_upper < 0.0, "Expected significant improvement below zero!"


def test_corrected_confidence_intervals_consistency():
    """Verify that all corrected confidence intervals contain their point estimates."""
    ci_file = BATCH1941_DIR / "CORRECTED_CONFIDENCE_INTERVALS.json"
    assert ci_file.exists(), "CORRECTED_CONFIDENCE_INTERVALS.json missing!"
    with open(ci_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Check macro document estimand
    for model_name, st in data["macro_document_estimand"].items():
        pe = st["point_estimate_macro_delta"]
        lo = st["ci_95_lower"]
        hi = st["ci_95_upper"]
        assert lo <= pe <= hi, f"{model_name} macro delta {pe} not inside [{lo}, {hi}]"
        assert st["point_estimate_inside_ci"] is True

    # Check token-weighted estimand
    for model_name, st in data["token_weighted_estimand"].items():
        pe = st["point_estimate_token_weighted_delta"]
        lo = st["ci_95_lower"]
        hi = st["ci_95_upper"]
        assert lo <= pe <= hi, f"{model_name} token-weighted delta {pe} not inside [{lo}, {hi}]"
        assert st["point_estimate_inside_ci"] is True


def test_natural_corpus_split_isolation():
    """Verify strict source-level group isolation across train, validation, and test splits."""
    corpus_manifest_p = BATCH1941_DIR / "data" / "CORPUS_MANIFEST.json"
    split_manifest_p = BATCH1941_DIR / "data" / "SPLIT_MANIFEST.json"
    assert corpus_manifest_p.exists(), "CORPUS_MANIFEST.json missing!"
    assert split_manifest_p.exists(), "SPLIT_MANIFEST.json missing!"

    with open(corpus_manifest_p, "r", encoding="utf-8") as f:
        corpus = json.load(f)
    with open(split_manifest_p, "r", encoding="utf-8") as f:
        splits = json.load(f)

    train_sources = set(splits["splits"]["train"]["sources"])
    val_sources = set(splits["splits"]["validation"]["sources"])
    test_sources = set(splits["splits"]["test"]["sources"])

    # Disjoint check across source sets
    assert train_sources.isdisjoint(val_sources), "Train and Val sources must be strictly disjoint!"
    assert train_sources.isdisjoint(test_sources), "Train and Test sources must be strictly disjoint!"
    assert val_sources.isdisjoint(test_sources), "Val and Test sources must be strictly disjoint!"

    # Check that documents strictly adhere to their assigned split
    for doc in corpus["documents"]:
        sid = doc["source_id"]
        sp = doc["split"]
        if sid in train_sources:
            assert sp == "train", f"Doc {doc['document_id']} from source {sid} assigned to {sp} instead of train!"
        elif sid in val_sources:
            assert sp == "validation", f"Doc {doc['document_id']} from source {sid} assigned to {sp} instead of validation!"
        elif sid in test_sources:
            assert sp == "test", f"Doc {doc['document_id']} from source {sid} assigned to {sp} instead of test!"

    # Verify no fallback loop placeholder repetition exists
    for doc in corpus["documents"]:
        assert "Domain prose indigenous system text specifications" not in doc["text"], "Found prohibited placeholder text!"


def test_all_batch1941_deliverables_exist():
    """Verify all 7 required deliverables are present and populated."""
    deliverables = [
        "STATISTICAL_AUDIT.md",
        "CORRECTED_CONFIDENCE_INTERVALS.json",
        "SOURCE_DISJOINT_CORPUS_AUDIT.md",
        "NATURAL_CORPUS_PROVENANCE.md",
        "data/CORPUS_MANIFEST.json",
        "data/SPLIT_MANIFEST.json",
        "TRAINING_READINESS_DECISION.md"
    ]
    for d in deliverables:
        p = BATCH1941_DIR / d
        assert p.exists(), f"Deliverable {d} is missing!"
        assert p.stat().st_size > 100, f"Deliverable {d} is unexpectedly empty!"
