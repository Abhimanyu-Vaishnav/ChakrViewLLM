#!/usr/bin/env python3
"""
Unit and regression tests for Batches 1981–2020:
Independent Statistical Reproduction, Corpus Quality Gate & Language Coverage Audit
"""

import hashlib
import json
import random
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
BATCH1981_DIR = REPO_ROOT / "experiments" / "batch1981_2020"
BATCH1941_DIR = REPO_ROOT / "experiments" / "batch1941_1980"
CANONICAL_CHECKPOINT_PATH = REPO_ROOT / "experiments" / "batch16_21_combined" / "main_training_10000" / "training_checkpoint_step_10000.pt"
ADAPTED_CHECKPOINT_PATH = REPO_ROOT / "experiments" / "batch1381_1420" / "checkpoints" / "chakrmicro_domain_adapted_experimental.pt"


def test_canonical_and_adapted_checkpoint_integrity():
    """Verify Canonical and Adapted reference checkpoints remain strictly immutable."""
    assert CANONICAL_CHECKPOINT_PATH.exists(), "Canonical checkpoint missing!"
    canon_hash = hashlib.sha256(CANONICAL_CHECKPOINT_PATH.read_bytes()).hexdigest()
    assert canon_hash == "96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee", "Canonical checkpoint hash violated!"

    assert ADAPTED_CHECKPOINT_PATH.exists(), "Adapted checkpoint missing!"
    adapted_hash = hashlib.sha256(ADAPTED_CHECKPOINT_PATH.read_bytes()).hexdigest()
    assert adapted_hash == "15d48176efd6941415cc446b0f0602573a266a49a17da56d72d365d445091139", "Adapted checkpoint hash violated!"


def test_reproduced_metrics_consistency():
    """Verify that all reproduced metrics and confidence intervals are mathematically consistent."""
    metrics_path = BATCH1981_DIR / "REPRODUCED_METRICS.json"
    assert metrics_path.exists(), "REPRODUCED_METRICS.json missing!"

    with open(metrics_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Check candidates
    for cand_name, metrics in data["candidates"].items():
        tw_delta = metrics["token_weighted_delta"]
        macro_delta = metrics["document_macro_delta"]

        doc_tw_ci = metrics["document_level_bootstrap_95ci"]["token_weighted"]
        doc_macro_ci = metrics["document_level_bootstrap_95ci"]["macro_document"]

        # Point estimates must lie inside their respective confidence intervals
        assert doc_tw_ci[0] <= tw_delta <= doc_tw_ci[1], (
            f"{cand_name} token-weighted delta {tw_delta} not in {doc_tw_ci}"
        )
        assert doc_macro_ci[0] <= macro_delta <= doc_macro_ci[1], (
            f"{cand_name} macro delta {macro_delta} not in {doc_macro_ci}"
        )

        src_tw_ci = metrics["source_clustered_bootstrap_95ci"]["token_weighted"]
        src_macro_ci = metrics["source_clustered_bootstrap_95ci"]["macro_document"]

        assert src_tw_ci[0] <= tw_delta <= src_tw_ci[1], (
            f"{cand_name} token-weighted delta {tw_delta} not in source CI {src_tw_ci}"
        )
        assert src_macro_ci[0] <= macro_delta <= src_macro_ci[1], (
            f"{cand_name} macro delta {macro_delta} not in source CI {src_macro_ci}"
        )


def test_corpus_quality_gate_checks():
    """Verify document counts, absence of exact duplicates, and source concentration in staged natural corpus."""
    manifest_p = BATCH1941_DIR / "data" / "CORPUS_MANIFEST.json"
    assert manifest_p.exists(), "CORPUS_MANIFEST.json missing!"

    with open(manifest_p, "r", encoding="utf-8") as f:
        corpus = json.load(f)

    docs = corpus["documents"]
    assert len(docs) == 2619, f"Expected exactly 2,619 documents, got {len(docs)}"

    # Check for zero duplicate documents
    seen_hashes = set()
    for d in docs:
        h = d["sha256"]
        assert h not in seen_hashes, f"Found duplicate document hash: {h}"
        seen_hashes.add(h)

    # Verify that test split concentration is accurately flagged
    test_docs = [d for d in docs if d["split"] == "test"]
    loney_docs = [d for d in test_docs if d["source_id"] == "gutenberg_loney_coordinate_geometry"]
    concentration = len(loney_docs) / len(test_docs)
    assert concentration > 0.99, f"Expected Loney geometry to constitute >99% of test docs, got {concentration:.3f}"


def test_all_batch1981_deliverables_exist():
    """Check that all required deliverables exist and are non-empty."""
    required = [
        "STATISTICAL_REPRODUCTION.md",
        "REPRODUCED_METRICS.json",
        "SOURCE_CLUSTER_UNCERTAINTY.md",
        "CORPUS_QUALITY_AUDIT.md",
        "PROVENANCE_AND_LICENSE_AUDIT.md",
        "LANGUAGE_COVERAGE_AUDIT.md",
        "TRAINING_READINESS_DECISION.md",
        "run_reproduction.py"
    ]
    for fname in required:
        p = BATCH1981_DIR / fname
        assert p.exists(), f"Deliverable {fname} is missing!"
        assert p.stat().st_size > 100, f"Deliverable {fname} is unexpectedly empty!"
