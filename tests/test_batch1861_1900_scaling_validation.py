"""Automated tests for Batches 1861-1900 scaling, validation and generalization study.
"""

import json
import os
import hashlib
import pytest
import torch

from chakrview.brain.model import ChakrMicro


def test_reference_checkpoints_immutability():
    """Verify canonical and domain-adapted checkpoints remain byte-for-byte untouched."""
    canonical_path = os.path.join(
        "experiments", "batch16_21_combined", "main_training_10000", "training_checkpoint_step_10000.pt"
    )
    adapted_path = os.path.join(
        "experiments", "batch1381_1420", "checkpoints", "chakrmicro_domain_adapted_experimental.pt"
    )

    expected_canonical_sha256 = "96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee"
    expected_adapted_sha256 = "15d48176efd6941415cc446b0f0602573a266a49a17da56d72d365d445091139"

    with open(canonical_path, "rb") as f:
        assert hashlib.sha256(f.read()).hexdigest() == expected_canonical_sha256

    with open(adapted_path, "rb") as f:
        assert hashlib.sha256(f.read()).hexdigest() == expected_adapted_sha256


def test_expanded_corpus_manifest_and_tokens():
    """Verify expanded corpus manifest exists, covers >10,000 prediction tokens and 48 held-out docs."""
    manifest_path = os.path.join("experiments", "batch1861_1900", "data", "EXPANDED_CORPUS_MANIFEST.json")
    assert os.path.exists(manifest_path), "Expanded corpus manifest missing!"

    with open(manifest_path, "r", encoding="utf-8") as f:
        m = json.load(f)

    assert m["split_counts"]["test_held_out"] == 48
    assert m["token_counts"]["test_held_out_tokens"] >= 10000
    assert len(m["domains"]) == 6


def test_expanded_experiments_results_artifacts():
    """Verify that expanded language modeling results artifact exists and records all 4 conditions."""
    results_path = os.path.join(
        "experiments", "batch1861_1900", "results", "EXPANDED_LANGUAGE_MODELING_RESULTS.json"
    )
    assert os.path.exists(results_path), "Expanded language modeling results missing!"

    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "canonical_baseline" in data
    assert "condition_b_continued_pretraining" in data
    assert "condition_c_content_skewed_control" in data
    assert "condition_d_token_budget_scaling" in data

    canon_loss = data["canonical_baseline"]["metrics"]["mean_cross_entropy"]
    cpt_loss = data["condition_b_continued_pretraining"]["mean_loss"]

    # Predeclared gate: reduction of at least 0.10
    delta = cpt_loss - canon_loss
    assert delta <= -0.10, f"Expected loss reduction <= -0.10, got {delta}"
    assert len(data["condition_b_continued_pretraining"]["runs"]) == 3, "Expected 3 seeds"


def test_mission_deliverables_presence():
    """Verify all 9 required documentation deliverables exist."""
    required = [
        "PREVIOUS_RESULT_REPRODUCTION.md",
        "EXPANDED_CORPUS_AND_PROVENANCE.md",
        "EVALUATION_PROTOCOL.md",
        "DATA_AND_TOKEN_BUDGET_COMPARISON.md",
        "CONTINUED_PRETRAINING_RESULTS.md",
        "PER_DOMAIN_GENERALIZATION.md",
        "PRACTICAL_TRANSFER_EVALUATION.md",
        "GO_NO_GO_DECISION.md",
        "FINAL_MISSION_REPORT.md",
    ]
    base_dir = os.path.join("experiments", "batch1861_1900")
    for doc in required:
        p = os.path.join(base_dir, doc)
        assert os.path.exists(p), f"Missing deliverable: {p}"
