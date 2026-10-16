"""Automated verification tests for Batches 1821-1860 foundational language modeling audit and continued pretraining.
"""

import json
import os
import hashlib
import pytest
import torch

from chakrview.brain.model import ChakrMicro


def test_checkpoint_parameter_count_resolution():
    """Verify exact 3,443,136 unique parameters and SHA-256 for canonical checkpoint."""
    canonical_path = os.path.join(
        "experiments", "batch16_21_combined", "main_training_10000", "training_checkpoint_step_10000.pt"
    )
    expected_sha256 = "96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee"

    with open(canonical_path, "rb") as f:
        assert hashlib.sha256(f.read()).hexdigest() == expected_sha256

    ckpt = torch.load(canonical_path, map_location="cpu", weights_only=False)
    model = ChakrMicro(ckpt["config"])
    model.load_state_dict(ckpt["model_state_dict"], strict=True)

    unique_params = sum(p.numel() for p in model.parameters())
    assert unique_params == 3443136, f"Expected exactly 3,443,136 unique parameters, got {unique_params}"

    serialized_elements = sum(v.numel() for v in ckpt["model_state_dict"].values())
    assert serialized_elements == 4229568, f"Expected 4,229,568 serialized tensor elements, got {serialized_elements}"


def test_corpus_manifest_integrity():
    """Verify multi-domain corpus manifest exists and contains 60 documents across 6 domains."""
    manifest_path = os.path.join("experiments", "batch1821_1860", "data", "CORPUS_MANIFEST.json")
    assert os.path.exists(manifest_path), "Corpus manifest missing!"

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest["total_documents"] == 60
    assert manifest["split_counts"]["train"] == 30
    assert manifest["split_counts"]["val"] == 12
    assert manifest["split_counts"]["test_held_out"] == 18
    assert len(manifest["domains"]) == 6


def test_language_modeling_results_artifacts():
    """Verify that language modeling experiment results artifact exists and confirms loss reduction."""
    results_path = os.path.join(
        "experiments", "batch1821_1860", "results", "LANGUAGE_MODELING_EXPERIMENT_RESULTS.json"
    )
    assert os.path.exists(results_path), "Language modeling results missing!"

    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "canonical_baseline" in data
    assert "continued_pretraining_summary" in data
    assert "data_matched_control" in data

    canon_loss = data["canonical_baseline"]["held_out_metrics"]["mean_cross_entropy"]
    cpt_loss = data["continued_pretraining_summary"]["mean_held_out_loss"]

    # Predeclared threshold: Loss reduction of at least 0.10
    delta = cpt_loss - canon_loss
    assert delta <= -0.10, f"Expected loss reduction <= -0.10, got {delta}"
    assert len(data["continued_pretraining_summary"]["runs"]) == 3, "Expected 3 random seeds"


def test_mission_deliverables_presence():
    """Verify all 9 required documentation deliverables exist."""
    required = [
        "ARCHITECTURE_AND_CHECKPOINT_AUDIT.md",
        "CORPUS_AND_TOKENIZER_AUDIT.md",
        "BASELINE_LANGUAGE_MODEL_EVALUATION.md",
        "CONTINUED_PRETRAINING_PROTOCOL.md",
        "CONTINUED_PRETRAINING_RESULTS.md",
        "GENERALIZATION_AND_TRANSFER.md",
        "OBJECTIVE_RESEARCH.md",
        "GO_NO_GO_DECISION.md",
        "FINAL_MISSION_REPORT.md",
    ]
    base_dir = os.path.join("experiments", "batch1821_1860")
    for doc in required:
        p = os.path.join(base_dir, doc)
        assert os.path.exists(p), f"Missing deliverable: {p}"
