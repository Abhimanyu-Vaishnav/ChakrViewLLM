"""Automated tests for Batches 1781-1820 neural root cause analysis, controlled interventions, and regression.
"""

import json
import os
import hashlib
import pytest

from experiments.batch1741_1780.spoken_number_parser import SpokenNumberParser
from chakrview.runtime.safe_calculator import SafeCalculator
from experiments.batch1781_1820.run_spoken_number_regression import run_spoken_number_regression


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


def test_controlled_experiments_results_artifacts():
    """Verify that all experimental results are recorded with complete keys and valid counts."""
    results_path = os.path.join(
        "experiments", "batch1781_1820", "results", "CONTROLLED_LEARNING_EXPERIMENT_RESULTS.json"
    )
    assert os.path.exists(results_path), "Controlled experiment results artifact missing!"

    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Check that TF-IDF baseline and all 3 interventions are recorded
    assert "tfidf_logistic_baseline" in data
    assert "intervention_a_adversarial_aware" in data
    assert "intervention_b_contrastive_margin" in data
    assert "intervention_c_attention_pooling" in data

    # Verify 3 random seeds are recorded for each neural intervention
    for k in ["intervention_a_adversarial_aware", "intervention_b_contrastive_margin", "intervention_c_attention_pooling"]:
        assert len(data[k]["seed_runs"]) == 3, f"Expected 3 seeds for {k}, got {len(data[k]['seed_runs'])}"


def test_spoken_number_expanded_regression():
    """Verify expanded spoken number regression suite runs with 100% correctness on safe calculator."""
    results = run_spoken_number_regression()
    assert results["valid_cases_passed"] == results["valid_cases_total"] == 18
    assert results["rejection_cases_passed"] == results["rejection_cases_total"] == 11
    assert results["valid_accuracy"] == 1.0
    assert results["rejection_accuracy"] == 1.0


def test_mission_deliverables_presence():
    """Verify all 9 required documentation deliverables exist."""
    required = [
        "ROOT_CAUSE_ANALYSIS.md",
        "LABEL_ONTOLOGY_AUDIT.md",
        "BASELINE_COMPARISON.md",
        "CONTROLLED_INTERVENTIONS.md",
        "ADVERSARIAL_FAILURE_ANALYSIS.md",
        "ARGUMENT_EXTRACTION_RESULTS.md",
        "SPOKEN_NUMBER_REGRESSION.md",
        "GO_NO_GO_DECISION.md",
        "FINAL_MISSION_REPORT.md",
    ]
    base_dir = os.path.join("experiments", "batch1781_1820")
    for doc in required:
        p = os.path.join(base_dir, doc)
        assert os.path.exists(p), f"Missing required deliverable: {p}"
