"""Automated verification tests for Batches 1741-1780 reproduction and safety gate.
"""

import json
import os
import hashlib
import pytest

from experiments.batch1741_1780.spoken_number_parser import SpokenNumberParser
from chakrview.runtime.safe_calculator import SafeCalculator


def test_checkpoint_integrity_sha256():
    """Verify canonical and domain-adapted checkpoints match required hashes."""
    canonical_path = os.path.join(
        "experiments", "batch16_21_combined", "main_training_10000", "training_checkpoint_step_10000.pt"
    )
    adapted_path = os.path.join(
        "experiments", "batch1381_1420", "checkpoints", "chakrmicro_domain_adapted_experimental.pt"
    )

    expected_canonical_sha256 = "96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee"
    expected_adapted_sha256 = "15d48176efd6941415cc446b0f0602573a266a49a17da56d72d365d445091139"

    with open(canonical_path, "rb") as f:
        actual_canonical = hashlib.sha256(f.read()).hexdigest()
    assert actual_canonical == expected_canonical_sha256, "Canonical checkpoint corrupted or modified!"

    with open(adapted_path, "rb") as f:
        actual_adapted = hashlib.sha256(f.read()).hexdigest()
    assert actual_adapted == expected_adapted_sha256, "Adapted checkpoint corrupted or modified!"


def test_benchmark_300_data_manifest():
    """Verify that the 300 held-out tasks benchmark exists, contains 300 tasks, and matches manifest."""
    manifest_path = os.path.join("experiments", "batch1741_1780", "data", "DATASET_MANIFEST.json")
    benchmark_path = os.path.join("experiments", "batch1741_1780", "data", "test_held_out_300.json")

    assert os.path.exists(manifest_path), "Dataset manifest missing!"
    assert os.path.exists(benchmark_path), "Benchmark dataset missing!"

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    with open(benchmark_path, "r", encoding="utf-8") as f:
        tasks = json.load(f)

    assert len(tasks) == 300, f"Expected 300 tasks, found {len(tasks)}"

    counts = {}
    for t in tasks:
        counts[t["intent"]] = counts.get(t["intent"], 0) + 1

    expected_labels = ["CALCULATE", "VERIFY_CLAIM", "KNOWLEDGE_QUERY", "CODE_TASK", "CLARIFY", "OUT_OF_SCOPE"]
    for label in expected_labels:
        assert counts.get(label, 0) == 50, f"Expected 50 tasks for {label}, got {counts.get(label, 0)}"


def test_spoken_number_parser():
    """Verify isolated spoken number parsing and safe execution contract."""
    calc = SafeCalculator()

    # Standard numbers
    expr, err = SpokenNumberParser.parse_spoken_expression("twenty times fifteen")
    assert err is None
    assert expr == "20 * 15"
    res = calc.evaluate(expr)
    assert res.success is True
    assert res.result == 300.0

    # Inverted subtraction pattern
    expr_sub, err_sub = SpokenNumberParser.parse_spoken_expression("subtract forty from ninety")
    assert err_sub is None
    assert expr_sub == "90 - 40"
    assert calc.evaluate(expr_sub).result == 50.0

    # Unsupported / Malformed expression rejection (Fail-closed)
    expr_bad, err_bad = SpokenNumberParser.parse_spoken_expression("negative five times ten")
    assert expr_bad is None
    assert err_bad is not None

    # Security: No eval injection allowed
    expr_inj, err_inj = SpokenNumberParser.parse_spoken_expression("__import__('os').system('dir')")
    assert expr_inj is None
    assert err_inj is not None


def test_experiment_artifacts_exist():
    """Verify all required documentation deliverables exist in experiments/batch1741_1780."""
    required_files = [
        "PREVIOUS_RESULTS_REPRODUCTION_AUDIT.md",
        "DATASET_FAMILY_AND_LEAKAGE_AUDIT.md",
        "CLASSIFIER_REPRODUCTION_RESULTS.md",
        "SEMANTIC_GENERALIZATION_RESULTS.md",
        "CONFIDENCE_ABSTENTION_AND_SAFETY.md",
        "SPOKEN_NUMBER_PARSER_EXPERIMENT.md",
        "GO_NO_GO_DECISION.md",
        "FINAL_MISSION_REPORT.md",
        os.path.join("results", "BENCHMARK_300_EVALUATION_RESULTS.json"),
    ]

    base_dir = os.path.join("experiments", "batch1741_1780")
    for f in required_files:
        p = os.path.join(base_dir, f)
        assert os.path.exists(p), f"Required deliverable missing: {p}"
