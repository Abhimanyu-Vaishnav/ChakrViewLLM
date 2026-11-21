"""
Unit tests for UCIP-22 Benchmark Validator and Schemas.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.ucip22.validators.benchmark_validator import UCIP22BenchmarkValidator, create_exemplar_record


def test_validator_on_exemplar():
    validator = UCIP22BenchmarkValidator()
    rec = create_exemplar_record()
    valid, errors = validator.validate_record(rec)
    assert valid, f"Validation failed on valid exemplar: {errors}"
    print("Test Validator Exemplar: PASSED")


def test_validator_detects_length_exceeded():
    validator = UCIP22BenchmarkValidator()
    rec = create_exemplar_record()
    # artificially bloat prompt
    rec["prompt"] = rec["prompt"] * 20
    valid, errors = validator.validate_record(rec)
    assert not valid, "Validator should have failed on bloated sequence"
    assert any("exceeds max context limit" in e for e in errors)
    print("Test Validator Length Detection: PASSED")


def test_validator_detects_abstention_inconsistency():
    validator = UCIP22BenchmarkValidator()
    rec = create_exemplar_record()
    rec["is_abstention"] = True
    rec["expected_abstention_reason"] = "none" # contradictory
    valid, errors = validator.validate_record(rec)
    assert not valid, "Validator should fail on contradictory abstention"
    print("Test Validator Abstention Inconsistency: PASSED")


def main():
    print("Running UCIP-22 Benchmark Validator Unit Tests...")
    test_validator_on_exemplar()
    test_validator_detects_length_exceeded()
    test_validator_detects_abstention_inconsistency()
    print("All UCIP-22 Unit Tests PASSED!")


if __name__ == "__main__":
    main()
