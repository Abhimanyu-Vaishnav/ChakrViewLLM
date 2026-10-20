"""
Tests for Intelligence Suite Evaluation Harness, False-Positive Scoring Prevention,
Missing Checkpoint Handling, Task Coverage, and Functional Code Evaluation.
"""
import pytest
import tempfile
import json
from pathlib import Path

from evaluation.intelligence_suite.scoring_evaluator import evaluate_response
from evaluation.intelligence_suite.run_intelligence_benchmark import run_benchmark


def test_scoring_prevents_false_positive_substring_match():
    # Number match: "7" should NOT match "17" or "27"
    passed, reason = evaluate_response("There were 17 items found", "7", "number_match")
    assert not passed
    assert "NO_WORD_BOUNDARY_MATCH" in reason

    # But "7" should match "There were 7 items"
    passed, reason = evaluate_response("There were 7 items", "7", "number_match")
    assert passed
    assert reason == "BOUNDARY_MATCH"


def test_scoring_prevents_false_positive_word_boundary():
    # "apple" should NOT match "pineapple"
    passed, reason = evaluate_response("I like pineapple juice", "apple", "word_boundary")
    assert not passed
    assert "NO_WORD_BOUNDARY_MATCH" in reason

    # "apple" SHOULD match "I ate an apple today"
    passed, reason = evaluate_response("I ate an apple today", "apple", "word_boundary")
    assert passed
    assert reason == "BOUNDARY_MATCH"


def test_functional_code_scoring_evaluation():
    # Code passes assertion
    gen_code = "def add(a, b):\n    return a + b\n"
    test_code = "assert add(2, 3) == 5"
    passed, reason = evaluate_response(gen_code, "a + b", "code_functional", test_code=test_code)
    assert passed
    assert reason == "FUNCTIONAL_TEST_PASSED"

    # Code fails assertion
    failing_code = "def add(a, b):\n    return a - b\n"
    passed, reason = evaluate_response(failing_code, "a + b", "code_functional", test_code=test_code)
    assert not passed
    assert "ASSERTION_FAILED" in reason


def test_smoke_mode_demarcation():
    # Run in smoke mode; ensure mode is smoke_verification and pass rate is 0/32
    results = run_benchmark(smoke_mode=True)
    assert results["mode"] == "smoke_verification"
    assert results["passed_tasks_count"] == 0
    assert results["checkpoint_present"] is False
    assert results["total_tasks"] == 32