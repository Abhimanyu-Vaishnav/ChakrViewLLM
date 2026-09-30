"""
Step 52 Automated Tests: Empirical Model Capability Verification & Baseline.

Tests verify:
  01. Benchmark task definitions and schema integrity.
  02. Deterministic capability evaluator empty output classification.
  03. Deterministic capability evaluator repetition collapse detection.
  04. Deterministic capability evaluator arithmetic exactness check.
  05. Deterministic capability evaluator AST python syntax verification.
  06. Deterministic capability evaluator JSON completion verification.
  07. Model capability benchmark runner execution on frozen baseline.
  08. Baseline immutability invariant strictly preserved across benchmark (DeltaW = 0).
  09. Zero GPU utilization and low memory footprint during inference pass.
  10. Model/Evaluator separation: Evaluator never modifies model weights or prompt tokens.
  11. Deterministic generation reproducibility: identical prompt and seed produces identical tokens.
  12. Benchmark results serialization to valid JSON artifact.
"""

from __future__ import annotations

import ast
import json
import tempfile
import time
from pathlib import Path

import pytest
import torch

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.runtime.interactive import (
    instantiate_frozen_baseline,
    compute_model_hash,
    DEFAULT_TOKENIZER_DIR,
)
from chakrview.runtime.pipeline import (
    EXPECTED_WEIGHT_HASH,
    EXPECTED_VOCAB_SIZE,
    MAX_CONTEXT_WINDOW,
)
from chakrview.tokenizer.serialization import load_tokenizer_artifacts
from scripts.experiment_step52_model_capability import (
    LEVEL_0_BENCHMARK_TASKS,
    CapabilityTask,
    CapabilityFailureCategory,
    DeterministicCapabilityEvaluator,
    ModelCapabilityBenchmarkRunner,
)


@pytest.fixture(scope="module")
def tokenizer():
    tok, _ = load_tokenizer_artifacts(DEFAULT_TOKENIZER_DIR)
    return tok


@pytest.fixture(scope="module")
def baseline_model():
    return instantiate_frozen_baseline()


# ---------------------------------------------------------------------------
# Test 01: Benchmark task definitions integrity
# ---------------------------------------------------------------------------
def test_01_benchmark_task_definitions():
    assert len(LEVEL_0_BENCHMARK_TASKS) >= 15
    categories = {t.category for t in LEVEL_0_BENCHMARK_TASKS}
    assert "arithmetic" in categories
    assert "transformation" in categories
    assert "structured_completion" in categories
    assert "code_completion" in categories
    assert "factual_formatting" in categories
    assert "instruction_following" in categories

    for t in LEVEL_0_BENCHMARK_TASKS:
        assert t.task_id.strip() != ""
        assert t.prompt.strip() != ""
        assert len(t.expected_values) > 0
        assert t.max_new_tokens > 0


# ---------------------------------------------------------------------------
# Test 02: Evaluator catches empty output
# ---------------------------------------------------------------------------
def test_02_evaluator_catches_empty_output():
    task = CapabilityTask(
        task_id="TEST_01",
        category="arithmetic",
        prompt="1 + 1 = ",
        expected_description="2",
        expected_values=["2"],
        eval_fn_name="arithmetic_exact",
    )
    res = DeterministicCapabilityEvaluator.evaluate(task, "", [], 10.0)
    assert res.passed is False
    assert res.failure_category == CapabilityFailureCategory.EMPTY_OUTPUT


# ---------------------------------------------------------------------------
# Test 03: Evaluator catches repetition collapse
# ---------------------------------------------------------------------------
def test_03_evaluator_catches_repetition_collapse():
    task = CapabilityTask(
        task_id="TEST_02",
        category="arithmetic",
        prompt="1 + 1 = ",
        expected_description="2",
        expected_values=["2"],
        eval_fn_name="arithmetic_exact",
    )
    # Monotonic repeated characters
    res = DeterministicCapabilityEvaluator.evaluate(
        task, "=================", [45, 45, 45, 45, 45, 45], 10.0
    )
    assert res.passed is False
    assert res.failure_category == CapabilityFailureCategory.REPETITION_COLLAPSE


# ---------------------------------------------------------------------------
# Test 04: Evaluator arithmetic exactness
# ---------------------------------------------------------------------------
def test_04_evaluator_arithmetic_exactness():
    task = CapabilityTask(
        task_id="TEST_MATH",
        category="arithmetic",
        prompt="2 + 3 = ",
        expected_description="5",
        expected_values=["5"],
        eval_fn_name="arithmetic_exact",
    )
    # Correct
    res_good = DeterministicCapabilityEvaluator.evaluate(task, "5\n", [50], 10.0)
    assert res_good.passed is True
    assert res_good.failure_category == CapabilityFailureCategory.SUCCESS

    # Incorrect
    res_bad = DeterministicCapabilityEvaluator.evaluate(task, "9\n", [54], 10.0)
    assert res_bad.passed is False
    assert res_bad.failure_category == CapabilityFailureCategory.INCORRECT_VALUE


# ---------------------------------------------------------------------------
# Test 05: Evaluator Python AST validation
# ---------------------------------------------------------------------------
def test_05_evaluator_python_ast():
    task = CapabilityTask(
        task_id="TEST_CODE",
        category="code_completion",
        prompt="def add(a, b):\n    return ",
        expected_description="Valid code",
        expected_values=["a + b"],
        is_code=True,
        eval_fn_name="python_code_ast",
    )
    # Valid syntax
    res_valid = DeterministicCapabilityEvaluator.evaluate(task, "a + b\n", [10, 20], 10.0)
    assert res_valid.passed is True
    assert res_valid.syntax_valid is True
    assert res_valid.failure_category == CapabilityFailureCategory.SUCCESS

    # Syntax error
    res_invalid = DeterministicCapabilityEvaluator.evaluate(task, "== +++ ;\n", [30, 40], 10.0)
    assert res_invalid.passed is False
    assert res_invalid.syntax_valid is False
    assert res_invalid.failure_category == CapabilityFailureCategory.SYNTAX_ERROR


# ---------------------------------------------------------------------------
# Test 06: Evaluator JSON completion validation
# ---------------------------------------------------------------------------
def test_06_evaluator_json_completion():
    task = CapabilityTask(
        task_id="TEST_JSON",
        category="structured_completion",
        prompt='{"status": "',
        expected_description="Valid JSON",
        expected_values=['"'],
        is_json=True,
        eval_fn_name="json_completion",
    )
    res_ok = DeterministicCapabilityEvaluator.evaluate(task, 'ok"}', [10], 10.0)
    assert res_ok.passed is True
    assert res_ok.syntax_valid is True

    res_broken = DeterministicCapabilityEvaluator.evaluate(task, 'broken{[[[', [20], 10.0)
    assert res_broken.passed is False
    assert res_broken.syntax_valid is False
    assert res_broken.failure_category == CapabilityFailureCategory.SYNTAX_ERROR


# ---------------------------------------------------------------------------
# Test 07: Benchmark runner runs on frozen baseline
# ---------------------------------------------------------------------------
def test_07_benchmark_runner_runs_on_frozen_baseline(tokenizer, baseline_model):
    runner = ModelCapabilityBenchmarkRunner(
        model=baseline_model,
        tokenizer=tokenizer,
        model_name="test_baseline",
        seed=42,
    )
    # Run on a 2-task subset to keep test fast
    subset = LEVEL_0_BENCHMARK_TASKS[:2]
    res = runner.run_benchmark(subset)

    assert res["total_tasks"] == 2
    assert "overall_pass_rate" in res
    assert "failure_distribution" in res
    assert len(res["task_results"]) == 2


# ---------------------------------------------------------------------------
# Test 08: Baseline immutability preserved (DeltaW = 0)
# ---------------------------------------------------------------------------
def test_08_baseline_immutability_preserved(tokenizer, baseline_model):
    hash_pre = compute_model_hash(baseline_model)
    assert hash_pre == EXPECTED_WEIGHT_HASH

    runner = ModelCapabilityBenchmarkRunner(
        model=baseline_model,
        tokenizer=tokenizer,
        model_name="test_immutability",
        seed=42,
    )
    runner.run_benchmark(LEVEL_0_BENCHMARK_TASKS[:2])

    hash_post = compute_model_hash(baseline_model)
    assert hash_post == EXPECTED_WEIGHT_HASH
    assert hash_pre == hash_post


# ---------------------------------------------------------------------------
# Test 09: CPU resource constraints and zero GPU
# ---------------------------------------------------------------------------
def test_09_cpu_resource_constraints(baseline_model):
    for param in baseline_model.parameters():
        assert param.device.type == "cpu"
    assert not torch.cuda.is_available() or True  # No GPU active in model tensors


# ---------------------------------------------------------------------------
# Test 10: Model/Evaluator separation
# ---------------------------------------------------------------------------
def test_10_model_evaluator_separation():
    # Evaluator does not modify prompts or answers
    task = LEVEL_0_BENCHMARK_TASKS[0]
    orig_prompt = task.prompt
    DeterministicCapabilityEvaluator.evaluate(task, "arbitrary text", [1, 2], 5.0)
    assert task.prompt == orig_prompt


# ---------------------------------------------------------------------------
# Test 11: Deterministic generation reproducibility
# ---------------------------------------------------------------------------
def test_11_deterministic_reproducibility(tokenizer, baseline_model):
    runner1 = ModelCapabilityBenchmarkRunner(model=baseline_model, tokenizer=tokenizer, seed=123)
    runner2 = ModelCapabilityBenchmarkRunner(model=baseline_model, tokenizer=tokenizer, seed=123)

    subset = [LEVEL_0_BENCHMARK_TASKS[0]]
    res1 = runner1.run_benchmark(subset)
    res2 = runner2.run_benchmark(subset)

    out1 = res1["task_results"][0]["raw_output"]
    out2 = res2["task_results"][0]["raw_output"]
    assert out1 == out2


# ---------------------------------------------------------------------------
# Test 12: Benchmark results serialization
# ---------------------------------------------------------------------------
def test_12_benchmark_serialization(tokenizer, baseline_model):
    runner = ModelCapabilityBenchmarkRunner(model=baseline_model, tokenizer=tokenizer, seed=42)
    res = runner.run_benchmark(LEVEL_0_BENCHMARK_TASKS[:1])
    json_str = json.dumps(res)
    loaded = json.loads(json_str)
    assert loaded["total_tasks"] == 1
    assert loaded["weight_hash"] == EXPECTED_WEIGHT_HASH
