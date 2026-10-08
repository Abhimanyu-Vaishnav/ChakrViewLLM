"""
Test suite for Waves 566-570: Grounding Audit, State Causality & Independent Evaluation.
"""

import pytest
import torch
from chakrview.brain.canonical_hash import verify_canonical_baseline
from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro

from chakrview.cognition.grounding_audit.state_causality import (
    CognitiveStateCausalityTracker, CausalityLevel
)
from chakrview.cognition.grounding_audit.stress_dataset import (
    build_100_case_grounding_benchmark
)
from chakrview.cognition.grounding_audit.independent_evaluator import (
    IndependentGroundingEvaluator
)


def test_canonical_baseline_hash_immutability():
    torch.manual_seed(42)
    m = ChakrMicro(ModelConfig())
    res = verify_canonical_baseline(m)
    assert res["is_bit_exact"] is True
    assert res["delta_w"] == 0


def test_wave566_level_3_cognitive_state_causality():
    trace = CognitiveStateCausalityTracker.run_full_causality_trace()
    assert trace.is_state_causality_established is True
    assert trace.observed_causality_level == CausalityLevel.LEVEL_3_CAUSAL_GROUNDING
    # Ensure intermediate state variables participate in reasoning
    assert "entity_binding" in trace.variables_participating_in_reasoning
    assert "epistemic_state" in trace.variables_participating_in_reasoning
    assert "reasoning_strategy" in trace.variables_participating_in_reasoning


def test_wave568_569_independent_100_case_benchmark():
    benchmark = build_100_case_grounding_benchmark()
    assert benchmark.total_cases == 100

    report = IndependentGroundingEvaluator.evaluate_benchmark(benchmark, partition_to_eval="FINAL_EVALUATION")
    assert report.all_passed_thresholds is True
    assert report.metrics["grounded_accuracy"].percentage == 100.0
    assert report.metrics["unsupported_claim_rate"].percentage == 0.0
    assert report.metrics["wrong_entity_rejection"].percentage == 100.0
    assert report.metrics["overclaim_prevention"].percentage == 100.0
