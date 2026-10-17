"""
Test suite for Batch 2181-2220: Adaptive Hardware Runtime & Verified Tool Router.
"""

import os
import pytest
import torch

from experiments.batch2181_2220.adaptive_runtime_engine import AdaptiveExecutionPolicy
from experiments.batch2181_2220.verified_tool_router import (
    VerifiedToolRouter,
    evaluate_arithmetic_probes_with_tool
)
from chakrview.runtime.safe_calculator import SafeCalculator


def test_adaptive_execution_policy():
    """Verify adaptive execution policy correctly detects hardware and selects device."""
    policy = AdaptiveExecutionPolicy()
    status = policy.get_hardware_status()
    assert "cpu_cores" in status
    assert "cuda_available" in status
    assert status["cpu_cores"] > 0
    
    # Batch 1 should select CPU to avoid PCI-e launch latency
    dev_small = policy.select_device(batch_size=1, seq_len=128, is_training=False)
    assert dev_small.type == "cpu"
    
    # Training workload should select CUDA if available
    dev_train = policy.select_device(batch_size=4, seq_len=256, is_training=True)
    if torch.cuda.is_available():
        assert dev_train.type == "cuda"
    else:
        assert dev_train.type == "cpu"


def test_safe_calculator_ast_security():
    """Verify that SafeCalculator strictly evaluates arithmetic and rejects malicious code."""
    calc = SafeCalculator()
    
    # Valid calculations
    res1 = calc.execute(expression="144 / 12")
    assert res1.success
    assert res1.output == 12
    
    res2 = calc.execute(expression="25 * 4 + 10")
    assert res2.success
    assert res2.output == 110
    
    # Rejection of system execution / imports
    res_malicious = calc.execute(expression="__import__('os').system('ls')")
    assert not res_malicious.success
    assert "Security violation" in res_malicious.error or "Disallowed" in res_malicious.error
    
    # Division by zero handled safely
    res_div0 = calc.execute(expression="100 / 0")
    assert not res_div0.success
    assert "Division" in res_div0.error


def test_verified_tool_router_extraction():
    """Verify VerifiedToolRouter correctly parses expressions and routes arithmetic."""
    router = VerifiedToolRouter()
    
    trace = router.route_and_execute("Please compute 500 - 234 for me.")
    assert trace is not None
    assert trace.success
    assert trace.tool_name == "safe_calculator"
    assert trace.tool_output == 266
    
    # Non-mathematical query should not trigger execution
    non_math_trace = router.route_and_execute("Hello, what is your name?")
    assert non_math_trace is None


def test_tool_grounded_arithmetic_accuracy():
    """Verify that tool-assisted arithmetic achieves 100% on the 20 canonical probes."""
    probes = [
        "12 + 15", "25 * 4", "100 - 37", "144 / 12", "7 * 8",
        "81 / 9", "45 + 55", "250 - 125", "16 * 16", "1000 / 8",
        "33 + 77", "64 - 28", "13 * 7", "225 / 15", "99 + 101",
        "500 - 234", "18 * 5", "360 / 6", "49 + 51", "120 - 45"
    ]
    targets = [
        27, 100, 63, 12, 56,
        9, 100, 125, 256, 125,
        110, 36, 91, 15, 200,
        266, 90, 60, 100, 75
    ]
    eval_res = evaluate_arithmetic_probes_with_tool(probes, targets)
    assert eval_res["correct"] == 20
    assert eval_res["accuracy"] == 1.0
