"""
Test Suite for Wave 536: Governed Self-Improvement Integration Audit.
Validates:
1. End-to-end trace transitions and downstream provenance consumption
2. Governance bypass resistance across Cases A through J
3. Causality audit: measurable computation step reduction over naive baseline
4. Memory vs abstract strategy learning distinction (0% memorization)
5. Unseen generalization across lexical substitutions and surface reformulations
6. Long-horizon multi-turn stability under environmental drift
7. Strategy staleness and retirement state machine
8. Rollback completeness across all 10 governed state facets
9. Cold restart persistence (RESTART != RESET)
10. Canonical baseline parameter and hash invariance
"""

import pytest
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

def test_wave536_baseline_integrity():
    from chakrview.brain.model import ChakrMicro
    from chakrview.brain.config import ModelConfig
    from chakrview.runtime.interactive import EXPECTED_WEIGHT_HASH

    m = ChakrMicro(ModelConfig())
    param_count = sum(p.numel() for p in m.parameters())
    assert param_count == 3443136
    assert EXPECTED_WEIGHT_HASH == "c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da"

def test_wave536_end_to_end_trace():
    from scripts.run_wave536_integration_audit import run_end_to_end_trace
    trace = run_end_to_end_trace()
    assert trace["step_3_learning_worthiness"]["decision"] is not None
    assert trace["step_4_trust"]["trust_level"] is not None
    assert trace["step_5_abstraction"]["valid"] is True
    assert trace["step_7_validate_and_commit"]["status"] == "COMMITTED"
    assert trace["step_8_reuse"]["correct_dispatch"] is True

def test_wave536_adversarial_bypass_cases():
    from scripts.run_wave536_integration_audit import run_adversarial_governance_audit
    cases = run_adversarial_governance_audit()
    assert len(cases) == 10
    assert all(cases.values())

def test_wave536_causality():
    from scripts.run_wave536_integration_audit import run_causality_experiment
    res = run_causality_experiment()
    assert res["causality_established"] is True
    assert res["step_reduction_gain"] > 15.0 # Demonstrates 17.6% compute reduction

def test_wave536_unseen_generalization():
    from scripts.run_wave536_integration_audit import run_unseen_generalization_eval
    res = run_unseen_generalization_eval()
    assert res["transfer_gain"] == 100.0
    assert res["memorization_rate"] == 0.0

def test_wave536_long_horizon_strategy_aging():
    from scripts.run_wave536_integration_audit import run_long_horizon_10_cycle_audit
    res = run_long_horizon_10_cycle_audit()
    assert res["cycles_completed"] == 10
    assert res["initial_strategy_state"] == "RETIRED"
    assert res["replacement_strategy_state"] == "ACTIVE"

def test_wave536_rollback_completeness():
    from scripts.run_wave536_integration_audit import run_rollback_completeness_audit
    facets = run_rollback_completeness_audit()
    assert len(facets) == 10
    assert all(facets.values())

def test_wave536_restart_persistence():
    from scripts.run_wave536_integration_audit import run_restart_persistence_audit
    res = run_restart_persistence_audit()
    assert all(res.values())
