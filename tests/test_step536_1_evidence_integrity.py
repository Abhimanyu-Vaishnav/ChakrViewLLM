"""
Test Suite for Wave 536.1: Evidence Integrity Audit.
Verifies:
1. Actual weight-byte serialization and SHA-256 match against c5571c...00a282da.
2. Controlled 5-way ablation isolation and step reduction arithmetic.
3. Policy generalization classification vs memory retrieval.
4. Retry availability and elimination validity.
5. Longitudinal dependence across multi-turn cycles.
6. 10-facet rollback completeness via cryptographic fingerprints.
7. Test harness independence classification.
"""

import pytest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

def test_wave536_1_actual_weight_byte_hash():
    from scripts.run_wave536_1_evidence_integrity_audit import audit_actual_weight_bytes
    res = audit_actual_weight_bytes()
    assert res["is_bit_exact"] is True
    assert res["param_count"] == 3443136
    assert res["computed_sha256"] == "c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da"
    assert res["total_weight_bytes"] == 13772544

def test_wave536_1_ablation_arithmetic():
    from scripts.run_wave536_1_evidence_integrity_audit import run_controlled_ablations
    res = run_controlled_ablations()
    arith = res["arithmetic"]
    assert arith["control_steps"] == 17
    assert arith["treatment_steps"] == 14
    assert round(arith["step_reduction_pct"], 2) == 17.65
    assert arith["retry_reduction_pct"] == 100.0

def test_wave536_1_transfer_classification():
    from scripts.run_wave536_1_evidence_integrity_audit import audit_transfer_and_memorization
    res = audit_transfer_and_memorization()
    assert res["lexical_overlap_count"] == 0
    assert res["classification"] == "POLICY_GENERALIZATION"

def test_wave536_1_retry_elimination():
    from scripts.run_wave536_1_evidence_integrity_audit import audit_retry_elimination
    res = audit_retry_elimination()
    assert res["status"] == "PROVEN"
    assert res["retry_mechanism_exists"] is True

def test_wave536_1_longitudinal_dependence():
    from scripts.run_wave536_1_evidence_integrity_audit import audit_long_horizon_dependence
    res = audit_long_horizon_dependence()
    assert res["longitudinal_dependence"] == "PROVEN"
    assert res["total_cycles"] == 10

def test_wave536_1_rollback_completeness():
    from scripts.run_wave536_1_evidence_integrity_audit import audit_rollback_completeness_fingerprints
    res = audit_rollback_completeness_fingerprints()
    assert res["fingerprints_match"] is True
    assert res["rollback_completeness"] == "PROVEN"

def test_wave536_1_test_independence():
    from scripts.run_wave536_1_evidence_integrity_audit import audit_test_independence
    res = audit_test_independence()
    assert res["overall_status"] == "PASS"
