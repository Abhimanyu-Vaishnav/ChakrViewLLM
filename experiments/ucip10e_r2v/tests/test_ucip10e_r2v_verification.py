"""
Unit test suite for UCIP-10E-R2V Independent Audit & Verification.
"""

import json
import hashlib
from pathlib import Path
import pytest

REPO_ROOT = Path(r"D:\Project\ChakrView")
BASE_R2V = REPO_ROOT / "experiments" / "ucip10e_r2v"
BASE_R2 = REPO_ROOT / "experiments" / "ucip10e_r2"

def test_protected_runtime_hash():
    session_path = REPO_ROOT / "chakrview" / "runtime" / "session.py"
    assert session_path.exists(), "Protected runtime session.py is missing!"
    h = hashlib.sha256(session_path.read_bytes()).hexdigest()
    assert h == "9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67", f"Protected runtime hash mismatch: {h}"

def test_checkpoint_hashes():
    expected = {
        REPO_ROOT / "releases" / "ucip_09_candidates" / "chakrview_ucip09_medium_model.pt": "5513fefe878ca85a637aa8401b3bebf5b8e3ca93ab362e07722d629e34294499",
        REPO_ROOT / "releases" / "ucip_09r_candidates" / "chakrview_ucip09r_medium_model.pt": "e1b943222e70fce2bac1b198e656bfea294209de1291f0042a2ebd677ccef9f0",
        REPO_ROOT / "experiments" / "ucip09s" / "checkpoints" / "ucip09s_balanced_pilot.pt": "c99dcc0a229310ec682857349c06e1a46957b559a8d180a33a91ea05b225d830"
    }
    for p, exp in expected.items():
        assert p.exists(), f"Checkpoint {p} does not exist!"
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        assert h == exp, f"Checkpoint hash mismatch for {p.name}: {h} != {exp}"

def test_recalculated_scores_match():
    recalc_file = BASE_R2V / "evaluations" / "recalculated_scores.json"
    assert recalc_file.exists(), "recalculated_scores.json is missing!"
    data = json.load(open(recalc_file, encoding="utf-8"))
    
    for model_name, info in data.items():
        assert info["total"] == 460
        assert info["passed"] == 10
        assert info["failed"] == 450
        assert abs(info["accuracy_pct"] - 2.173913) < 0.001
        assert info["matches_reported_summary"] is True

def test_logic_passes_forensics():
    logic_file = BASE_R2V / "evaluations" / "logic_passes_forensics.json"
    assert logic_file.exists(), "logic_passes_forensics.json is missing!"
    data = json.load(open(logic_file, encoding="utf-8"))
    
    assert len(data) == 30, f"Expected 30 logic pass records, found {len(data)}"
    for item in data:
        assert item["raw_output"] == "Yes."
        assert "All squares are rectangles" in item["prompt"]
        assert item["score"] == 1.0

def test_zero_evaluator_false_negatives():
    fn_file = BASE_R2V / "evaluations" / "false_negative_audit.json"
    assert fn_file.exists(), "false_negative_audit.json is missing!"
    data = json.load(open(fn_file, encoding="utf-8"))
    assert data["total_false_negatives_found"] == 0, f"False negatives detected: {data['total_false_negatives_found']}"

def test_reproducibility_30_audit():
    rep_file = BASE_R2V / "evaluations" / "reproducibility_30_audit.json"
    assert rep_file.exists(), "reproducibility_30_audit.json is missing!"
    data = json.load(open(rep_file, encoding="utf-8"))
    
    for model in ["ucip09", "ucip09r", "ucip09s"]:
        m_data = data[model]
        assert m_data["total_probes_tested"] == 30
        assert m_data["exact_matches"] == 30
        assert m_data["mismatches"] == 0
        assert m_data["deterministic_token_match"] is True

def test_all_nine_reports_exist():
    expected_reports = [
        "PREFLIGHT_AND_HASH_AUDIT.md",
        "SCORE_RECALCULATION_AUDIT.md",
        "LOGIC_PASSES_FORENSIC_AUDIT.md",
        "LANGUAGE_FAILURES_INVESTIGATION.md",
        "EVALUATOR_FALSE_NEGATIVE_AUDIT.md",
        "MODEL_ADAPTER_AND_INFERENCE_PROTOCOL_AUDIT.md",
        "CROSS_CHECKPOINT_COMPARATIVE_AUDIT.md",
        "REPRODUCIBILITY_AUDIT.md",
        "FINAL_INDEPENDENT_R2V_VERDICT.md"
    ]
    reports_dir = BASE_R2V / "reports"
    for r in expected_reports:
        p = reports_dir / r
        assert p.exists(), f"Required report {r} missing!"
        assert p.stat().st_size > 100, f"Report {r} appears empty!"
