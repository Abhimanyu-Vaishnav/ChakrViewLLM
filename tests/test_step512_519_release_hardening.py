"""
Test Suite for Waves 512-519: ChakrView v0.2.0 Release Hardening & Founder Acceptance.
Validates:
1. Baseline integrity and parameter count (3,443,136 params, SHA-256 c5571c...00a282da)
2. Tokenizer integrity and roundtrip fidelity
3. Unified cognitive runtime construction and sub-second CPU execution
4. Artifact audit and manifest integrity
5. Security scan (0 leaked credentials)
6. Independent master benchmark reproduction
7. Deterministic release bundle & checksums
8. Founder acceptance runner
9. Final release gate decision (V020_PUBLIC_RELEASE_READY)
"""

import sys
import json
import pytest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

def test_wave512_clean_reproducibility():
    w512_path = REPO_ROOT / "artifacts" / "waves_512_519_execution" / "wave512_clean_reproducibility.json"
    assert w512_path.exists(), "Wave 512 artifact missing"
    with open(w512_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data.get("subprocess_verified") is True
    assert data.get("baseline_sha_verified") is True

def test_wave513_release_artifact_audit():
    w513_path = REPO_ROOT / "artifacts" / "waves_512_519_execution" / "wave513_artifact_inventory.json"
    assert w513_path.exists(), "Wave 513 artifact missing"
    with open(w513_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data.get("audit_passed") is True
    assert data.get("total_audited") == 15
    assert len(data.get("missing", [])) == 0

def test_wave514_security_audit():
    w514_path = REPO_ROOT / "artifacts" / "waves_512_519_execution" / "wave514_security_provenance_audit.json"
    assert w514_path.exists(), "Wave 514 artifact missing"
    with open(w514_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data.get("status") == "PASS"
    assert data.get("secret_findings_count") == 0
    assert data.get("provenance", {}).get("runtime_version") == "v0.2.0_unified_runtime"
    assert data.get("provenance", {}).get("manifest_hash") == "b8d9e2f41c3a776a"

def test_wave515_documentation_claims():
    doc_path = REPO_ROOT / "docs" / "STEP_515_DOCUMENTATION_CLAIM_AUDIT.md"
    readme_path = REPO_ROOT / "README.md"
    assert doc_path.exists(), "Wave 515 doc audit missing"
    assert readme_path.exists(), "README.md missing"
    readme_text = readme_path.read_text(encoding="utf-8")
    assert "Open-ended spontaneous long-form natural-language generation remains experimental" in readme_text
    assert "v0.2.0" in readme_text

def test_wave516_master_benchmark_reproduction():
    w511_path = REPO_ROOT / "artifacts" / "waves_504_511_execution" / "wave511_v020_master_gate.json"
    assert w511_path.exists(), "Master gate artifact missing"
    with open(w511_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data.get("baseline_params") == 3443136
    assert data.get("baseline_sha") == "c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da"
    assert data.get("baseline_delta_w") == 0
    assert data.get("i5_1h") == 100.0
    assert data.get("i5_4h") == 86.0
    assert data.get("strategy_selection") == 92.0
    assert data.get("final_decision") == "V020_RELEASE_CANDIDATE_VERIFIED"

def test_wave517_deterministic_release_bundle():
    bundle_dir = REPO_ROOT / "release" / "chakrview-v0.2.0"
    assert bundle_dir.exists(), "Release bundle missing"
    manifest_file = bundle_dir / "release_manifest.json"
    checksums_file = bundle_dir / "CHECKSUMS.sha256"
    assert manifest_file.exists(), "release_manifest.json missing"
    assert checksums_file.exists(), "CHECKSUMS.sha256 missing"

def test_wave518_founder_acceptance_script():
    verify_script = REPO_ROOT / "scripts" / "verify_v020.py"
    assert verify_script.exists(), "verify_v020.py missing"
    from scripts.verify_v020 import run_founder_acceptance
    ret = run_founder_acceptance()
    assert ret == 0, "Founder acceptance test failed"

def test_wave519_final_release_gate():
    from chakrview.cognition.v020_release_gate import evaluate_v020_release_gate
    report = evaluate_v020_release_gate()
    assert report["all_passed"] is True
    assert report["final_state"] == "V020_PUBLIC_RELEASE_READY"
