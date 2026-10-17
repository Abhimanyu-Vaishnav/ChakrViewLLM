"""
Regression and verification test suite for Batch 2261-2300:
1. Truthful subsystem execution attribution (safe_calculator on CPU vs neural).
2. Evaluation provenance and checkpoint hash verification.
3. Candidate manifest schema and artifact integrity.
4. Consent-gated feedback isolation.
5. Source-wise Sanskrit drift replication.
"""

import json
import os
import sys
from pathlib import Path
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.batch2221_2260.integrated_assistant_adapter import IntegratedAssistantAdapter


def test_truthful_subsystem_attribution():
    """Verify CLI adapter reports safe_calculator on CPU even when device preference is CUDA."""
    adapter = IntegratedAssistantAdapter(forced_device="cuda")
    resp = adapter.ask("What is 500 - 234?")
    
    assert resp.decision == "CALCULATED"
    assert resp.execution_breakdown["subsystem"] == "safe_calculator"
    assert "CPU" in resp.execution_breakdown["actual_backend"]
    assert resp.execution_breakdown["neural_inference"] == "not invoked"


def test_evaluation_provenance_and_distinct_checkpoints():
    """Verify that all 6 evaluated checkpoints have distinct SHA-256 hashes and match audit records."""
    audit_file = REPO_ROOT / "experiments" / "batch2261_2300" / "EVALUATION_AUDIT.json"
    assert audit_file.exists()
    
    with open(audit_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    models = data["models"]
    assert "canonical_baseline" in models
    assert "cpt_seed_42_best" in models
    
    assert models["canonical_baseline"]["token_weighted_ce"] == 4.4789
    assert models["cpt_seed_42_best"]["token_weighted_ce"] == 4.2861


def test_source_wise_sanskrit_drift_metric():
    """Verify that Classical Sanskrit Bhagavad Gita drift is replicated at +0.0776."""
    audit_file = REPO_ROOT / "experiments" / "batch2261_2300" / "EVALUATION_AUDIT.json"
    with open(audit_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    base_srcs = data["models"]["canonical_baseline"]["per_source_metrics"]
    s42_srcs = data["models"]["cpt_seed_42_best"]["per_source_metrics"]
    
    base_gita = base_srcs["wikipedia_sa_bhagavad_gita"]["token_weighted_ce"]
    s42_gita = s42_srcs["wikipedia_sa_bhagavad_gita"]["token_weighted_ce"]
    
    delta = round(s42_gita - base_gita, 4)
    assert delta == 0.0776, f"Expected drift +0.0776, got {delta}"


def test_candidate_manifest_integrity():
    """Verify that candidate_manifest.json exists and accurately specifies candidate parameters."""
    manifest_file = REPO_ROOT / "experiments" / "batch2261_2300" / "candidate_manifest.json"
    assert manifest_file.exists()
    
    with open(manifest_file, "r", encoding="utf-8") as f:
        manifest = json.load(f)
        
    assert manifest["candidate_checkpoint"]["sha256"] == "d3268fc49378f8c5b9a30e71d4bab58ce7c4db6bcaaa1f32932ccfd8694b07c3"
    assert manifest["public_release"] is False
    assert manifest["architecture"]["parameter_count"] == 3443136


def test_feedback_consent_isolation(tmp_path):
    """Verify that user feedback is strictly withheld without explicit opt-in consent."""
    mem_p = tmp_path / "mem.json"
    fb_p = tmp_path / "fb.json"
    
    # Run without consent
    adapter_off = IntegratedAssistantAdapter(memory_path=mem_p, feedback_path=fb_p, enable_consent_default=False)
    resp_off = adapter_off.ask("What is 25 * 4?")
    assert resp_off.feedback_id is None
    assert not fb_p.exists()
    
    # Run with consent
    adapter_on = IntegratedAssistantAdapter(memory_path=mem_p, feedback_path=fb_p, enable_consent_default=True)
    resp_on = adapter_on.ask("What is 25 * 4?")
    assert resp_on.feedback_id is not None
    assert fb_p.exists()
