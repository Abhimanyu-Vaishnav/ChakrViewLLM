"""
Tests for Batch 37:
Independent Dataset Audit, Source File Verification, Baseline Metric Reproduction, and Training Gate.
"""

import json
import os
import pytest
import torch

from chakrview.brain.model import ChakrMicro, ModelConfig
from chakrview.runtime.conversation import ConversationTemplate, ChatMessage, Role
from chakrview.runtime.local_runtime import DEFAULT_TOKENIZER_DIR, LocalModelRuntime
from chakrview.tokenizer.serialization import load_tokenizer_artifacts


def test_source_inventory_all_hashes_match():
    inventory_path = "experiments/batch37_independent_audit/audits/source_inventory.json"
    if not os.path.exists(inventory_path):
        pytest.skip("Source inventory not found")

    with open(inventory_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert len(data) == 27
    mismatches = [k for k, v in data.items() if not v["sha_match"]]
    assert len(mismatches) == 0, f"Found SHA mismatches: {mismatches}"


def test_batch36_claims_audit_identifies_expansion_discrepancy():
    claims_path = "experiments/batch37_independent_audit/audits/batch36_claims_audit.json"
    if not os.path.exists(claims_path):
        pytest.skip("Batch 36 claims audit not found")

    with open(claims_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    eval_claims = data["claims_evaluated"]
    assert eval_claims["claim_1_provenance_review_scope"]["finding"] == "INACCURATE / MISALIGNED"
    assert eval_claims["claim_4_training_gate_decision"]["finding"] == "CONFIRMED_ACCURATE"


def test_independent_baseline_metric_reproduction():
    baseline_summary_path = "experiments/batch37_independent_audit/evaluation/baseline_summary.json"
    if not os.path.exists(baseline_summary_path):
        pytest.skip("Baseline summary not found")

    with open(baseline_summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    # Base checkpoint must have 0% first-token EOS rate across all 10 evaluation prompts
    assert summary["total_evaluated"] == 10
    assert summary["first_token_eos_count"] == 0
    assert summary["first_token_eos_rate"] == 0.0
    assert summary["empty_decoded_count"] == 0


def test_training_gate_conditions():
    plan_path = "experiments/batch37_independent_audit/data/expansion_plan.json"
    if not os.path.exists(plan_path):
        pytest.skip("Expansion plan not found")

    with open(plan_path, "r", encoding="utf-8") as f:
        plan = json.load(f)

    # Training gate must be BLOCKED
    assert "BLOCKED" in plan["training_gate_recommendation"]
