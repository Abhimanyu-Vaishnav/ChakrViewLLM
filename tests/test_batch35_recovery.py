"""
Tests for Batch 35:
Token-Level EOS Verification, Session Prompt Format Unification, and Provenance Disjointness.
"""

import json
import os
import pytest
import torch

from chakrview.brain.model import ChakrMicro, ModelConfig
from chakrview.runtime.conversation import ConversationTemplate, ChatMessage, Role
from chakrview.runtime.local_runtime import DEFAULT_TOKENIZER_DIR, LocalModelRuntime
from chakrview.runtime.session import LocalModelSession, ConversationRole
from chakrview.tokenizer.serialization import load_tokenizer_artifacts


def test_session_prompt_unification_uses_chatml():
    sess = LocalModelSession(session_id="test_chatml_unification")
    tok, _ = load_tokenizer_artifacts(DEFAULT_TOKENIZER_DIR)

    prompt, envelope, turns = sess.build_prompt_package(
        user_input="Hello world",
        tokenizer=tok,
        max_new_tokens=32,
    )

    # Must use canonical ChatML formatting
    assert "<|im_start|>user\nHello world<|im_end|>\n<|im_start|>assistant\n" in prompt
    assert "<|im_start|>" in prompt
    assert "<|im_end|>" in prompt


def test_provenance_manifest_zero_leakage():
    manifest_path = "experiments/batch35_sft_recovery/data/corrected_provenance_manifest.json"
    if not os.path.exists(manifest_path):
        pytest.skip("Corrected provenance manifest not found")

    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["document_level_overlap"] == 0
    assert data["prompt_level_overlap"] == 0
    assert data["unsupported_records_count"] == 0
    assert data["verified_grounded_count"] == 13


def test_token_level_eos_audit_distinction():
    audit_file = "experiments/batch35_sft_recovery/evaluation/candidate_comparison_summary.json"
    if not os.path.exists(audit_file):
        pytest.skip("Candidate comparison summary not found")

    with open(audit_file, "r", encoding="utf-8") as f:
        summary = json.load(f)

    # Base checkpoint must have 0% first-token EOS rate
    assert summary["step_10000_base"]["first_token_eos_rate"] == 0.0
    # Step 200 SFT must demonstrate elevated first-token EOS rate
    assert summary["step_200_sft"]["first_token_eos_rate"] >= 0.70
