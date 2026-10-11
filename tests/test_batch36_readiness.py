"""
Tests for Batch 36:
Runtime Format Verification, Expanded Provenance Grounding, and Multi-Turn ChatML Alignment.
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


def test_expanded_dataset_document_disjointness():
    manifest_path = "experiments/batch36_sft_readiness/data/dataset_provenance_manifest.json"
    if not os.path.exists(manifest_path):
        pytest.skip("Batch 36 provenance manifest not found")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    train_docs = set(manifest["train_documents"])
    val_docs = set(manifest["val_documents"])
    overlap = train_docs.intersection(val_docs)

    assert len(overlap) == 0, f"Found document leakage: {overlap}"
    assert manifest["document_overlap_count"] == 0
    assert manifest["prompt_overlap_count"] == 0
    assert manifest["train_record_count"] == 12
    assert manifest["val_record_count"] == 7


def test_provenance_review_zero_unverified():
    review_path = "experiments/batch36_sft_readiness/audits/provenance_review.json"
    if not os.path.exists(review_path):
        pytest.skip("Provenance review audit not found")

    with open(review_path, "r", encoding="utf-8") as f:
        recs = json.load(f)

    assert len(recs) == 13
    unverified = [r for r in recs if r["status"] != "verified_source_grounded"]
    assert len(unverified) == 0


def test_runtime_chatml_token_sequence_structure():
    tok, _ = load_tokenizer_artifacts(DEFAULT_TOKENIZER_DIR)
    msgs = [
        ChatMessage(role=Role.USER, content="Test prompt"),
    ]
    chatml_str = ConversationTemplate.format_conversation(msgs, add_generation_prompt=True)
    tokens = tok.encode(chatml_str, add_bos=True, add_eos=False)

    # First token is BOS
    assert tokens[0] == 0
    # <|im_start|> decomposes to [63, 127, 1012, 98, 1832, 119, 127, 65]
    start_tag_ids = tok.encode("<|im_start|>", add_bos=False, add_eos=False)
    assert tokens[1:1+len(start_tag_ids)] == start_tag_ids
