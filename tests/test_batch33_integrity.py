"""
Tests for Batch 33:
Data Provenance, Document-Level Disjointness, Causal Masking Loss Alignment, and Git Integrity Bounds.
"""

import json
import os
import pytest
import torch
import torch.nn.functional as F

from chakrview.brain.model import ChakrMicro, ModelConfig
from chakrview.runtime.conversation import ConversationTemplate, ChatMessage, Role
from chakrview.runtime.local_runtime import DEFAULT_TOKENIZER_DIR
from chakrview.tokenizer.serialization import load_tokenizer_artifacts


def test_clean_sft_document_disjointness():
    manifest_path = "experiments/batch33_integrity/data/dataset_provenance_manifest.json"
    if not os.path.exists(manifest_path):
        pytest.skip("Batch 33 dataset manifest not found")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    train_docs = set(manifest["train_documents"])
    val_docs = set(manifest["val_documents"])
    overlap = train_docs.intersection(val_docs)

    assert len(overlap) == 0, f"Found document leakage: {overlap}"
    assert manifest["document_overlap_count"] == 0
    assert manifest["prompt_overlap_count"] == 0
    assert manifest["target_overlap_count"] == 0


def test_causal_sft_forward_loss_alignment():
    tok, _ = load_tokenizer_artifacts(DEFAULT_TOKENIZER_DIR)
    messages = [
        ChatMessage(role=Role.USER, content="Explain RMSNorm in ChakrMicro."),
        ChatMessage(role=Role.ASSISTANT, content="RMSNorm normalizes root-mean-square activations."),
    ]

    inputs, targets, active_tokens = ConversationTemplate.encode_multiturn_sft(messages, tok, max_seq_len=512)
    assert len(inputs) == len(targets)
    assert active_tokens > 0

    model = ChakrMicro(ModelConfig(max_seq_len=512))
    model.eval()

    inp_t = torch.tensor([inputs], dtype=torch.long)
    tgt_t = torch.tensor([targets], dtype=torch.long)

    with torch.no_grad():
        logits = model(inp_t)
        shift_logits = logits.view(-1, logits.size(-1))
        shift_targets = tgt_t.view(-1)
        loss = F.cross_entropy(shift_logits, shift_targets, ignore_index=-100)

        assert not torch.isnan(loss)
        assert not torch.isinf(loss)
        assert float(loss) > 0.0


def test_git_integrity_record_exists():
    git_audit_file = "experiments/batch33_integrity/audits/git_state.json"
    assert os.path.exists(git_audit_file)
    with open(git_audit_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["policy_enforcement"]["permission_rule"] == "ABSOLUTE_RESTRICTION"
    assert data["policy_enforcement"]["git_mutations_permitted"] is False
    assert data["local_remote_sync_status"] == "EXACT_MATCH_UP_TO_DATE"
