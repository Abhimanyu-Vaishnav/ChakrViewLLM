"""
Tests for Batch 34:
SFT Causal Alignment, EOS Supervision, Diagnostic Synthetic Loss, and Provenance Truth.
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


def test_synthetic_diagnostic_sft_alignment():
    # Synthetic tokens: Prompt = [10, 20], Response = [30, 40], BOS = 0, EOS = 1
    p_toks = [10, 20]
    r_toks = [30, 40]
    bos_id = 0
    eos_id = 1
    ignore = -100

    input_seq = [bos_id] + p_toks + r_toks
    target_seq = ([ignore] * len(p_toks)) + r_toks + [eos_id]

    assert len(input_seq) == len(target_seq)
    assert input_seq == [0, 10, 20, 30, 40]
    assert target_seq == [-100, -100, 30, 40, 1]

    # In causal language modeling:
    # Position 0 (BOS) predicts position 1 (p_toks[0]=10) -> target is masked (-100)
    # Position 1 (p_toks[0]) predicts position 2 (p_toks[1]=20) -> target is masked (-100)
    # Position 2 (p_toks[1]) predicts position 3 (r_toks[0]=30) -> target is 30!
    # Position 3 (r_toks[0]) predicts position 4 (r_toks[1]=40) -> target is 40!
    # Position 4 (r_toks[1]) predicts position 5 (EOS=1) -> target is 1!
    active_targets = [t for t in target_seq if t != ignore]
    assert active_targets == [30, 40, 1]
    assert active_targets[-1] == eos_id


def test_synthetic_loss_responds_to_target_variation():
    model = ChakrMicro(ModelConfig(max_seq_len=512))
    model.eval()

    inp = torch.tensor([[0, 10, 20, 30, 40]], dtype=torch.long)
    target_correct = torch.tensor([[-100, -100, 30, 40, 1]], dtype=torch.long)
    target_wrong = torch.tensor([[-100, -100, 999, 888, 1]], dtype=torch.long)

    with torch.no_grad():
        logits = model(inp)
        l_correct = F.cross_entropy(logits.view(-1, logits.size(-1)), target_correct.view(-1), ignore_index=-100)
        l_wrong = F.cross_entropy(logits.view(-1, logits.size(-1)), target_wrong.view(-1), ignore_index=-100)

        assert not torch.isnan(l_correct) and not torch.isnan(l_wrong)
        assert l_correct.item() != l_wrong.item()


def test_provenance_review_integrity():
    review_path = "experiments/batch34_sft_root_cause/audits/provenance_record_review.json"
    if not os.path.exists(review_path):
        pytest.skip("Provenance review audit not found")

    with open(review_path, "r", encoding="utf-8") as f:
        recs = json.load(f)

    # Ensure unsupported records are flagged honestly
    unsupported = [r for r in recs if r["status"] == "unsupported"]
    assert len(unsupported) == 2
    assert "TR-CD-01" in [r["id"] for r in unsupported]
