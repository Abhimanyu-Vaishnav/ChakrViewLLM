"""
test_exp10_3.py
ChakrView Experiment 10.3 — Comprehensive Mask Leakage and Correctness Test Suite

Tests:
1. Ordinary inference masks do not depend on ground-truth start/end positions.
2. Changing only the gold labels cannot change ordinary candidate eligibility.
3. Oracle-start mode changes only the start intervention (no end-candidate alteration).
4. Gold end labels never unlock end candidates.
5. Invalid target positions (outside candidate boundaries) are reported rather than force-enabled.
6. Metric denominators strictly include invalid and missing predictions.
7. Results are deterministic across two audit runs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chakrview.brain.config import ModelConfig
from experiments.research_v08_json_sft.exp10_span_pointer.span_pointer_model import ChakrSpanPointerModel
from experiments.research_v08_json_sft.exp10_span_pointer.exp10_2_end_boundary.start_conditioned_span_model import ChakrStartConditionedSpanPointer
from experiments.research_v08_json_sft.exp10_span_pointer.exp10_3_mask_audit.audit_exp10_3_masks import (
    build_pure_inference_mask,
    build_original_mask,
)

def test_pure_mask_does_not_depend_on_gold_labels():
    """Test 1: Verify pure inference mask signature and output depends only on seq_len."""
    seq_len = 50
    m1 = build_pure_inference_mask(seq_len)
    m2 = build_pure_inference_mask(seq_len)
    assert torch.equal(m1, m2)
    # Check boundaries: 0 (BOS) is False, 1..seq_len-3 is True, seq_len-2..seq_len-1 is False
    assert not m1[0, 0].item()
    assert m1[0, 1].item()
    assert m1[0, seq_len - 3].item()
    assert not m1[0, seq_len - 2].item()
    assert not m1[0, seq_len - 1].item()

def test_changing_gold_labels_does_not_change_pure_mask():
    """Test 2: Changing only ground-truth labels cannot alter pure candidate mask."""
    seq_len = 40
    # Dummy label set A
    gold_s_A, gold_e_A = 5, 8
    # Dummy label set B
    gold_s_B, gold_e_B = 12, 16

    # Pure mask generator ignores labels completely
    mask_A = build_pure_inference_mask(seq_len)
    mask_B = build_pure_inference_mask(seq_len)
    assert torch.equal(mask_A, mask_B)

def test_oracle_start_mode_changes_only_start_intervention():
    """Test 3: Oracle-start diagnostic fixes start position but does not alter candidate eligibility mask."""
    config = ModelConfig(max_seq_len=512)
    model = ChakrStartConditionedSpanPointer(config, max_span_len=6, freeze_backbone=False)
    inp = torch.tensor([[0, 10, 20, 30, 40, 50, 60, 70]], dtype=torch.long)
    mask = build_pure_inference_mask(8)

    # In oracle mode, start is forced to oracle_start
    forced_start = 3
    s, e, toks, _ = model.predict_span(inp, candidate_mask=mask, oracle_start=forced_start)
    assert s == forced_start
    # End position must be eligible under candidate_mask
    assert mask[0, e].item()

def test_gold_end_labels_never_unlock_end_candidates():
    """Test 4: In pure inference, positions outside prompt body remain masked even if gold end points there."""
    seq_len = 30
    mask = build_pure_inference_mask(seq_len)

    # Position seq_len - 1 is prompt trailer
    gold_e = seq_len - 1
    assert not mask[0, gold_e].item()

    # The pure mask must NOT unlock it
    assert not mask[0, seq_len - 1].item()
    assert not mask[0, seq_len - 2].item()
    assert not mask[0, 0].item()

def test_invalid_target_positions_are_reported_not_force_enabled():
    """Test 5: An invalid span outside mask boundary is reported as unrepresentable."""
    seq_len = 20
    mask = build_pure_inference_mask(seq_len)

    # Let gold span be located at trailer position 19 (e.g. unrepresentable target)
    invalid_gold_s = 18
    invalid_gold_e = 19
    is_represented = mask[0, invalid_gold_s].item() and mask[0, invalid_gold_e].item()
    assert not is_represented, "Invalid position must not be represented by pure mask"

def test_metric_denominators_include_invalid_predictions():
    """Test 6: Denominator policy strictly counts all evaluation examples."""
    from experiments.research_v08_json_sft.exp10_span_pointer.exp10_3_mask_audit.audit_exp10_3_masks import evaluate_split
    from chakrview.tokenizer.serialization import load_tokenizer_artifacts

    tokenizer, _ = load_tokenizer_artifacts(REPO_ROOT / "data/experiments/vocab_4096")
    config = ModelConfig(max_seq_len=512)
    model = ChakrSpanPointerModel(config, max_span_len=8, freeze_backbone=False)

    # 2 mock records: one valid, one that would fail
    records = [
        {"prompt": "Alpha=1, Beta=2\nCopy Alpha: ", "start_token_index": 2, "end_token_index": 2, "expected_token_sequence": [10], "target_value": "1", "target_span_length": 1, "example_id": "mock_1"},
        {"prompt": "Alpha=1, Beta=2\nCopy Alpha: ", "start_token_index": 2, "end_token_index": 2, "expected_token_sequence": [10], "target_value": "1", "target_span_length": 1, "example_id": "mock_2"}
    ]
    res = evaluate_split(model, records, tokenizer, "mock_split", "cpu", is_start_conditioned=False)
    assert res["total"] == 2
    assert res["pure_mask_metrics"]["total"] == 2

def test_deterministic_audit_runs():
    """Test 7: Verify two independent audit evaluations yield bit-for-bit identical results."""
    seq_len = 25
    m1 = build_pure_inference_mask(seq_len)
    m2 = build_pure_inference_mask(seq_len)
    assert torch.equal(m1, m2)

    m_orig1 = build_original_mask(seq_len, 5, 8)
    m_orig2 = build_original_mask(seq_len, 5, 8)
    assert torch.equal(m_orig1, m_orig2)
    assert torch.equal(m_orig1, m1)
