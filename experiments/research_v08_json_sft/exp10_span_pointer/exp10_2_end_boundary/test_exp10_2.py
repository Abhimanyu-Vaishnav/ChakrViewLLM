"""
test_exp10_2.py
ChakrView Experiment 10.2 — Comprehensive Unit and Diagnostic Separation Test Suite

Tests:
1. One-token span extraction.
2. Multi-token spans (lengths 2, 3, 4).
3. Repeated values at multiple locations.
4. Punctuation-adjacent spans.
5. Maximum-length boundaries ($L_{max}=8$).
6. Ineligible end positions and candidate masking.
7. Oracle-Start mode separation from normal inference.
8. Deterministic metric recomputation from saved predictions.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chakrview.brain.config import ModelConfig
from experiments.research_v08_json_sft.exp10_span_pointer.exp10_2_end_boundary.start_conditioned_span_model import ChakrStartConditionedSpanPointer
from experiments.research_v08_json_sft.exp10_span_pointer.exp10_2_end_boundary.evaluate_exp10_2 import compute_metrics_for_eval

def test_single_token_span():
    config = ModelConfig(max_seq_len=512)
    model = ChakrStartConditionedSpanPointer(config, max_span_len=8, freeze_backbone=False)
    inp = torch.tensor([[0, 10, 20, 500, 30, 40]], dtype=torch.long)
    mask = torch.tensor([[False, True, True, True, True, False]], dtype=torch.bool)
    s, e, toks, _ = model.predict_span(inp, mask)
    assert s <= e
    assert 1 <= s <= 4
    assert 1 <= e <= 4

def test_multitoken_spans_lengths_2_to_4():
    config = ModelConfig(max_seq_len=512)
    model = ChakrStartConditionedSpanPointer(config, max_span_len=8, freeze_backbone=False)
    # Len 2, 3, 4 tests
    for length in [2, 3, 4]:
        seq = [0, 10] + list(range(100, 100 + length)) + [20, 30]
        inp = torch.tensor([seq], dtype=torch.long)
        mask = torch.ones_like(inp, dtype=torch.bool)
        mask[0, 0] = False
        mask[0, -1] = False
        s, e, toks, _ = model.predict_span(inp, mask, oracle_start=2)
        assert s == 2
        assert s <= e
        assert e - s + 1 <= 8

def test_repeated_values():
    config = ModelConfig(max_seq_len=512)
    model = ChakrStartConditionedSpanPointer(config, max_span_len=8, freeze_backbone=False)
    # 50, 60 appears twice
    inp = torch.tensor([[0, 50, 60, 10, 20, 50, 60, 99]], dtype=torch.long)
    mask = torch.tensor([[False, True, True, True, True, True, True, False]], dtype=torch.bool)
    # Test targeting first instance vs second instance via oracle start
    s1, e1, toks1, _ = model.predict_span(inp, mask, oracle_start=1)
    assert s1 == 1
    assert e1 >= 1
    s2, e2, toks2, _ = model.predict_span(inp, mask, oracle_start=5)
    assert s2 == 5
    assert e2 >= 5

def test_punctuation_adjacent_spans():
    config = ModelConfig(max_seq_len=512)
    model = ChakrStartConditionedSpanPointer(config, max_span_len=8, freeze_backbone=False)
    # Punctuation adjacent tokens
    inp = torch.tensor([[0, 25, 100, 101, 26, 99]], dtype=torch.long) # e.g. [tok1, tok2]
    mask = torch.tensor([[False, False, True, True, False, False]], dtype=torch.bool)
    s, e, toks, _ = model.predict_span(inp, mask)
    assert s in (2, 3)
    assert e in (2, 3)
    assert s <= e

def test_max_length_boundary():
    config = ModelConfig(max_seq_len=512)
    max_len = 4
    model = ChakrStartConditionedSpanPointer(config, max_span_len=max_len, freeze_backbone=False)
    inp = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]], dtype=torch.long)
    mask = torch.ones_like(inp, dtype=torch.bool)
    # Fix oracle start at 1
    s, e, toks, _ = model.predict_span(inp, mask, oracle_start=1)
    assert s == 1
    assert (e - s + 1) <= max_len

def test_ineligible_end_positions():
    config = ModelConfig(max_seq_len=512)
    model = ChakrStartConditionedSpanPointer(config, max_span_len=8, freeze_backbone=False)
    inp = torch.tensor([[0, 10, 20, 30, 40, 50]], dtype=torch.long)
    # Only position 1 and 2 are candidates, position 3, 4, 5 are masked out
    mask = torch.tensor([[False, True, True, False, False, False]], dtype=torch.bool)
    s, e, toks, _ = model.predict_span(inp, mask)
    assert s in (1, 2)
    assert e in (1, 2)
    assert s <= e

def test_oracle_start_separation():
    config = ModelConfig(max_seq_len=512)
    model = ChakrStartConditionedSpanPointer(config, max_span_len=8, freeze_backbone=False)
    inp = torch.tensor([[0, 10, 20, 30, 40, 50]], dtype=torch.long)
    mask = torch.tensor([[False, True, True, True, True, False]], dtype=torch.bool)
    # Normal inference uses joint argmax
    norm_s, norm_e, _, _ = model.predict_span(inp, mask)
    # Oracle mode strictly forces start to requested position regardless of start logits
    forced_s = 4
    orc_s, orc_e, _, _ = model.predict_span(inp, mask, oracle_start=forced_s)
    assert orc_s == forced_s
    assert orc_e >= forced_s

def test_deterministic_metric_recomputation():
    records = [
        {"gold_start": 2, "gold_end": 4, "pred_start": 2, "pred_end": 4, "gold_tokens": [1, 2, 3], "pred_tokens": [1, 2, 3], "gold_val": "foo", "pred_val": "foo", "target_token_len": 3, "punct_adjacent": True, "is_premature": False, "is_invalid": False},
        {"gold_start": 2, "gold_end": 4, "pred_start": 2, "pred_end": 3, "gold_tokens": [1, 2, 3], "pred_tokens": [1, 2], "gold_val": "foo", "pred_val": "fo", "target_token_len": 3, "punct_adjacent": True, "is_premature": True, "is_invalid": False},
    ]
    m1 = compute_metrics_for_eval(records)
    m2 = compute_metrics_for_eval(records)
    assert m1 == m2
    assert m1["total"] == 2
    assert m1["exact_span"] == 1
    assert m1["exact_span_acc"] == 0.5
    assert m1["start_acc"] == 1.0
    assert m1["end_acc"] == 0.5
    assert m1["premature_end_count"] == 1
    assert m1["premature_end_rate"] == 0.5

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
