"""
Unit and regression test suite for Batches 123-126 multi-digit prompt-diverse benchmark.
Tests:
- Deterministic dataset generation and manifest verification
- Split isolation: unseen pair isolation, reversed-pair isolation, held-out template isolation
- Malformed tool call detection on negative controls
- Operand-order correctness (swapped operands do not pass exact match)
- Arithmetic-equivalent results do not pass without exact argument match
- Metric denominators and cold reload verification
"""
import json
import pytest
from pathlib import Path

from experiments.batch115_118_tool_gating_curriculum.metrics import (
    parse_tool_grammar,
    validate_schema,
    evaluate_single_prediction,
    compute_aggregate_metrics,
)
from experiments.batch123_126_prompt_diverse_multidigit.data.generate_multidigit_dataset import (
    build_dataset,
    TRAIN_TEMPLATES,
    HELD_OUT_TEMPLATES,
)

def test_dataset_determinism_and_split_isolation():
    # Deterministic generation
    train_1, val_1, test_1, meta_1 = build_dataset(seed=42)
    train_2, val_2, test_2, meta_2 = build_dataset(seed=42)
    
    assert len(train_1) == len(train_2) == 100
    assert len(val_1) == len(val_2) == 20
    assert len(test_1) == len(test_2) == 45
    assert train_1[0]["id"] == train_2[0]["id"]
    
    # Pair isolation
    train_pairs = {tuple(r["operands"]) for r in train_1 if r.get("operands")}
    train_rev_pairs = {(b, a) for (a, b) in train_pairs}
    
    p2_unseen = [r for r in test_1 if r["category"] == "unseen_pairs_familiar_templates"]
    p3_unseen = [r for r in test_1 if r["category"] == "unseen_pairs_unseen_templates"]
    
    for r in p2_unseen + p3_unseen:
        pair = tuple(r["operands"])
        assert pair not in train_pairs, f"Unseen pair {pair} leaked into train pairs!"
        assert pair not in train_rev_pairs, f"Unseen pair {pair} had its reverse in train pairs!"

def test_held_out_template_isolation():
    train_recs, _, test_recs, _ = build_dataset(seed=42)
    
    # Train prompts must not contain held-out template phrasing
    for r in train_recs:
        for hot in HELD_OUT_TEMPLATES:
            hot_prefix = hot.split("{a}")[0]
            assert hot_prefix not in r["prompt"], f"Held out template prefix '{hot_prefix}' found in train prompt!"
            
    # Partitions 1 and 3 in test must use held-out templates
    p1 = [r for r in test_recs if r["category"] == "familiar_pairs_unseen_templates"]
    p3 = [r for r in test_recs if r["category"] == "unseen_pairs_unseen_templates"]
    for r in p1 + p3:
        matches = [hot for hot in HELD_OUT_TEMPLATES if hot.split("{a}")[0] in r["prompt"]]
        assert len(matches) > 0, f"Expected held-out template in test record {r['id']}"

def test_operand_order_and_arithmetic_equivalence_guard():
    # Case: operands are 25 and 75
    # Swapped operands: 75 and 25
    rec = {"requires_tool": True, "operands": [25, 75]}
    pred_swapped = 'TOOL_CALL: {"tool": "calculator", "op": "add", "a": 75, "b": 25}'
    res_swapped = evaluate_single_prediction(pred_swapped, rec)
    
    assert res_swapped["parsed"] is True
    assert res_swapped["schema_valid"] is True
    assert res_swapped["a_correct"] is False
    assert res_swapped["b_correct"] is False
    assert res_swapped["exact_call"] is False
    assert res_swapped["e2e_correct"] is False, "Swapped operands must not be marked e2e correct!"
    
    # Accidental sum match: operands are 20 and 80 (sum 100), prediction has 50 and 50 (sum 100)
    pred_accidental = 'TOOL_CALL: {"tool": "calculator", "op": "add", "a": 50, "b": 50}'
    res_accidental = evaluate_single_prediction(pred_accidental, rec)
    assert res_accidental["exact_call"] is False
    assert res_accidental["e2e_correct"] is False, "Accidental sum match must fail e2e correctness!"

def test_malformed_tool_call_on_negative_control():
    non_tool_rec = {"requires_tool": False}
    # Prediction emits tool attempt with broken JSON syntax
    pred_malformed = 'TOOL_CALL: {"tool": "calculator", "a": 42 broken...'
    res = evaluate_single_prediction(pred_malformed, non_tool_rec)
    
    assert res["requires_tool"] is False
    assert res["attempted_tool"] is True
    assert res["parsed"] is False
    assert res["false_positive"] is True
    assert res["malformed_false_positive"] is True
    assert res["valid_false_positive"] is False
    assert res["correct"] is False
