"""
Unit tests for Batch 115 Evaluation Semantics & Multi-Metric Module.
Tests deterministic metric calculations using synthetic predictions with
deliberately correct, malformed, extra-field, and boundary cases.
Does not require or depend on neural model weights.
"""
import pytest
from experiments.batch115_118_tool_gating_curriculum.metrics import (
    parse_tool_grammar,
    validate_schema,
    evaluate_single_prediction,
    compute_aggregate_metrics,
)

def test_grammar_parsing_valid_and_invalid():
    # 1. Standard TOOL_CALL
    p1 = "TOOL_CALL: {\"tool\": \"calculator\", \"op\": \"add\", \"a\": 1, \"b\": 2}<|im_end|>"
    ok, data, err = parse_tool_grammar(p1)
    assert ok is True
    assert data == {"tool": "calculator", "op": "add", "a": 1, "b": 2}
    
    # 2. Malformed JSON
    p2 = "TOOL_CALL: {\"tool\": \"calculator\", broken json"
    ok, data, err = parse_tool_grammar(p2)
    assert ok is False
    assert data is None
    
    # 3. Plain refusal text (non-tool)
    p3 = "I only support calculator addition.<|im_end|>"
    ok, data, err = parse_tool_grammar(p3)
    assert ok is False
    assert data is None

def test_schema_validation_strictness():
    # 1. Valid payload
    v = {"tool": "calculator", "op": "add", "a": 3, "b": 5}
    ok, err = validate_schema(v)
    assert ok is True
    
    # 2. Missing field
    missing = {"tool": "calculator", "a": 3, "b": 5}
    ok, err = validate_schema(missing)
    assert ok is False
    assert "Missing required fields" in err
    
    # 3. Prohibited extra field (security injection test)
    extra = {"tool": "calculator", "op": "add", "a": 3, "b": 5, "malicious": "rm -rf"}
    ok, err = validate_schema(extra)
    assert ok is False
    assert "Prohibited extra fields" in err
    
    # 4. Wrong operand types (floats or strings)
    wrong_type = {"tool": "calculator", "op": "add", "a": "3", "b": 5}
    ok, err = validate_schema(wrong_type)
    assert ok is False
    assert "must be integers" in err

def test_accidental_sum_does_not_pass_e2e_correctness():
    # Case: expected is (1, 8) -> sum 9.
    # Prediction emits (5, 4) -> also sum 9!
    # e2e_correct must be FALSE because operands do not match input request!
    rec = {"requires_tool": True, "operands": [1, 8]}
    pred = "TOOL_CALL: {\"tool\": \"calculator\", \"op\": \"add\", \"a\": 5, \"b\": 4}"
    res = evaluate_single_prediction(pred, rec)
    
    assert res["parsed"] is True
    assert res["schema_valid"] is True
    assert res["executed"] is True
    assert res["a_correct"] is False
    assert res["b_correct"] is False
    assert res["exact_call"] is False
    assert res["e2e_correct"] is False, "Accidental arithmetic result match must not count as e2e success!"

def test_exact_match_and_execution_success():
    rec = {"requires_tool": True, "operands": [4, 7]}
    pred = "TOOL_CALL: {\"tool\": \"calculator\", \"op\": \"add\", \"a\": 4, \"b\": 7}<|im_end|>"
    res = evaluate_single_prediction(pred, rec)
    
    assert res["parsed"] is True
    assert res["schema_valid"] is True
    assert res["tool_correct"] is True
    assert res["op_correct"] is True
    assert res["a_correct"] is True
    assert res["b_correct"] is True
    assert res["exact_call"] is True
    assert res["executed"] is True
    assert res["e2e_correct"] is True

def test_false_positive_tool_call_detection():
    # Negative control prompt (e.g., translation)
    non_tool_rec = {"requires_tool": False}
    
    # Case A: Model outputs tool call on non-tool prompt -> False Positive!
    pred_bad = "TOOL_CALL: {\"tool\": \"calculator\", \"op\": \"add\", \"a\": 0, \"b\": 0}"
    res_bad = evaluate_single_prediction(pred_bad, non_tool_rec)
    assert res_bad["false_positive"] is True
    assert res_bad["correct"] is False
    
    # Case B: Model cleanly refuses
    pred_good = "I only support addition requests.<|im_end|>"
    res_good = evaluate_single_prediction(pred_good, non_tool_rec)
    assert res_good["false_positive"] is False
    assert res_good["correct"] is True

def test_aggregate_metrics_calculations():
    records = [
        {"requires_tool": True, "operands": [1, 2]},
        {"requires_tool": True, "operands": [3, 4]},
        {"requires_tool": False},
        {"requires_tool": False},
    ]
    preds = [
        "TOOL_CALL: {\"tool\": \"calculator\", \"op\": \"add\", \"a\": 1, \"b\": 2}", # correct
        "TOOL_CALL: {\"tool\": \"calculator\", \"op\": \"add\", \"a\": 1, \"b\": 2}", # wrong operands for (3,4)
        "Refusal text",                                                               # clean refusal
        "TOOL_CALL: {\"tool\": \"calculator\", \"op\": \"add\", \"a\": 0, \"b\": 0}", # spurious tool call
    ]
    evals = [evaluate_single_prediction(p, r) for p, r in zip(preds, records)]
    agg = compute_aggregate_metrics(evals)
    
    t_m = agg["tool_metrics"]
    nt_m = agg["non_tool_metrics"]
    
    assert agg["tool_examples_count"] == 2
    assert agg["non_tool_examples_count"] == 2
    assert t_m["parse_rate"] == 1.0
    assert t_m["schema_validity_rate"] == 1.0
    assert t_m["exact_call_accuracy"] == 0.5 # 1 out of 2
    assert t_m["e2e_correctness_rate"] == 0.5 # 1 out of 2
    assert t_m["collapse_rate"] == 1.0       # both tool predictions were identical
    assert nt_m["total_false_positive_rate"] == 0.5 # 1 out of 2 non-tool
    assert nt_m["valid_false_positive_rate"] == 0.5
    assert nt_m["malformed_false_positive_rate"] == 0.0
    assert nt_m["clean_refusal_rate"] == 0.5           # 1 out of 2

def test_malformed_tool_call_on_non_tool_prompt():
    # Non-tool prompt receiving a malformed tool call attempt
    non_tool_rec = {"requires_tool": False}
    
    # Case: Model starts TOOL_CALL or emits {"tool": but malformed JSON/grammar
    pred_malformed = "TOOL_CALL: {\"tool\": \"calculator\", broken json..."
    res = evaluate_single_prediction(pred_malformed, non_tool_rec)
    
    assert res["requires_tool"] is False
    assert res["attempted_tool"] is True
    assert res["parsed"] is False
    assert res["schema_valid"] is False
    assert res["false_positive"] is True, "Malformed tool call on non-tool prompt must be counted as a false positive!"
    assert res["malformed_false_positive"] is True
    assert res["valid_false_positive"] is False
    assert res["correct"] is False

def test_swapped_operands_do_not_count_as_exact():
    rec = {"requires_tool": True, "operands": [2, 7]}
    # Swapped operands: a=7, b=2
    pred = "TOOL_CALL: {\"tool\": \"calculator\", \"op\": \"add\", \"a\": 7, \"b\": 2}"
    res = evaluate_single_prediction(pred, rec)
    
    assert res["parsed"] is True
    assert res["schema_valid"] is True
    assert res["a_correct"] is False
    assert res["b_correct"] is False
    assert res["exact_call"] is False
    assert res["e2e_correct"] is False, "Swapped operands must not pass exact call or e2e correctness!"
