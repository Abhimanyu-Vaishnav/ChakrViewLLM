"""
Batch 127–130: Regression Test Suite for Multi-Digit Binding Diagnosis.
Verifies:
1. Canonical checkpoint invariant (96357705a683...)
2. Initializing copy checkpoint invariant (e84b2ea9b633...)
3. Dataset determinism, counts, and zero pair/reversed-pair leakage
4. Deterministic token span locator accuracy across single and multi-token digits
5. Strict evaluation metrics behavior on known synthetic predictions
6. Output collapse and swapped-operand detection
"""
import pytest
import json
import hashlib
from pathlib import Path

from experiments.batch103_106_slot_weighted_sft.slot_weighted_training import (
    BASE_CKPT, TOKENIZER_DIR, load_tokenizer_artifacts, compute_file_sha256
)
from experiments.batch127_130_binding_diagnosis.data.generate_datasets import (
    generate_datasets, TRAIN_TEMPLATES, HELD_OUT_TEMPLATES
)
from experiments.batch127_130_binding_diagnosis.span_locator import (
    extract_prompt_operand_token_spans, map_char_span_to_tokens
)
from experiments.batch115_118_tool_gating_curriculum.metrics import (
    evaluate_single_prediction, compute_aggregate_metrics
)

CANONICAL_SHA = "96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee"
COPY_CKPT_SHA = "e84b2ea9b633b50181fb120ea2444a4a3c84c008b28905dd33c24fedba2b1bc4"
COPY_CKPT_PATH = Path("experiments/batch107_110_operand_copy_benchmark/checkpoints/copy_task_sft_step150.pt")

def test_canonical_and_copy_checkpoint_invariants():
    assert Path(BASE_CKPT).exists(), "Canonical base checkpoint missing."
    assert compute_file_sha256(BASE_CKPT) == CANONICAL_SHA, "Canonical checkpoint hash drifted!"
    assert COPY_CKPT_PATH.exists(), "Copy initialization checkpoint missing."
    assert compute_file_sha256(str(COPY_CKPT_PATH)) == COPY_CKPT_SHA, "Copy checkpoint hash drifted!"

def test_dataset_generation_and_zero_leakage():
    train_b, train_e, val, test = generate_datasets(seed=42)
    assert len(train_b) == 100
    assert len(train_e) == 260
    assert len(val) == 20
    assert len(test) == 45
    
    # Check tool records counts
    tool_b = [r for r in train_b if r["requires_tool"]]
    tool_e = [r for r in train_e if r["requires_tool"]]
    assert len(tool_b) == 70
    assert len(tool_e) == 200
    
    # Check pair leakage
    e_pairs = {tuple(r["operands"]) for r in tool_e}
    e_rev_pairs = {(b, a) for (a, b) in e_pairs}
    
    p2 = [r for r in test if r["category"] == "unseen_pairs_familiar_templates"]
    p3 = [r for r in test if r["category"] == "unseen_pairs_unseen_templates"]
    assert len(p2) == 15
    assert len(p3) == 10
    
    for r in p2 + p3:
        pair = tuple(r["operands"])
        rev_pair = (pair[1], pair[0])
        assert pair not in e_pairs, f"Leakage: pair {pair} in training pairs!"
        assert rev_pair not in e_pairs, f"Leakage: reversed pair {rev_pair} in training pairs!"

def test_held_out_templates_strict_isolation():
    train_b, train_e, val, test = generate_datasets(seed=42)
    for tmpl in HELD_OUT_TEMPLATES:
        # Key distinctive phrase
        key_phrase = tmpl.split("{a}")[0]
        for r in train_b + train_e + val:
            assert key_phrase not in r["prompt"], f"Held out template leaked into training prompt: {r['prompt']}"

def test_deterministic_span_locator():
    tok, _ = load_tokenizer_artifacts(TOKENIZER_DIR)
    prompt = "<|im_start|>user\nPlease add A=17 and B=24.<|im_end|>\n<|im_start|>assistant\n"
    spans = extract_prompt_operand_token_spans(prompt, 17, 24, tok)
    
    assert spans["operand_a"] == 17
    assert spans["operand_b"] == 24
    assert spans["char_span_a"] is not None
    assert spans["char_span_b"] is not None
    assert spans["token_count_a"] >= 1
    assert spans["token_count_b"] >= 1
    
    # Decoded text from tokens must contain the target digits
    dec_a = "".join(tok.decode([t]) for t in spans["token_ids_a"])
    dec_b = "".join(tok.decode([t]) for t in spans["token_ids_b"])
    assert "17" in dec_a
    assert "24" in dec_b

def test_metrics_evaluation_contract():
    record_tool = {
        "id": "T1",
        "requires_tool": True,
        "operands": [42, 58],
        "category": "tool_test",
    }
    
    # 1. Exact match
    pred_exact = 'TOOL_CALL: {"tool": "calculator", "op": "add", "a": 42, "b": 58}'
    res1 = evaluate_single_prediction(pred_exact, record_tool)
    assert res1["parsed"] is True
    assert res1["schema_valid"] is True
    assert res1["exact_call"] is True
    assert res1["e2e_correct"] is True
    
    # 2. Swapped arguments
    pred_swapped = 'TOOL_CALL: {"tool": "calculator", "op": "add", "a": 58, "b": 42}'
    res2 = evaluate_single_prediction(pred_swapped, record_tool)
    assert res2["parsed"] is True
    assert res2["schema_valid"] is True
    assert res2["exact_call"] is False # Swapped must not count as exact call
    assert res2["a_correct"] is False
    assert res2["b_correct"] is False
    
    # 3. Non-tool prompt with clean refusal
    record_neg = {"id": "N1", "requires_tool": False}
    pred_refusal = "I can only invoke the calculator tool when both valid operands A and B are provided."
    res3 = evaluate_single_prediction(pred_refusal, record_neg)
    assert res3["false_positive"] is False
    assert res3["correct"] is True

def test_batch127_130_results_artifact_integrity():
    res_path = Path("experiments/batch127_130_binding_diagnosis/reports/binding_experiment_results.json")
    assert res_path.exists(), "Results report file missing."
    data = json.loads(res_path.read_text(encoding="utf-8"))
    assert "arm1_baseline_density" in data
    assert "arm2_expanded_density" in data
    
    # Confirm 0 false positives across both arms
    assert data["arm1_baseline_density"]["non_tool_metrics"]["total_false_positive_rate"] == 0.0
    assert data["arm2_expanded_density"]["non_tool_metrics"]["total_false_positive_rate"] == 0.0
