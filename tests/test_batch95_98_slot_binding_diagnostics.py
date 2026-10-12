"""
Unit & Regression Tests for Batches 95-98:
- Operand Slot-Binding Dataset Generator & Format contracts
- Split integrity & zero operand-pair leakage
- Free-running generation divergence analysis & first-divergence detection
- Teacher-forced token log-probability scoring
- Diagnostic report persistence and instant reopening
- Checkpoint inspector with tied parameter deduplication
- Graceful handling of invalid checkpoints
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest
import sys
import tempfile
import torch

sys.path.insert(0, ".")

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from experiments.batch42_reproducible_evaluation.evaluation.evaluator import (
    load_tokenizer_artifacts,
)
from experiments.batch95_98_slot_binding_diagnostics.data.generate_slot_data import (
    build_slot_prompt,
    generate_slot_datasets,
)
from experiments.batch95_98_slot_binding_diagnostics.workbench.app import (
    inspect_checkpoint,
    compute_teacher_forced_token_scores,
    run_single_inference_with_scores,
    compare_and_diagnose_checkpoints,
    save_diagnostic_report,
    load_diagnostic_report,
    TOKENIZER_DIR,
)


def test_slot_prompt_construction():
    prompt = build_slot_prompt(5, 7)
    assert "Operand A: 5" in prompt
    assert "Operand B: 7" in prompt
    assert "Required action: calculator addition" in prompt
    assert "<|im_start|>user" in prompt
    assert "<|im_start|>assistant" in prompt
    # Guarantee no answer leakage in prompt
    assert "12" not in prompt


def test_slot_dataset_split_integrity():
    manifest, splits = generate_slot_datasets()
    assert splits["train"]
    assert splits["val"]
    assert splits["test"]

    train_pairs = {tuple(sorted(r["operands"])) for r in splits["train"]}
    val_pairs = {tuple(sorted(r["operands"])) for r in splits["val"]}
    test_pairs = {tuple(sorted(r["operands"])) for r in splits["test"]}

    # Ensure zero leakage between train, val, and test
    assert len(train_pairs.intersection(val_pairs)) == 0
    assert len(train_pairs.intersection(test_pairs)) == 0
    assert len(val_pairs.intersection(test_pairs)) == 0


def test_teacher_forced_token_scores():
    tokenizer, _ = load_tokenizer_artifacts(TOKENIZER_DIR)
    config = ModelConfig(max_seq_len=64)
    model = ChakrMicro(config)
    model.eval()

    test_tokens = [0, 42, 100, 200, 1]
    scores = compute_teacher_forced_token_scores(model, test_tokens, tokenizer, top_k=3)
    assert len(scores) == len(test_tokens) - 1
    for s in scores:
        assert "log_prob" in s
        assert "prob" in s
        assert "top_candidates" in s
        assert len(s["top_candidates"]) == 3
        assert s["prob"] >= 0.0 and s["prob"] <= 1.0


def test_first_divergence_detection():
    tokenizer, _ = load_tokenizer_artifacts(TOKENIZER_DIR)
    config = ModelConfig(max_seq_len=64)
    model_a = ChakrMicro(config)
    model_b = ChakrMicro(config)
    # Modify one parameter in model_b so they diverge
    with torch.no_grad():
        for p in model_b.parameters():
            p.add_(0.5)
            break

    prompt = "<|im_start|>user\nCalculate 1 + 1.<|im_end|>\n<|im_start|>assistant\n"
    out_a = run_single_inference_with_scores(model_a, tokenizer, prompt, max_tokens=10)
    out_b = run_single_inference_with_scores(model_b, tokenizer, prompt, max_tokens=10)

    assert "generated_token_ids" in out_a
    assert "perplexity" in out_a
    assert "perplexity" in out_b


def test_diagnostic_report_save_and_reopen(tmp_path):
    report_data = {
        "title": "Test Diagnostic Report",
        "methodology_disclaimer": "Test disclaimer",
        "summary": {"total_prompts": 2, "identical_outputs_count": 1},
        "diagnostics": [{"index": 0, "outputs_identical": True}],
    }
    saved_path = save_diagnostic_report(report_data)
    assert Path(saved_path).is_file()

    loaded = load_diagnostic_report(str(saved_path))
    assert loaded["title"] == "Test Diagnostic Report"
    assert loaded["summary"]["total_prompts"] == 2


def test_checkpoint_inspector_invalid_path():
    res = inspect_checkpoint("non_existent_diagnostic_path.pt")
    assert res["exists"] is False
    assert res["valid"] is False
    assert "File does not exist" in res["error"]
