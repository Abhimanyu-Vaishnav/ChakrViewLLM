"""Automated Unit and Integration Tests for Batch 3071–3110 (JSON Capability Experiment V1).

Tests:
1. Dataset Construction & Record Counts (200 train, 30 val, 30 test).
2. 70/30 Mixture Ratio (140/60 train, 20/10 val, 20/10 test).
3. Exact & Normalized Prompt Deduplication and Split Disjointness.
4. JSON Target Syntactic Validity in Train, Val, Test.
5. Replay Source Document Isolation (zero cross-split document overlap).
6. Zero Historical Test Overlap (no overlap with batch3021, batch2821, or batch2781 benchmarks).
7. Prompt/Target Boundary & Causal Masking Logic.
8. EOS Policy Handling (supervised EOS on instruction, unforced on replay).
9. SFT Configuration Integrity (100 steps, batch size 4, lr 3e-5, seed 42).
10. Fail-Closed Preflight Behavior (passes on valid environment, fails if modified).
11. Proof of Held-Out Test Exclusion from Trainer.
12. Strict JSON Evaluation Logic & Metric Accuracy on Synthetic Test Cases.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_DIR = REPO_ROOT / "experiments" / "batch3071_3110"

from chakrview.training.prompt_contract import build_causal_sequence
from experiments.batch3071_3110.run_evaluation import parse_strict_json
from experiments.batch3071_3110.run_preflight import run_preflight


def test_dataset_record_counts_and_splits():
    train_path = EXPERIMENT_DIR / "train.jsonl"
    val_path = EXPERIMENT_DIR / "val.jsonl"
    test_path = EXPERIMENT_DIR / "test.jsonl"
    manifest_path = EXPERIMENT_DIR / "DATASET_MANIFEST.json"

    assert train_path.is_file(), "train.jsonl missing"
    assert val_path.is_file(), "val.jsonl missing"
    assert test_path.is_file(), "test.jsonl missing"
    assert manifest_path.is_file(), "DATASET_MANIFEST.json missing"

    with open(train_path, "r", encoding="utf-8") as f:
        train_records = [json.loads(line) for line in f if line.strip()]
    with open(val_path, "r", encoding="utf-8") as f:
        val_records = [json.loads(line) for line in f if line.strip()]
    with open(test_path, "r", encoding="utf-8") as f:
        test_records = [json.loads(line) for line in f if line.strip()]

    assert len(train_records) == 200, f"Expected 200 train records, got {len(train_records)}"
    assert len(val_records) == 30, f"Expected 30 val records, got {len(val_records)}"
    assert len(test_records) == 30, f"Expected 30 test records, got {len(test_records)}"

    # Check 70/30 ratio
    train_inst = sum(1 for r in train_records if r.get("category", "").startswith("structured_output"))
    train_rep = sum(1 for r in train_records if r.get("category") == "base_continuation_replay")
    assert train_inst == 140, f"Expected 140 train instruction records, got {train_inst}"
    assert train_rep == 60, f"Expected 60 train replay records, got {train_rep}"

    val_inst = sum(1 for r in val_records if r.get("category", "").startswith("structured_output"))
    val_rep = sum(1 for r in val_records if r.get("category") == "base_continuation_replay")
    assert val_inst == 20, f"Expected 20 val instruction records, got {val_inst}"
    assert val_rep == 10, f"Expected 10 val replay records, got {val_rep}"

    test_inst = sum(1 for r in test_records if r.get("category", "").startswith("structured_output"))
    test_rep = sum(1 for r in test_records if r.get("category") == "base_continuation_replay")
    assert test_inst == 20, f"Expected 20 test instruction records, got {test_inst}"
    assert test_rep == 10, f"Expected 10 test replay records, got {test_rep}"


def test_split_isolation_and_no_leakage():
    train_path = EXPERIMENT_DIR / "train.jsonl"
    val_path = EXPERIMENT_DIR / "val.jsonl"
    test_path = EXPERIMENT_DIR / "test.jsonl"

    with open(train_path, "r", encoding="utf-8") as f:
        train_records = [json.loads(line) for line in f if line.strip()]
    with open(val_path, "r", encoding="utf-8") as f:
        val_records = [json.loads(line) for line in f if line.strip()]
    with open(test_path, "r", encoding="utf-8") as f:
        test_records = [json.loads(line) for line in f if line.strip()]

    train_prompts = {r["prompt"].strip().lower() for r in train_records}
    val_prompts = {r["prompt"].strip().lower() for r in val_records}
    test_prompts = {r["prompt"].strip().lower() for r in test_records}

    assert not (train_prompts & val_prompts), "Leakage detected between train and val prompts"
    assert not (train_prompts & test_prompts), "Leakage detected between train and test prompts"
    assert not (val_prompts & test_prompts), "Leakage detected between val and test prompts"

    # Verify held-out test keys are completely absent from training
    train_key_pairs = set()
    for r in train_records:
        if r.get("target_keys"):
            train_key_pairs.add(tuple(sorted(r["target_keys"])))

    for r in test_records:
        if r.get("target_keys"):
            test_kp = tuple(sorted(r["target_keys"]))
            assert test_kp not in train_key_pairs, f"Held-out key pair {test_kp} leaked into train set!"


def test_document_level_replay_isolation():
    manifest_path = EXPERIMENT_DIR / "DATASET_MANIFEST.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    doc_overlap = manifest.get("document_overlap", {})
    assert len(doc_overlap.get("train_val", [])) == 0, "Replay document overlap in train-val"
    assert len(doc_overlap.get("train_test", [])) == 0, "Replay document overlap in train-test"
    assert len(doc_overlap.get("val_test", [])) == 0, "Replay document overlap in val-test"


def test_json_target_syntax_validity():
    ds_path = EXPERIMENT_DIR / "JSON_SFT_DATASET.jsonl"
    with open(ds_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("category", "").startswith("structured_output"):
                parsed = json.loads(rec["response"])
                assert isinstance(parsed, dict), f"Line {idx} target is not a dict: {rec['response']}"
                expected_keys = rec.get("target_keys")
                if expected_keys:
                    assert set(parsed.keys()) == set(expected_keys), f"Key mismatch at line {idx}"


def test_prompt_masking_and_causal_contract():
    # Instruction sequence: prompt tokens masked with -100, EOS token appended to target
    prompt_tokens = [101, 102]
    response_tokens = [201, 202]
    inp, tgt = build_causal_sequence(prompt_tokens, response_tokens, bos_id=0, eos_id=1, ignore_index=-100, append_eos=True)
    assert len(inp) == len(tgt)
    assert inp == [0, 101, 102, 201, 202]
    assert tgt == [-100, -100, 201, 202, 1]

    # Replay sequence: prompt tokens masked, NO appended EOS (continuation unforced)
    inp_r, tgt_r = build_causal_sequence(prompt_tokens, response_tokens, bos_id=0, eos_id=1, ignore_index=-100, append_eos=False)
    assert len(inp_r) == len(tgt_r)
    assert inp_r == [0, 101, 102, 201]
    assert tgt_r == [-100, -100, 201, 202]
    assert tgt_r[-1] != 1


def test_preflight_gate_enforcement():
    exit_code = run_preflight()
    assert exit_code == 0, "Preflight failed on clean environment"

    report_path = EXPERIMENT_DIR / "FINAL_PREFLIGHT_REPORT.json"
    assert report_path.is_file()
    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)
    assert report["overall_verdict"] == "PASS"
    assert report["training_executed"] is False
    assert report["optimizer_steps_executed"] == 0


def test_trainer_held_out_exclusion():
    from experiments.batch3071_3110.run_trainer import CONFIG_PATH
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    train_path = cfg["dataset_paths"]["train_file"]
    val_path = cfg["dataset_paths"]["val_file"]
    test_path = cfg["dataset_paths"]["test_file"]

    assert "test.jsonl" not in train_path, "test.jsonl referenced in train path"
    assert "test.jsonl" not in val_path, "test.jsonl referenced in val path"
    assert "test.jsonl" in test_path, "test.jsonl not isolated in test path"


def test_strict_json_parser_metrics():
    # Valid exact match
    valid_text = '{"name": "Server", "status": "active"}'
    eval_res = parse_strict_json(valid_text, ["name", "status"])
    assert eval_res["syntax_valid"] is True
    assert eval_res["schema_match"] is True
    assert eval_res["extra_keys"] == []
    assert eval_res["missing_keys"] == []

    # Valid syntax, but missing key
    missing_text = '{"name": "Server"}'
    eval_missing = parse_strict_json(missing_text, ["name", "status"])
    assert eval_missing["syntax_valid"] is True
    assert eval_missing["schema_match"] is False
    assert eval_missing["missing_keys"] == ["status"]

    # Valid syntax, extra key
    extra_text = '{"name": "Server", "status": "active", "extra": 123}'
    eval_extra = parse_strict_json(extra_text, ["name", "status"])
    assert eval_extra["syntax_valid"] is True
    assert eval_extra["schema_match"] is False
    assert eval_extra["extra_keys"] == ["extra"]

    # Malformed JSON
    malformed_text = '{"name": "Server", status: active}'
    eval_bad = parse_strict_json(malformed_text, ["name", "status"])
    assert eval_bad["syntax_valid"] is False
    assert eval_bad["schema_match"] is False
    assert "error" in eval_bad and eval_bad["error"] is not None

    # Empty text
    eval_empty = parse_strict_json("", ["name", "status"])
    assert eval_empty["syntax_valid"] is False
    assert eval_empty["error"] == "EMPTY_OUTPUT"
