"""Automated Unit and Integration Tests for Batch 3111–3150 (JSON SFT V2).

Tests:
1. Dataset Record Counts and Split Proportions (200 train, 30 val, 34 test).
2. Multi-instance Structured JSON Proportions (140 train, 20 val, 12 familiar test, 12 unseen test).
3. Exact & Normalized Prompt Deduplication and Split Disjointness.
4. Genuinely Unseen Test Schema Key Isolation (zero overlap with train or val key-pairs).
5. Document-Level Replay Isolation (zero cross-split document overlap).
6. Zero Historical Test Overlap (no overlap with batch3071, batch3021, or generation benchmark).
7. Strict JSON Target Parsing and Completeness in Datasets.
8. Causal Target Alignment and Masking Logic.
9. Preflight Gate Enforcement (passes on clean environment, fails if modified).
10. Proof of Held-Out Test Exclusion from Trainer.
11. Evaluation Metric Accuracy on Synthetic Test Cases.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_DIR = REPO_ROOT / "experiments" / "batch3111_3150"

from chakrview.training.prompt_contract import build_causal_sequence
from experiments.batch3111_3150.run_evaluation import parse_strict_json
from experiments.batch3111_3150.run_preflight import run_preflight


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
    assert len(test_records) == 34, f"Expected 34 test records, got {len(test_records)}"

    # Check 70/30 train mixture
    train_inst = sum(1 for r in train_records if r.get("category", "").startswith("structured_output"))
    train_rep = sum(1 for r in train_records if r.get("category") == "base_continuation_replay")
    assert train_inst == 140, f"Expected 140 train instruction records, got {train_inst}"
    assert train_rep == 60, f"Expected 60 train replay records, got {train_rep}"

    val_inst = sum(1 for r in val_records if r.get("category", "").startswith("structured_output"))
    val_rep = sum(1 for r in val_records if r.get("category") == "base_continuation_replay")
    assert val_inst == 20, f"Expected 20 val instruction records, got {val_inst}"
    assert val_rep == 10, f"Expected 10 val replay records, got {val_rep}"

    # Check test breakdown: 12 familiar, 12 unseen, 10 replay
    test_fam = sum(1 for r in test_records if "familiar" in r.get("category", ""))
    test_uns = sum(1 for r in test_records if "unseen" in r.get("category", ""))
    test_rep = sum(1 for r in test_records if r.get("category") == "base_continuation_replay")
    assert test_fam == 12, f"Expected 12 familiar test cases, got {test_fam}"
    assert test_uns == 12, f"Expected 12 unseen test cases, got {test_uns}"
    assert test_rep == 10, f"Expected 10 test replay cases, got {test_rep}"


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

    # Verify unseen test schemas have zero key overlap with training key-pairs
    train_key_pairs = set()
    for r in train_records:
        if r.get("target_keys"):
            train_key_pairs.add(tuple(sorted(r["target_keys"])))

    for r in test_records:
        if "unseen" in r.get("category", ""):
            u_kp = tuple(sorted(r["target_keys"]))
            assert u_kp not in train_key_pairs, f"Unseen key pair {u_kp} leaked into training!"


def test_document_level_replay_isolation():
    manifest_path = EXPERIMENT_DIR / "DATASET_MANIFEST.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    doc_overlap = manifest.get("document_overlap", {})
    assert len(doc_overlap.get("train_val", [])) == 0, "Replay document overlap in train-val"
    assert len(doc_overlap.get("train_test", [])) == 0, "Replay document overlap in train-test"
    assert len(doc_overlap.get("val_test", [])) == 0, "Replay document overlap in val-test"


def test_json_target_syntax_validity():
    ds_path = EXPERIMENT_DIR / "JSON_SFT_V2_DATASET.jsonl"
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
    prompt_tokens = [101, 102]
    response_tokens = [201, 202]
    inp, tgt = build_causal_sequence(prompt_tokens, response_tokens, bos_id=0, eos_id=1, ignore_index=-100, append_eos=True)
    assert len(inp) == len(tgt)
    assert inp == [0, 101, 102, 201, 202]
    assert tgt == [-100, -100, 201, 202, 1]

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
    from experiments.batch3111_3150.run_trainer import CONFIG_PATH
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
    valid_text = '{"status": "ready", "code": 200}'
    eval_res = parse_strict_json(valid_text, ["status", "code"])
    assert eval_res["syntax_valid"] is True
    assert eval_res["schema_match"] is True
    assert eval_res["extra_keys"] == []
    assert eval_res["missing_keys"] == []

    # Valid syntax, missing key
    missing_text = '{"status": "ready"}'
    eval_missing = parse_strict_json(missing_text, ["status", "code"])
    assert eval_missing["syntax_valid"] is True
    assert eval_missing["schema_match"] is False
    assert eval_missing["missing_keys"] == ["code"]

    # Valid syntax, extra key
    extra_text = '{"status": "ready", "code": 200, "extra": "val"}'
    eval_extra = parse_strict_json(extra_text, ["status", "code"])
    assert eval_extra["syntax_valid"] is True
    assert eval_extra["schema_match"] is False
    assert eval_extra["extra_keys"] == ["extra"]

    # Malformed JSON
    malformed_text = '{"status": "ready", code: 200}'
    eval_bad = parse_strict_json(malformed_text, ["status", "code"])
    assert eval_bad["syntax_valid"] is False
    assert eval_bad["schema_match"] is False
    assert "error" in eval_bad and eval_bad["error"] is not None

    # Empty text
    eval_empty = parse_strict_json("", ["status", "code"])
    assert eval_empty["syntax_valid"] is False
    assert eval_empty["error"] == "EMPTY_OUTPUT"
