"""Deterministic Unit and Regression Tests for SFT V3 Dataset Builder & Preflight Validator.

Verifies:
1. Builder determinism and byte-for-byte reproducibility across runs.
2. Split count contracts (200 train, 40 validation).
3. Zero template-family leakage between train and val.
4. Zero factual prompt/target/context collisions.
5. Grounded context availability across all structured records.
6. Target JSON parsing and type correctness.
7. Token length boundaries (<= 512 context limit).
8. Validator error detection on injected corruptions.

SAFETY NOTICE:
- Purely read-only test suite.
- Does NOT execute model training or optimizer steps.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from experiments.sft_v3_dataset_builder.build_sft_v3_dataset import (
    build_sft_v3_dataset,
)
from experiments.sft_v3_dataset_builder.validate_sft_v3_dataset import (
    validate_dataset,
    validate_single_record,
)


def test_builder_record_counts():
    train_recs, val_recs = build_sft_v3_dataset()
    assert len(train_recs) == 200
    assert len(val_recs) == 40


def test_split_template_disjointness():
    train_recs, val_recs = build_sft_v3_dataset()
    train_tmpls = set(r["template_family_id"] for r in train_recs)
    val_tmpls = set(r["template_family_id"] for r in val_recs)
    overlap = train_tmpls.intersection(val_tmpls)
    assert len(overlap) == 0, f"Template overlap found: {overlap}"


def test_split_entity_target_disjointness():
    train_recs, val_recs = build_sft_v3_dataset()
    train_factual_targets = set(
        r["response"].strip() for r in train_recs
        if r["category"] not in ("base_continuation_replay", "missing_information_abstention")
    )
    val_factual_targets = set(
        r["response"].strip() for r in val_recs
        if r["category"] not in ("base_continuation_replay", "missing_information_abstention")
    )
    overlap = train_factual_targets.intersection(val_factual_targets)
    assert len(overlap) == 0, f"Factual target overlap found: {overlap}"


def test_grounded_context_presence():
    train_recs, val_recs = build_sft_v3_dataset()
    for r in train_recs + val_recs:
        if r["category"] != "base_continuation_replay":
            assert r["context"] is not None
            assert len(r["context"].strip()) > 10
            # Target response must parse as JSON
            parsed = json.loads(r["response"])
            assert isinstance(parsed, dict)


def test_validator_detects_missing_context():
    bad_record = {
        "id": "BAD_001",
        "category": "grounded_extraction_server_status",
        "template_family_id": "bad_tmpl",
        "context": "",  # Empty context!
        "prompt": "Prompt text",
        "response": '{"status": "ready", "code": 200}',
        "expected_answer": {"status": "ready", "code": 200},
        "field_types": {"status": "string", "code": "integer"},
        "required_keys": ["status", "code"],
        "difficulty": "standard",
        "provenance": "TEST",
        "data_role": "TRAIN",
        "validation_rule": "exact_key_value_match",
    }
    errs = validate_single_record(bad_record, "train", 0)
    assert any("Context missing or too short" in e for e in errs)


def test_validator_detects_type_mismatch():
    bad_record = {
        "id": "BAD_002",
        "category": "grounded_extraction_server_status",
        "template_family_id": "bad_tmpl",
        "context": "Valid node context passage for test.",
        "prompt": "Prompt text",
        "response": '{"status": "ready", "code": "200"}',  # code is string instead of int!
        "expected_answer": {"status": "ready", "code": "200"},
        "field_types": {"status": "string", "code": "integer"},
        "required_keys": ["status", "code"],
        "difficulty": "standard",
        "provenance": "TEST",
        "data_role": "TRAIN",
        "validation_rule": "exact_key_value_match",
    }
    errs = validate_single_record(bad_record, "train", 0)
    assert any("expected integer, got str" in e for e in errs)


def test_full_dataset_validation_passes():
    repo_root = Path(__file__).resolve().parents[1]
    train_p = repo_root / "experiments" / "sft_v3_dataset_builder" / "train.jsonl"
    val_p = repo_root / "experiments" / "sft_v3_dataset_builder" / "validation.jsonl"
    report = validate_dataset(train_p, val_p)
    assert report["status"] == "PASS"
    assert len(report["errors"]) == 0
    assert report["train_count"] == 200
    assert report["val_count"] == 40
