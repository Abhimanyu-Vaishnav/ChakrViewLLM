"""
Unit and regression tests for Exp34 Candidate V2 SFT dataset diversity, schema compliance,
and reproducible generation.
"""

import hashlib
import json
from pathlib import Path

import pytest
from chakrview.training.dataset_validator import validate_sft_dataset_file
from chakrview.training.sft_engine import SFTDataset, load_tokenizer_artifacts


CANDIDATE_V2_JSONL = Path("data/sft/exp34_candidate_v2/exp34_instruction_mixture_train_v2.jsonl")
CANDIDATE_V2_MANIFEST = Path("data/sft/exp34_candidate_v2/exp34_instruction_mixture_manifest_v2.json")
CANDIDATE_V2_REPORT = Path("data/sft/exp34_candidate_v2/exp34_instruction_mixture_quality_report_v2.md")
EXPECTED_SHA256 = "6836ff38e16d012e8577c77171d8853bc59e44cb5cb1d8a14af4dbf217dd2010"


def test_candidate_v2_files_exist():
    assert CANDIDATE_V2_JSONL.exists(), f"Missing {CANDIDATE_V2_JSONL}"
    assert CANDIDATE_V2_MANIFEST.exists(), f"Missing {CANDIDATE_V2_MANIFEST}"
    assert CANDIDATE_V2_REPORT.exists(), f"Missing {CANDIDATE_V2_REPORT}"


def test_candidate_v2_exact_sha256():
    with open(CANDIDATE_V2_JSONL, "rb") as f:
        actual_sha256 = hashlib.sha256(f.read()).hexdigest()
    assert actual_sha256 == EXPECTED_SHA256, (
        f"Candidate V2 SHA-256 mismatch! Expected {EXPECTED_SHA256}, got {actual_sha256}"
    )


def test_candidate_v2_manifest_consistency():
    with open(CANDIDATE_V2_MANIFEST, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    assert manifest["jsonl_sha256"] == EXPECTED_SHA256
    assert manifest["total_records"] == 1000
    assert manifest["unique_prompts_count"] == 1000
    assert manifest["unique_responses_count"] == 1000
    assert len(manifest["category_counts"]) == 10
    assert all(count == 100 for count in manifest["category_counts"].values())


def test_candidate_v2_validator_pass():
    result = validate_sft_dataset_file(str(CANDIDATE_V2_JSONL))
    assert result["total_records"] == 1000
    assert result["unique_prompts"] == 1000
    assert result["unique_pairs"] == 1000
    assert result["duplicate_prompts_count"] == 0
    assert result["duplicate_pairs_count"] == 0
    assert result["leakage_checked"] is True


def test_candidate_v2_perfect_uniqueness_and_no_groupby_collapse():
    with open(CANDIDATE_V2_JSONL, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f]

    assert len(records) == 1000
    prompts = [r["prompt"] for r in records]
    responses = [r["response"] for r in records]

    # Exactly 1,000 distinct prompts and 1,000 distinct responses
    assert len(set(prompts)) == 1000
    assert len(set(responses)) == 1000

    # Ensure the v1 groupby collapse (60 identical responses) is resolved (at most 1)
    groupby_count = sum(1 for r in responses if "groupby" in r.lower())
    assert groupby_count <= 1, f"Found {groupby_count} groupby responses, expected <= 1"


def test_candidate_v2_sft_engine_loading():
    tok, _ = load_tokenizer_artifacts(Path("data/experiments/vocab_4096"))
    ds = SFTDataset(
        CANDIDATE_V2_JSONL,
        tokenizer=tok,
        max_seq_len=512,
        mask_prompt=True,
        is_training=True,
    )
    assert len(ds) == 1000

    # Verify encoding of the first example
    rec = ds.records[0]
    inp, tgt = ds.encode_example(rec["prompt"], rec["response"])
    assert len(inp) == len(tgt)
    assert tgt[0] == -100
    assert tgt[-1] == 1  # EOS
    assert tgt.count(-100) > 0
    assert sum(1 for x in tgt if x != -100) > 0

