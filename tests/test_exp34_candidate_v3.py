"""
Unit and regression tests for Exp34 Candidate V3 SFT dataset diversity, schema compliance,
and reproducible generation.
"""

import hashlib
import json
from pathlib import Path

import pytest
from chakrview.training.dataset_validator import validate_sft_dataset_file
from chakrview.training.sft_engine import SFTDataset, load_tokenizer_artifacts


CANDIDATE_V3_JSONL = Path("data/sft/exp34_candidate_v3/exp34_instruction_mixture_train_v3.jsonl")
CANDIDATE_V3_MANIFEST = Path("data/sft/exp34_candidate_v3/exp34_instruction_mixture_manifest_v3.json")
EXPECTED_SHA256 = "e09439b7aee8c1f56c9d6a105f062a76f02c04b106b1f0c8e4a4a8a5d277848d"


def test_candidate_v3_files_exist():
    assert CANDIDATE_V3_JSONL.exists(), f"Missing {CANDIDATE_V3_JSONL}"
    assert CANDIDATE_V3_MANIFEST.exists(), f"Missing {CANDIDATE_V3_MANIFEST}"


def test_candidate_v3_exact_sha256():
    with open(CANDIDATE_V3_JSONL, "rb") as f:
        actual_sha256 = hashlib.sha256(f.read()).hexdigest()
    assert actual_sha256 == EXPECTED_SHA256, (
        f"Candidate V3 SHA-256 mismatch! Expected {EXPECTED_SHA256}, got {actual_sha256}"
    )


def test_candidate_v3_manifest_consistency():
    with open(CANDIDATE_V3_MANIFEST, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    assert manifest["jsonl_sha256"] == EXPECTED_SHA256
    assert manifest["total_records"] == 1000
    assert manifest["unique_prompts_count"] == 1000
    assert manifest["unique_responses_count"] == 1000
    assert len(manifest["category_counts"]) == 10
    assert all(count == 100 for count in manifest["category_counts"].values())


def test_candidate_v3_validator_pass():
    result = validate_sft_dataset_file(str(CANDIDATE_V3_JSONL))
    assert result["total_records"] == 1000
    assert result["unique_prompts"] == 1000
    assert result["unique_pairs"] == 1000
    assert result["duplicate_prompts_count"] == 0
    assert result["duplicate_pairs_count"] == 0
    assert result["leakage_checked"] is True


def test_candidate_v3_elimination_of_v1_and_v2_repetition_patterns():
    with open(CANDIDATE_V3_JSONL, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f]

    assert len(records) == 1000
    prompts = [r["prompt"] for r in records]
    responses = [r["response"] for r in records]

    # Global uniqueness
    assert len(set(prompts)) == 1000
    assert len(set(responses)) == 1000

    # 1. itertools.groupby collapse from V1 (was 60, now <= 1)
    groupby_count = sum(1 for r in responses if "groupby" in r.lower())
    assert groupby_count <= 1, f"Expected <= 1 groupby responses, got {groupby_count}"

    # 2. mutable default argument cache={} collapse from V2 (was 80, now <= 1)
    mutable_def_count = sum(1 for r in responses if "mutable default" in r.lower())
    assert mutable_def_count <= 1, f"Expected <= 1 mutable default responses, got {mutable_def_count}"

    # 3. BST DFS validation collapse from V2 (was 80, now <= 1)
    bst_dfs_count = sum(1 for r in responses if "bst property" in r.lower() or "min_val" in r.lower())
    assert bst_dfs_count <= 1, f"Expected <= 1 BST DFS responses, got {bst_dfs_count}"

    # 4. Conversational boilerplate tail from V2 (was 100, now 0)
    conv_tail_count = sum(1 for r in responses if "predictable and reproducible model behavior" in r.lower())
    assert conv_tail_count == 0, f"Expected 0 boilerplate tail responses, got {conv_tail_count}"

    # 5. Pipeline buffer formatting collapse from V2 (was 25, now 0)
    pipeline_count = sum(1 for r in responses if "pipeline_buffer" in r.lower())
    assert pipeline_count == 0, f"Expected 0 pipeline buffer responses, got {pipeline_count}"


def test_candidate_v3_sft_engine_loading():
    tok, _ = load_tokenizer_artifacts(Path("data/experiments/vocab_4096"))
    ds = SFTDataset(
        CANDIDATE_V3_JSONL,
        tokenizer=tok,
        max_seq_len=512,
        mask_prompt=True,
        is_training=True,
    )
    assert len(ds) == 1000

    rec = ds.records[0]
    inp, tgt = ds.encode_example(rec["prompt"], rec["response"])
    assert len(inp) == len(tgt)
    assert tgt[0] == -100
    assert tgt[-1] == 1  # EOS
    assert tgt.count(-100) > 0
    assert sum(1 for x in tgt if x != -100) > 0

