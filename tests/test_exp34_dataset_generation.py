"""
Unit tests for Exp34 dataset generator and compiled dataset integrity.

Verifies:
1. Generator reproducibility: two independent runs produce bit-exact identical JSONL and manifest.
2. Target count constraint: exactly 1,000 valid records.
3. Category balance: exactly 100 records in each of the 10 distinct categories.
4. Schema validation: all rows conform to SFTDataset requirements.
5. Zero duplicate prompts and zero duplicate (prompt, response) pairs.
6. Zero benchmark leakage against intelligence and coding evaluation tasks.
"""

import hashlib
import json
from pathlib import Path
import pytest

from chakrview.training.dataset_validator import validate_sft_dataset_file
from scripts.compile_exp34_dataset import compile_1000_sft_dataset, write_verified_dataset


def test_generator_deterministic_reproducibility(tmp_path):
    """Verifies that compiling the dataset twice produces bit-exact identical files."""
    run1_jsonl = tmp_path / "run1.jsonl"
    run1_manifest = tmp_path / "run1.json"
    run2_jsonl = tmp_path / "run2.jsonl"
    run2_manifest = tmp_path / "run2.json"

    recs1 = compile_1000_sft_dataset()
    m1 = write_verified_dataset(recs1, run1_jsonl, run1_manifest, seed=42)

    recs2 = compile_1000_sft_dataset()
    m2 = write_verified_dataset(recs2, run2_jsonl, run2_manifest, seed=42)

    assert run1_jsonl.read_bytes() == run2_jsonl.read_bytes(), "JSONL bytes differ across runs!"
    assert run1_manifest.read_bytes() == run2_manifest.read_bytes(), "Manifest bytes differ across runs!"
    assert m1["jsonl_sha256"] == m2["jsonl_sha256"]


def test_dataset_category_distribution():
    """Verifies that all 10 categories are represented with exactly 100 records each."""
    recs = compile_1000_sft_dataset()
    assert len(recs) == 1000

    expected_categories = {
        "technical_explanations",
        "python_programming",
        "code_debugging",
        "algorithmic_reasoning",
        "arithmetic_reasoning",
        "passage_summarization",
        "constrained_formatting",
        "conversational_clarification",
        "hindi_comprehension",
        "bilingual_hinglish",
    }

    counts = {}
    for r in recs:
        c = r["category"]
        counts[c] = counts.get(c, 0) + 1

    assert set(counts.keys()) == expected_categories
    for c, count in counts.items():
        assert count == 100, f"Category '{c}' has {count} records, expected exactly 100"


def test_compiled_exp34_dataset_file_passes_validator():
    """Validates the actual on-disk dataset at data/sft/exp34_instruction_mixture_train.jsonl."""
    dataset_path = Path("data/sft/exp34_instruction_mixture_train.jsonl")
    assert dataset_path.exists(), "Target dataset file does not exist!"

    res = validate_sft_dataset_file(dataset_path, check_leakage=True)
    assert res["total_records"] == 1000
    assert res["unique_prompts"] == 1000
    assert res["unique_pairs"] == 1000
    assert res["duplicate_prompts_count"] == 0
    assert res["duplicate_pairs_count"] == 0
    assert res["leakage_checked"] is True

