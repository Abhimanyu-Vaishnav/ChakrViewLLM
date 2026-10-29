"""
Unit tests for SFT dataset validator and quality gate.

Verifies:
1. Valid dataset passing with correct summary metrics.
2. Malformed JSON / missing fields detection.
3. Empty prompt / whitespace-only rejection.
4. Exact duplicate (prompt, response) pair detection.
5. Benchmark contamination / leakage rejection.
"""

from pathlib import Path
import json
import pytest

from chakrview.training.dataset_validator import (
    validate_sft_dataset_file,
    DatasetValidationError,
)


def test_validator_accepts_clean_dataset(tmp_path):
    data_file = tmp_path / "clean.jsonl"
    records = [
        {"prompt": f"Write a poem about star {i}", "response": f"Star {i} shines in the night sky."}
        for i in range(5)
    ]
    with open(data_file, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    summary = validate_sft_dataset_file(data_file, check_leakage=False)
    assert summary["total_records"] == 5
    assert summary["unique_prompts"] == 5
    assert summary["duplicate_pairs_count"] == 0


def test_validator_rejects_malformed_json(tmp_path):
    data_file = tmp_path / "bad_json.jsonl"
    data_file.write_text('{"prompt": "hello", "response": "world"}\n{bad_json}\n', encoding="utf-8")

    with pytest.raises(DatasetValidationError, match="Invalid JSON"):
        validate_sft_dataset_file(data_file, check_leakage=False)


def test_validator_rejects_missing_fields(tmp_path):
    data_file = tmp_path / "missing_field.jsonl"
    data_file.write_text('{"prompt": "hello"}\n', encoding="utf-8")

    with pytest.raises(DatasetValidationError, match="Missing required field 'response'"):
        validate_sft_dataset_file(data_file, check_leakage=False)


def test_validator_rejects_empty_prompt_or_response(tmp_path):
    data_file = tmp_path / "empty_p.jsonl"
    data_file.write_text('{"prompt": "   ", "response": "answer"}\n', encoding="utf-8")

    with pytest.raises(DatasetValidationError, match="empty or invalid 'prompt'"):
        validate_sft_dataset_file(data_file, check_leakage=False)


def test_validator_rejects_duplicate_pairs(tmp_path):
    data_file = tmp_path / "duplicates.jsonl"
    data_file.write_text(
        '{"prompt": "duplicate prompt", "response": "same response"}\n'
        '{"prompt": "unique prompt", "response": "another response"}\n'
        '{"prompt": "duplicate prompt", "response": "same response"}\n',
        encoding="utf-8",
    )

    with pytest.raises(DatasetValidationError, match="exact duplicate"):
        validate_sft_dataset_file(data_file, check_leakage=False)


def test_validator_detects_benchmark_leakage(tmp_path):
    data_file = tmp_path / "leaked.jsonl"
    # GREET-01 prompt from intelligence_tasks.json: "Hello! Who are you?"
    data_file.write_text(
        '{"prompt": "Hello! Who are you?", "response": "I am an assistant."}\n',
        encoding="utf-8",
    )

    with pytest.raises(DatasetValidationError, match="LEAKAGE DETECTED"):
        validate_sft_dataset_file(data_file, check_leakage=True)
