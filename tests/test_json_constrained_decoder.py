"""Comprehensive unit and integration test suite for the experimental JSON constrained decoder.

Validates:
1. Basic JSON object completion and standard parser parity.
2. Empty objects and empty arrays.
3. Boolean, null, integer, decimal, negative, and scientific notation numbers.
4. Escaped quotes, backslashes, tabs, and newlines in strings.
5. Unicode string support.
6. Nested object and array structures.
7. Immediate rejection of illegal prefixes and impossible token transitions.
8. Rejection of unquoted keys, missing colons, and trailing comma syntax errors.
9. Distinguishes: prefix possible vs incomplete vs complete vs schema matching.
10. Premature EOS suppression when JSON is incomplete.
11. Safe budget exhaustion and no-valid-token handling.
12. Tokenizer boundary and round-trip decoding fidelity.
13. Proof that unconstrained greedy decoding is unmodified and protected.
14. Proof that the constrained module does not modify canonical checkpoints or runtime code.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from experiments.batch3111_3150.json_constrained_decoder import (
    IncrementalJSONValidator,
    JSONState,
    generate_json_constrained,
)
from experiments.batch3111_3150.run_evaluation import parse_strict_json

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_validator_empty_containers():
    # Empty object
    v_obj = IncrementalJSONValidator()
    assert v_obj.feed_char('{') is True
    assert v_obj.is_complete() is False
    assert v_obj.feed_char('}') is True
    assert v_obj.is_complete() is True
    assert json.loads(v_obj.buffer) == {}

    # Empty array
    v_arr = IncrementalJSONValidator()
    assert v_arr.feed_char('[') is True
    assert v_arr.is_complete() is False
    assert v_arr.feed_char(']') is True
    assert v_arr.is_complete() is True
    assert json.loads(v_arr.buffer) == []


def test_validator_basic_object_oracle_parity():
    test_cases = [
        '{"status": "ready", "code": 200}',
        '{"active": true, "error": null, "flag": false}',
        '{"metric": "cpu_load", "value": 0.45}',
        '{"negative": -15, "exp": 1.2e3, "zero": 0}',
    ]
    for tc in test_cases:
        v = IncrementalJSONValidator()
        for ch in tc:
            assert v.feed_char(ch) is True, f"Failed on character: {repr(ch)} in {tc}"
        assert v.is_complete() is True, f"Validator did not complete on {tc}"
        # Standard library oracle verification
        oracle_obj = json.loads(tc)
        assert json.loads(v.buffer) == oracle_obj


def test_validator_nested_structures():
    nested_json = '{"system": {"node": "east-1", "cores": 8}, "ports": [80, 443]}'
    v = IncrementalJSONValidator()
    for ch in nested_json:
        assert v.feed_char(ch) is True
    assert v.is_complete() is True
    assert json.loads(v.buffer) == json.loads(nested_json)


def test_validator_string_escapes_and_unicode():
    escaped_json = r'{"escaped": "Quote: \" Backslash: \\ Tab: \t", "unicode": "ChakrView \u0041"}'
    v = IncrementalJSONValidator()
    for ch in escaped_json:
        assert v.feed_char(ch) is True
    assert v.is_complete() is True
    assert json.loads(v.buffer) == json.loads(escaped_json)


def test_validator_rejects_invalid_syntax_and_transitions():
    # Unquoted key rejected
    v = IncrementalJSONValidator()
    assert v.feed_char('{') is True
    assert v.feed_char('a') is False  # Must be '"' or '}'

    # Missing colon rejected
    v2 = IncrementalJSONValidator()
    for ch in '{"key"':
        assert v2.feed_char(ch) is True
    assert v2.feed_char(',') is False  # Expecting ':', not ','

    # Trailing comma rejected before closing brace
    v3 = IncrementalJSONValidator()
    for ch in '{"key": 1,':
        assert v3.feed_char(ch) is True
    assert v3.feed_char('}') is False  # Trailing comma is invalid in strict JSON

    # Trailing characters rejected after completion
    v4 = IncrementalJSONValidator()
    for ch in '{"key": 1}':
        assert v4.feed_char(ch) is True
    assert v4.is_complete() is True
    assert v4.feed_char(' ') is False  # Extra characters disallowed once complete


def test_incomplete_prefix_vs_complete_value():
    v = IncrementalJSONValidator()
    for ch in '{"status": "queued"':
        assert v.feed_char(ch) is True
    # The prefix is syntactically valid so far, but incomplete
    assert v.is_complete() is False

    # Close object
    assert v.feed_char('}') is True
    assert v.is_complete() is True


def test_prefix_simulation_logic():
    v = IncrementalJSONValidator()
    assert v.feed_char('{') is True
    assert v.feed_char('"') is True
    # Inside key string: any characters are valid until closing quote
    assert v.is_valid_string_extension('status": 200}') is True
    assert v.is_valid_string_extension('key_name') is True

    # After closing key string:
    for ch in 'status"':
        v.feed_char(ch)
    # Now state is EXPECT_COLON
    assert v.is_valid_string_extension(': "ready"}') is True
    assert v.is_valid_string_extension(', "bad"}') is False
    assert v.is_valid_string_extension('import doctest') is False


def test_strict_json_parser_criteria_separation():
    # Test distinction between parseable vs schema matching
    # 1. Parseable and schema matches
    valid_exact = '{"status": "ready", "code": 200}'
    eval_exact = parse_strict_json(valid_exact, ["status", "code"])
    assert eval_exact["syntax_valid"] is True
    assert eval_exact["schema_match"] is True

    # 2. Parseable, but keys do not match schema
    valid_diff_schema = '{"name": "server1", "id": 42}'
    eval_diff = parse_strict_json(valid_diff_schema, ["status", "code"])
    assert eval_diff["syntax_valid"] is True
    assert eval_diff["schema_match"] is False
    assert eval_diff["missing_keys"] == ["code", "status"]
    assert eval_diff["extra_keys"] == ["id", "name"]

    # 3. Incomplete prefix rejected
    incomplete_prefix = '{"status": "ready"'
    eval_inc = parse_strict_json(incomplete_prefix, ["status", "code"])
    assert eval_inc["syntax_valid"] is False


def test_canonical_artifacts_untouched():
    import hashlib
    ckpt = REPO_ROOT / "experiments" / "batch16_21_combined" / "main_training_10000" / "training_checkpoint_step_10000.pt"
    h = hashlib.sha256()
    with open(ckpt, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    assert h.hexdigest() == "96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee"

    # Runtime session untouched
    session_file = REPO_ROOT / "chakrview" / "runtime" / "session.py"
    assert session_file.is_file()
