"""Unit tests for the v0.8 Single-Task JSON Evaluator.

Tests:
1. Valid JSON, correct schema, correct values -> All Pass.
2. Valid JSON, correct schema, WRONG status value -> Parse Pass, Schema Pass, Status Fail, Full Semantic Fail.
3. Valid JSON, correct schema, WRONG code value -> Parse Pass, Schema Pass, Code Fail, Full Semantic Fail.
4. Valid JSON, WRONG schema (extra keys, e.g. {"status": "ok", "code": 100, "extra": 1}) -> Schema Fail.
5. Valid JSON, WRONG schema (missing key, e.g. {"status": "ok"}) -> Schema Fail.
6. Valid JSON, WRONG types (e.g. {"status": 123, "code": "nominal"}) -> Schema Fail.
7. Boolean value for code (e.g. {"status": "nominal", "code": true}) -> Schema Fail (bool is subclass of int in python).
8. Malformed JSON (syntax error, e.g. {"status": "nominal", "code":) -> Parse Fail.
9. Empty string & whitespace string -> Rejection / Fail.
10. Markdown fenced JSON -> Handled and evaluated correctly.
11. Replay meaningful continuation threshold (repetition_ratio < 0.40).
12. Replay repetitive loop (repetition_ratio >= 0.40) -> Meaningful Fail.
"""

from __future__ import annotations

import pytest
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.research_v08_json_sft.evaluate_checkpoint import evaluate_output_case


def test_evaluator_perfect_success():
    res = evaluate_output_case(
        category="two_key_json_extraction",
        output_text='{"status": "standby_degraded", "code": 842}',
        expected_status="standby_degraded",
        expected_code=842,
    )
    assert res["json_parsed"] is True
    assert res["schema_valid"] is True
    assert res["status_match"] is True
    assert res["code_match"] is True
    assert res["full_semantic_success"] is True


def test_evaluator_wrong_status_value():
    res = evaluate_output_case(
        category="two_key_json_extraction",
        output_text='{"status": "nominal", "code": 842}',
        expected_status="standby_degraded",
        expected_code=842,
    )
    assert res["json_parsed"] is True
    assert res["schema_valid"] is True
    assert res["status_match"] is False
    assert res["code_match"] is True
    assert res["full_semantic_success"] is False


def test_evaluator_wrong_code_value():
    res = evaluate_output_case(
        category="two_key_json_extraction",
        output_text='{"status": "standby_degraded", "code": 105}',
        expected_status="standby_degraded",
        expected_code=842,
    )
    assert res["json_parsed"] is True
    assert res["schema_valid"] is True
    assert res["status_match"] is True
    assert res["code_match"] is False
    assert res["full_semantic_success"] is False


def test_evaluator_schema_extra_keys():
    res = evaluate_output_case(
        category="two_key_json_extraction",
        output_text='{"status": "standby_degraded", "code": 842, "extra_field": "hallucinated"}',
        expected_status="standby_degraded",
        expected_code=842,
    )
    assert res["json_parsed"] is True
    assert res["schema_valid"] is False
    assert res["full_semantic_success"] is False


def test_evaluator_schema_missing_key():
    res = evaluate_output_case(
        category="two_key_json_extraction",
        output_text='{"status": "standby_degraded"}',
        expected_status="standby_degraded",
        expected_code=842,
    )
    assert res["json_parsed"] is True
    assert res["schema_valid"] is False
    assert res["full_semantic_success"] is False


def test_evaluator_schema_wrong_types():
    res = evaluate_output_case(
        category="two_key_json_extraction",
        output_text='{"status": 999, "code": "standby_degraded"}',
        expected_status="standby_degraded",
        expected_code=842,
    )
    assert res["json_parsed"] is True
    assert res["schema_valid"] is False
    assert res["full_semantic_success"] is False


def test_evaluator_schema_boolean_code_rejection():
    # In Python, isinstance(True, int) is True, but booleans are invalid status codes!
    res = evaluate_output_case(
        category="two_key_json_extraction",
        output_text='{"status": "standby_degraded", "code": true}',
        expected_status="standby_degraded",
        expected_code=842,
    )
    assert res["json_parsed"] is True
    assert res["schema_valid"] is False
    assert res["full_semantic_success"] is False


def test_evaluator_malformed_syntax():
    res = evaluate_output_case(
        category="two_key_json_extraction",
        output_text='{"status": "standby_degraded", "code": ',
        expected_status="standby_degraded",
        expected_code=842,
    )
    assert res["json_parsed"] is False
    assert res["schema_valid"] is False
    assert res["full_semantic_success"] is False


def test_evaluator_empty_and_first_eos():
    res_empty = evaluate_output_case(
        category="two_key_json_extraction",
        output_text="",
        is_empty=True,
    )
    assert res_empty["json_parsed"] is False
    assert res_empty["full_semantic_success"] is False

    res_eos = evaluate_output_case(
        category="two_key_json_extraction",
        output_text="",
        first_token_is_eos=True,
    )
    assert res_eos["json_parsed"] is False
    assert res_eos["full_semantic_success"] is False


def test_evaluator_markdown_code_fencing():
    fenced = '```json\n{"status": "standby_degraded", "code": 842}\n```'
    res = evaluate_output_case(
        category="two_key_json_extraction",
        output_text=fenced,
        expected_status="standby_degraded",
        expected_code=842,
    )
    assert res["json_parsed"] is True
    assert res["schema_valid"] is True
    assert res["status_match"] is True
    assert res["code_match"] is True
    assert res["full_semantic_success"] is True


def test_evaluator_replay_metrics():
    # Good continuation
    res_good = evaluate_output_case(
        category="base_continuation_replay",
        output_text="proportional to their distance in flat spacetime.",
        repetition_ratio=0.10,
        is_empty=False,
    )
    assert res_good["replay_non_empty"] is True
    assert res_good["replay_meaningful"] is True

    # Repetitive loop continuation (ratio >= 0.40)
    res_loop = evaluate_output_case(
        category="base_continuation_replay",
        output_text="the world of the world of the world of the world of the world",
        repetition_ratio=0.55,
        is_empty=False,
    )
    assert res_loop["replay_non_empty"] is True
    assert res_loop["replay_meaningful"] is False

    # Empty replay
    res_empty = evaluate_output_case(
        category="base_continuation_replay",
        output_text="",
        is_empty=True,
    )
    assert res_empty["replay_non_empty"] is False
    assert res_empty["replay_meaningful"] is False
