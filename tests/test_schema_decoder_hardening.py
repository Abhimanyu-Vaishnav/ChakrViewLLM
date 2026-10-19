"""Deterministic tests for deadlocks, trailing commas, whitespace, and edge cases."""

import json
import pytest
import torch

from experiments.batch3111_3150.schema_prefix_trie import (
    JSONSchema,
    JSONType,
    PrefixTrie,
    SchemaConstrainedJSONValidator,
    generate_schema_constrained,
)
from tests.test_fake_model_schema_decoder import FakeModel, FakeTokenizer


def test_comma_disallowed_when_no_remaining_keys():
    """Regression test: prevents trailing comma deadlock when all schema keys are emitted."""
    schema = JSONSchema.from_keys(required_keys=["feature", "enabled"])
    val = SchemaConstrainedJSONValidator(schema)

    # Feed '{"feature": "val", "enabled": true'
    prefix = '{"feature": "val", "enabled": true'
    for c in prefix:
        assert val.feed_char(c) is True

    # At this point, both keys are emitted and allow_additional_properties=False.
    # Emitting ',' must be rejected!
    assert val.feed_char(',') is False

    # Emitting '}' must be allowed and complete the object!
    assert val.feed_char('}') is True
    assert val.is_complete() is True
    assert json.loads(val.base_validator.buffer) == {"feature": "val", "enabled": True}


def test_escaped_characters_and_quotes_in_values():
    """Verifies that escaped strings and escaped characters in values are supported."""
    schema = JSONSchema.from_keys(required_keys=["msg", "count"], type_map={"msg": JSONType.STRING, "count": JSONType.INTEGER})
    val = SchemaConstrainedJSONValidator(schema)

    test_str = r'{"msg": "line1\nline2 \"quoted\" \\ slash", "count": 12}'
    for c in test_str:
        assert val.feed_char(c) is True
    assert val.is_complete() is True
    parsed = json.loads(val.base_validator.buffer)
    assert parsed["msg"] == 'line1\nline2 "quoted" \\ slash'
    assert parsed["count"] == 12


def test_strict_spacing_vs_general_whitespace():
    """Documents strict_spacing behavior: disallows newlines/tabs outside strings."""
    schema = JSONSchema.from_keys(required_keys=["a"])

    # In strict_spacing=True (default), newline outside string is rejected
    val_strict = SchemaConstrainedJSONValidator(schema, strict_spacing=True)
    assert val_strict.feed_char('{') is True
    assert val_strict.feed_char('\n') is False

    # In strict_spacing=False, newline outside string is permitted per standard RFC 8259
    val_loose = SchemaConstrainedJSONValidator(schema, strict_spacing=False)
    assert val_loose.feed_char('{') is True
    assert val_loose.feed_char('\n') is True
    assert val_loose.feed_char(' ') is True
    assert val_loose.feed_char('"') is True
    assert val_loose.feed_char('a') is True
    assert val_loose.feed_char('"') is True
    assert val_loose.feed_char(':') is True
    assert val_loose.feed_char('\n') is True
    assert val_loose.feed_char('1') is True
    assert val_loose.feed_char('}') is True
    assert val_loose.is_complete() is True
    assert json.loads(val_loose.base_validator.buffer) == {"a": 1}


def test_fake_model_avoids_trailing_comma_deadlock():
    """Scripted fake-model test verifying that when model predicts comma after final key,
    the decoder rejects comma and selects closing brace instead."""
    vocab_list = ["<pad>", "<eos>", "{", "}", "\"", ": ", ", ", "k1", "v1", "k2", "v2"]
    tok = FakeTokenizer(vocab_list)
    eos_id = 1

    # Model attempts to predict comma (id 6) after v2, but closing brace (id 3) is also scored
    script = {
        (0,): {2: 10.0},
        (0, 2): {4: 10.0},
        (0, 2, 4): {7: 10.0},  # 'k1'
        (0, 2, 4, 7): {4: 10.0},
        (0, 2, 4, 7, 4): {5: 10.0}, # ': '
        (0, 2, 4, 7, 4, 5): {4: 10.0},
        (0, 2, 4, 7, 4, 5, 4): {8: 10.0}, # 'v1'
        (0, 2, 4, 7, 4, 5, 4, 8): {4: 10.0},
        (0, 2, 4, 7, 4, 5, 4, 8, 4): {6: 10.0}, # ', '
        (0, 2, 4, 7, 4, 5, 4, 8, 4, 6): {4: 10.0},
        (0, 2, 4, 7, 4, 5, 4, 8, 4, 6, 4): {9: 10.0}, # 'k2'
        (0, 2, 4, 7, 4, 5, 4, 8, 4, 6, 4, 9): {4: 10.0},
        (0, 2, 4, 7, 4, 5, 4, 8, 4, 6, 4, 9, 4): {5: 10.0}, # ': '
        (0, 2, 4, 7, 4, 5, 4, 8, 4, 6, 4, 9, 4, 5): {4: 10.0},
        (0, 2, 4, 7, 4, 5, 4, 8, 4, 6, 4, 9, 4, 5, 4): {10: 10.0}, # 'v2'
        (0, 2, 4, 7, 4, 5, 4, 8, 4, 6, 4, 9, 4, 5, 4, 10): {4: 10.0}, # '"'
        # Crucial branch: model favors ', ' (id 6) with score 12, '}' (id 3) with score 8
        (0, 2, 4, 7, 4, 5, 4, 8, 4, 6, 4, 9, 4, 5, 4, 10, 4): {6: 12.0, 3: 8.0},
    }
    model = FakeModel(script, len(vocab_list))
    schema = JSONSchema.from_keys(required_keys=["k1", "k2"])

    out = generate_schema_constrained(model, tok, [0], schema, max_new_tokens=25, eos_id=eos_id)
    assert out["stop_reason"] == "COMPLETE_JSON"
    assert out["is_valid_json"] is True
    assert out["parsed_object"] == {"k1": "v1", "k2": "v2"}
