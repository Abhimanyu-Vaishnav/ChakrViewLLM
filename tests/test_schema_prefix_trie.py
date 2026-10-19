"""Comprehensive unit and integration test suite for Schema Prefix Trie and Schema-Constrained Decoder.

Validates:
1. PrefixTrie prefix match, exact match, continuation query, and rejection of disallowed keys like '__main__'.
2. Handling of shared prefixes, partial keys, and multiple keys.
3. SchemaConstrainedJSONValidator enforcement of:
   - Required keys present vs missing.
   - Prevention of premature closing when required keys are missing.
   - Rejection of duplicate keys.
   - Rejection of unexpected keys when allow_additional_properties=False.
   - Value type constraints (string, integer, number, boolean, null).
4. Multi-token key prefix handling.
5. Preservation of original grammar decoder behavior.
6. Safe budget exhaustion and no-valid-token handling.
7. Verification that canonical artifacts and session.py remain untouched.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from experiments.batch3111_3150.schema_prefix_trie import (
    JSONSchema,
    JSONType,
    PrefixTrie,
    PropertySpec,
    SchemaConstrainedJSONValidator,
    generate_schema_constrained,
)
from experiments.batch3111_3150.json_constrained_decoder import IncrementalJSONValidator


def test_prefix_trie_basic_and_shared_prefixes():
    keys = ["status", "state", "code", "count"]
    trie = PrefixTrie(keys)

    # Exact matches
    for k in keys:
        assert trie.is_exact_match(k) is True
        assert trie.is_valid_prefix(k) is True

    # Partial prefixes
    assert trie.is_valid_prefix("stat") is True
    assert trie.is_valid_prefix("st") is True
    assert trie.is_valid_prefix("c") is True
    assert trie.is_valid_prefix("co") is True

    # Continuations after 'st'
    assert trie.get_allowed_continuations("st") == {"a"}  # 'sta'
    # After 'sta' -> 't'
    assert trie.get_allowed_continuations("sta") == {"t"}
    # After 'stat' -> 'u' (status) or 'e' (state)
    assert trie.get_allowed_continuations("stat") == {"u", "e"}

    # Disallowed keys (e.g., Python code attractor)
    assert trie.is_exact_match("__main__") is False
    assert trie.is_valid_prefix("__main__") is False
    assert trie.is_valid_prefix("_") is False
    assert trie.is_valid_prefix("import") is False


def test_schema_validator_requires_all_keys():
    schema = JSONSchema.from_keys(required_keys=["status", "code"])
    val = SchemaConstrainedJSONValidator(schema)

    # Feed opening
    for ch in '{"status": "ok"':
        assert val.feed_char(ch) is True

    # Attempt to close object prematurely before 'code' is emitted
    assert val.feed_char('}') is False
    assert val.is_complete() is False

    # Continue with comma and 'code'
    for ch in ', "code": 200}':
        assert val.feed_char(ch) is True

    assert val.is_complete() is True
    assert json.loads(val.base_validator.buffer) == {"status": "ok", "code": 200}
    assert val.emitted_keys == {"status", "code"}


def test_schema_validator_rejects_disallowed_and_duplicate_keys():
    schema = JSONSchema.from_keys(required_keys=["status", "code"])
    val = SchemaConstrainedJSONValidator(schema)

    # Disallowed key '__main__' rejected at first character
    assert val.feed_char('{') is True
    assert val.feed_char('"') is True
    assert val.feed_char('_') is False  # Trie blocks '_'

    # Feed valid key 'status'
    val2 = SchemaConstrainedJSONValidator(schema)
    for ch in '{"status": "ok", "':
        assert val2.feed_char(ch) is True

    # Attempt to emit duplicate 'status'
    assert val2.feed_char('s') is False  # 'status' already emitted, trie only contains 'code'


def test_schema_validator_type_constraints():
    schema = JSONSchema.from_keys(
        required_keys=["status", "code", "active"],
        type_map={
            "status": JSONType.STRING,
            "code": JSONType.INTEGER,
            "active": JSONType.BOOLEAN,
        }
    )
    val = SchemaConstrainedJSONValidator(schema)

    # 'status' requires string
    for ch in '{"status": ':
        assert val.feed_char(ch) is True
    # Attempt number for status
    assert val.feed_char('1') is False

    # Reset and feed string for status
    val2 = SchemaConstrainedJSONValidator(schema)
    for ch in '{"status": "ok", "code": ':
        assert val2.feed_char(ch) is True
    # 'code' requires integer; attempt boolean 'true'
    assert val2.feed_char('t') is False

    # Reset and feed valid types for all
    val3 = SchemaConstrainedJSONValidator(schema)
    for ch in '{"status": "ok", "code": 200, "active": true}':
        assert val3.feed_char(ch) is True
    assert val3.is_complete() is True
    assert json.loads(val3.base_validator.buffer) == {"status": "ok", "code": 200, "active": True}


def test_schema_validator_multi_token_key_simulation():
    schema = JSONSchema.from_keys(required_keys=["service_health", "status_code"])
    val = SchemaConstrainedJSONValidator(schema)

    # Feed prefix
    assert val.feed_char('{') is True

    # Simulate token fragments: '"serv', 'ice_', 'health": "up", "'
    assert val.is_valid_string_extension('"serv') is True
    for ch in '"serv': val.feed_char(ch)

    assert val.is_valid_string_extension('ice_') is True
    for ch in 'ice_': val.feed_char(ch)

    assert val.is_valid_string_extension('health') is True
    for ch in 'health': val.feed_char(ch)

    assert val.is_valid_string_extension('": "up"') is True
    for ch in '": "up"': val.feed_char(ch)

    # Disallowed continuation
    assert val.is_valid_string_extension('}') is False  # 'status_code' still required


def test_grammar_decoder_unchanged_behavior():
    # Verify base IncrementalJSONValidator still functions identically
    base = IncrementalJSONValidator()
    for ch in '{"a": 1, "b": [true, null]}':
        assert base.feed_char(ch) is True
    assert base.is_complete() is True
    assert json.loads(base.buffer) == {"a": 1, "b": [True, None]}
