"""Schema Representation, Prefix Trie, and Schema-Constrained JSON Validator.

This module provides:
1. JSONSchema: An explicit schema specification defining required/optional properties,
   allowed value types, and object completion constraints.
2. PrefixTrie: A Trie data structure over allowed strings supporting prefix matching,
   exact match detection, and valid continuation queries.
3. SchemaConstrainedJSONValidator: An incremental validator combining the JSON grammar PDA
   with schema-level property name and type filtering.
4. generate_schema_constrained: A constrained greedy generation loop that enforces schema
   property names and types without fabricating values or masking native model failures.

METHODOLOGICAL INTEGRITY:
- Schema constraints enforce that keys emitted are valid members of the schema trie.
- Values are generated solely by model logits from candidate tokens allowed by the schema type.
- Values are NEVER injected, repaired, or fabricated on behalf of the model.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import torch

from experiments.batch3111_3150.json_constrained_decoder import (
    IncrementalJSONValidator,
    JSONState,
)


class JSONType(Enum):
    STRING = "string"
    INTEGER = "integer"
    NUMBER = "number"
    BOOLEAN = "boolean"
    NULL = "null"
    OBJECT = "object"
    ARRAY = "array"
    ANY = "any"


@dataclass
class PropertySpec:
    """Specification for an object property in a schema."""
    name: str
    allowed_types: Set[JSONType] = field(default_factory=lambda: {JSONType.ANY})
    required: bool = True
    nested_schema: Optional[JSONSchema] = None

    def accepts_type(self, val_type: JSONType) -> bool:
        if JSONType.ANY in self.allowed_types:
            return True
        if val_type in self.allowed_types:
            return True
        if val_type == JSONType.INTEGER and JSONType.NUMBER in self.allowed_types:
            return True
        return False


@dataclass
class JSONSchema:
    """Schema definition for JSON object validation."""
    properties: Dict[str, PropertySpec] = field(default_factory=dict)
    allow_additional_properties: bool = False
    allow_duplicate_keys: bool = False

    @property
    def required_keys(self) -> Set[str]:
        return {p.name for p in self.properties.values() if p.required}

    @property
    def all_keys(self) -> Set[str]:
        return set(self.properties.keys())

    @classmethod
    def from_keys(
        cls,
        required_keys: List[str],
        optional_keys: Optional[List[str]] = None,
        type_map: Optional[Dict[str, Union[JSONType, Set[JSONType]]]] = None,
        allow_additional_properties: bool = False,
    ) -> JSONSchema:
        """Helper to build schema from list of keys and optional type definitions."""
        props: Dict[str, PropertySpec] = {}
        for k in required_keys:
            t = type_map.get(k, {JSONType.ANY}) if type_map else {JSONType.ANY}
            types = {t} if isinstance(t, JSONType) else set(t)
            props[k] = PropertySpec(name=k, allowed_types=types, required=True)
        if optional_keys:
            for k in optional_keys:
                t = type_map.get(k, {JSONType.ANY}) if type_map else {JSONType.ANY}
                types = {t} if isinstance(t, JSONType) else set(t)
                props[k] = PropertySpec(name=k, allowed_types=types, required=False)
        return cls(
            properties=props,
            allow_additional_properties=allow_additional_properties,
            allow_duplicate_keys=False,
        )


class TrieNode:
    def __init__(self):
        self.children: Dict[str, TrieNode] = {}
        self.is_terminal: bool = False
        self.value: Optional[str] = None


class PrefixTrie:
    """Prefix Trie for allowed string tokens/properties."""

    def __init__(self, words: Optional[List[str]] = None):
        self.root = TrieNode()
        self.words: Set[str] = set()
        if words:
            for w in words:
                self.insert(w)

    def insert(self, word: str) -> None:
        self.words.add(word)
        node = self.root
        for ch in word:
            if ch not in node.children:
                node.children[ch] = TrieNode()
            node = node.children[ch]
        node.is_terminal = True
        node.value = word

    def is_valid_prefix(self, prefix: str) -> bool:
        """Returns True if prefix matches any active path in the trie."""
        node = self.root
        for ch in prefix:
            if ch not in node.children:
                return False
            node = node.children[ch]
        return True

    def is_exact_match(self, word: str) -> bool:
        node = self.root
        for ch in word:
            if ch not in node.children:
                return False
            node = node.children[ch]
        return node.is_terminal

    def get_allowed_continuations(self, prefix: str) -> Set[str]:
        """Returns the set of next single characters allowed after prefix."""
        node = self.root
        for ch in prefix:
            if ch not in node.children:
                return set()
            node = node.children[ch]
        return set(node.children.keys())


class SchemaConstrainedJSONValidator:
    """Incremental JSON validator combining PDA grammar checks with schema prefix trie constraints."""

    def __init__(self, schema: JSONSchema, max_depth: int = 16, strict_spacing: bool = True):
        self.schema = schema
        self.max_depth = max_depth
        self.strict_spacing = strict_spacing
        self.base_validator = IncrementalJSONValidator(max_depth=max_depth)

        # Schema tracking state
        self.emitted_keys: Set[str] = set()
        self.current_key_buffer: str = ""
        self.current_active_key: Optional[str] = None
        self.current_val_type: Optional[JSONType] = None
        self.is_in_key: bool = False

        # Build initial prefix trie
        self._rebuild_key_trie()

    def _rebuild_key_trie(self) -> None:
        """Builds a trie of keys that are currently permissible."""
        if self.schema.allow_duplicate_keys:
            candidate_keys = list(self.schema.all_keys)
        else:
            candidate_keys = [k for k in self.schema.all_keys if k not in self.emitted_keys]
        self.active_trie = PrefixTrie(candidate_keys)

    def is_valid_string_extension(self, s: str) -> bool:
        """Simulates feeding string s to a cloned validator state."""
        sim = copy.deepcopy(self)
        for ch in s:
            if not sim.feed_char(ch):
                return False
        return True

    def feed_char(self, ch: str) -> bool:
        prev_state = self.base_validator.state
        prev_in_string = self.base_validator.in_string
        prev_stack_len = len(self.base_validator.stack)

        # 1. Check if object close '}' is attempted
        if ch == '}' and not self.base_validator.in_string:
            # If closing root object, verify all required keys are present
            if len(self.base_validator.stack) == 1 and self.base_validator.stack[-1] == '{':
                missing_required = self.schema.required_keys - self.emitted_keys
                if missing_required:
                    # Closing not permitted if required keys are missing!
                    return False

        # 1b. Check if comma ',' is attempted when no further keys are allowed
        if ch == ',' and not self.base_validator.in_string:
            if len(self.base_validator.stack) == 1 and self.base_validator.stack[-1] == '{':
                if not self.schema.allow_additional_properties:
                    remaining_keys = self.schema.all_keys - self.emitted_keys
                    if not remaining_keys and not self.schema.allow_duplicate_keys:
                        # Cannot emit comma if no more keys are allowed in this object!
                        return False

        # 2. Key context detection: are we entering or inside a key string?
        # A key begins when state is IN_OBJECT_EXPECT_KEY_OR_CLOSE or IN_OBJECT_EXPECT_KEY and ch == '"'
        entering_key = False
        if not prev_in_string and ch == '"':
            if prev_state in (JSONState.IN_OBJECT_EXPECT_KEY_OR_CLOSE, JSONState.IN_OBJECT_EXPECT_KEY):
                entering_key = True

        # Disallow unproductive whitespace loops when strict_spacing is enabled
        # In strict_spacing mode: disallow newlines/tabs outside strings which cause doctest indentation traps
        if self.strict_spacing and not prev_in_string:
            if ch in '\n\r\t':
                return False

        # Check grammar acceptance first
        if not self.base_validator.feed_char(ch):
            return False

        new_state = self.base_validator.state
        new_in_string = self.base_validator.in_string

        # State updates for key tracking
        if entering_key:
            self.is_in_key = True
            self.current_key_buffer = ""
            return True

        if self.is_in_key:
            if prev_in_string and not new_in_string and ch == '"':
                # Key string finished!
                finished_key = self.current_key_buffer
                # Verify exact key is in schema
                if not self.schema.allow_additional_properties:
                    if not self.active_trie.is_exact_match(finished_key):
                        return False
                if not self.schema.allow_duplicate_keys and finished_key in self.emitted_keys:
                    return False

                self.emitted_keys.add(finished_key)
                self.current_active_key = finished_key
                self.is_in_key = False
                self.current_key_buffer = ""
                self._rebuild_key_trie()
                return True
            else:
                # Character inside key
                self.current_key_buffer += ch
                if not self.schema.allow_additional_properties:
                    if not self.active_trie.is_valid_prefix(self.current_key_buffer):
                        return False
                return True

        # 3. Value type constraints (if current_active_key has type restrictions)
        if self.current_active_key and self.current_active_key in self.schema.properties:
            prop = self.schema.properties[self.current_active_key]
            # If entering a value (from outside a string):
            if prev_state == JSONState.IN_OBJECT_EXPECT_VALUE and not prev_in_string:
                # Check what value type is starting
                if ch == '"':
                    if not prop.accepts_type(JSONType.STRING): return False
                elif ch in '-0123456789':
                    if not (prop.accepts_type(JSONType.INTEGER) or prop.accepts_type(JSONType.NUMBER)): return False
                elif ch == 't' or ch == 'f':
                    if not prop.accepts_type(JSONType.BOOLEAN): return False
                elif ch == 'n':
                    if not prop.accepts_type(JSONType.NULL): return False
                elif ch == '{':
                    if not prop.accepts_type(JSONType.OBJECT): return False
                elif ch == '[':
                    if not prop.accepts_type(JSONType.ARRAY): return False

        # Reset current_active_key once comma is reached
        if new_state == JSONState.IN_OBJECT_EXPECT_KEY:
            self.current_active_key = None

        return True

    def is_complete(self) -> bool:
        if not self.base_validator.is_complete():
            return False
        # Verify all required keys are present
        missing = self.schema.required_keys - self.emitted_keys
        return len(missing) == 0


def generate_schema_constrained(
    model: Any,
    tokenizer: Any,
    prompt_tokens: List[int],
    schema: JSONSchema,
    max_new_tokens: int = 48,
    eos_id: int = 1,
    repetition_penalty: float = 1.0,
    strict_spacing: bool = True,
) -> Dict[str, Any]:
    """Greedy generation with grammar and schema-prefix trie logit filtering.
    
    This function:
    - Never modifies model weights.
    - Never fabricates or injects field values.
    - Dynamically queries SchemaConstrainedJSONValidator for candidate validity.
    - Returns transparent diagnostics: stop_reason, token_count, parsed_object.
    """
    validator = SchemaConstrainedJSONValidator(schema=schema, strict_spacing=strict_spacing)
    gen_tokens: List[int] = []
    curr = list(prompt_tokens)
    stop_reason = "MAX_TOKENS"

    vocab_size = tokenizer.vocab_size
    decoded_vocab = {}
    for tid in range(vocab_size):
        try:
            decoded_vocab[tid] = tokenizer.decode([tid], errors="replace")
        except Exception:
            decoded_vocab[tid] = tokenizer.decode([tid])

    for step_idx in range(max_new_tokens):
        if len(curr) >= 512:
            stop_reason = "CONTEXT_LIMIT"
            break

        x = torch.tensor([curr], dtype=torch.long)
        with torch.no_grad():
            logits = model(x)[0, -1, :].clone()

        # If validator is complete, EOS is allowed and preferred
        if validator.is_complete():
            stop_reason = "COMPLETE_JSON"
            gen_tokens.append(eos_id)
            break

        # Apply repetition penalty if configured
        if repetition_penalty > 1.0:
            for prev_tok in set(gen_tokens):
                if logits[prev_tok] > 0:
                    logits[prev_tok] /= repetition_penalty
                else:
                    logits[prev_tok] *= repetition_penalty

        # Find allowed tokens: test top candidates
        topk_candidates = torch.topk(logits, min(128, vocab_size)).indices.tolist()

        chosen_token = None
        for cand_tok in topk_candidates:
            if cand_tok == eos_id:
                if validator.is_complete():
                    chosen_token = cand_tok
                    break
                else:
                    continue  # Incomplete JSON cannot emit EOS

            cand_str = decoded_vocab[cand_tok]
            # Empty string tokens (like pad/unk) cannot make progress
            if not cand_str:
                continue
            if validator.is_valid_string_extension(cand_str):
                chosen_token = cand_tok
                break

        # Fallback to scan full vocabulary if none in top-128 is valid
        if chosen_token is None:
            sorted_all = torch.argsort(logits, descending=True).tolist()
            for cand_tok in sorted_all:
                if cand_tok == eos_id and validator.is_complete():
                    chosen_token = cand_tok
                    break
                cand_str = decoded_vocab[cand_tok]
                if not cand_str:
                    continue
                if validator.is_valid_string_extension(cand_str):
                    chosen_token = cand_tok
                    break

        if chosen_token is None:
            stop_reason = "NO_VALID_TOKENS"
            break

        if chosen_token == eos_id:
            stop_reason = "STOP_TOKEN"
            break

        tok_str = decoded_vocab[chosen_token]
        for ch in tok_str:
            validator.feed_char(ch)

        gen_tokens.append(chosen_token)
        curr.append(chosen_token)

        # If feeding this token completed the JSON object, mark complete and stop cleanly
        if validator.is_complete():
            stop_reason = "COMPLETE_JSON"
            break

    try:
        out_text = tokenizer.decode(gen_tokens, errors="replace")
    except Exception:
        out_text = tokenizer.decode(gen_tokens)

    # Clean EOS token representation if decoded
    for eos_repr in ["<|endoftext|>", "<eos>"]:
        if out_text.endswith(eos_repr):
            out_text = out_text[:-len(eos_repr)].strip()
    if gen_tokens and gen_tokens[-1] == eos_id:
        # Also decode without final EOS token for safety
        try:
            out_text = tokenizer.decode(gen_tokens[:-1], errors="replace").strip()
        except Exception:
            out_text = tokenizer.decode(gen_tokens[:-1]).strip()

    is_valid_json = False
    parsed_obj = None
    missing_keys = list(schema.required_keys - validator.emitted_keys)
    extra_keys = list(validator.emitted_keys - schema.all_keys)

    try:
        parsed_obj = json.loads(out_text.strip())
        is_valid_json = isinstance(parsed_obj, dict)
    except Exception:
        is_valid_json = False

    schema_match = False
    if is_valid_json and isinstance(parsed_obj, dict):
        obj_keys = set(parsed_obj.keys())
        schema_match = (schema.required_keys.issubset(obj_keys) and 
                        (schema.allow_additional_properties or obj_keys.issubset(schema.all_keys)))

    return {
        "generated_tokens": gen_tokens,
        "token_count": len(gen_tokens),
        "text": out_text,
        "stop_reason": stop_reason,
        "is_valid_json": is_valid_json,
        "schema_match": schema_match,
        "parsed_object": parsed_obj,
        "emitted_keys": list(validator.emitted_keys),
        "missing_keys": missing_keys,
        "extra_keys": extra_keys,
    }
