"""Experimental Tokenizer-Aware JSON Constrained Decoder for ChakrView.

This module provides an isolated, training-free grammar-constrained decoding engine.
It uses an incremental JSON pushdown automaton (PDA) to dynamically filter vocabulary
logits to only tokens whose byte-level extensions preserve valid JSON prefixes.

IMPORTANT METHODOLOGICAL BOUNDARY:
- Constrained decoding is a runtime-enforced decoding mechanism, NOT native model capability.
- Native unconstrained greedy decoding remains available and unmodified in run_evaluation.py.
- The grammar state machine never invents or alters field values; it only suppresses syntactically
  invalid vocabulary transitions.
"""

from __future__ import annotations

import json
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Tuple

import torch

class JSONState(Enum):
    EXPECT_VALUE = auto()
    IN_OBJECT_EXPECT_KEY_OR_CLOSE = auto()
    IN_OBJECT_EXPECT_KEY = auto()
    IN_OBJECT_EXPECT_COLON = auto()
    IN_OBJECT_EXPECT_VALUE = auto()
    IN_OBJECT_EXPECT_COMMA_OR_CLOSE = auto()
    IN_ARRAY_EXPECT_VALUE_OR_CLOSE = auto()
    IN_ARRAY_EXPECT_VALUE = auto()
    IN_ARRAY_EXPECT_COMMA_OR_CLOSE = auto()
    IN_STRING = auto()
    IN_STRING_ESCAPE = auto()
    IN_NUMBER = auto()
    IN_TRUE = auto()
    IN_FALSE = auto()
    IN_NULL = auto()
    COMPLETE = auto()
    ERROR = auto()


class IncrementalJSONValidator:
    """Incremental character-by-character JSON syntax validator tracking PDA states."""

    def __init__(self, max_depth: int = 16):
        self.max_depth = max_depth
        self.stack: List[str] = []  # Stack of container types: '{' or '['
        self.state = JSONState.EXPECT_VALUE
        self.in_string = False
        self.escape = False
        self.literal_expected: str = ""
        self.literal_index: int = 0
        self.buffer = ""

    def feed_char(self, ch: str) -> bool:
        """Feeds a single character and updates state. Returns False if character violates JSON grammar."""
        self.buffer += ch

        # 1. String content handling
        if self.in_string:
            if self.escape:
                # Valid escape characters in JSON
                if ch in ['"', '\\', '/', 'b', 'f', 'n', 'r', 't', 'u']:
                    self.escape = False
                    return True
                return False
            if ch == '\\':
                self.escape = True
                return True
            if ch == '"':
                self.in_string = False
                # What context were we in?
                if not self.stack:
                    self.state = JSONState.COMPLETE
                    return True
                top = self.stack[-1]
                if top == '{':
                    # Was this an object key or a value?
                    if self.state in [JSONState.IN_OBJECT_EXPECT_KEY_OR_CLOSE, JSONState.IN_OBJECT_EXPECT_KEY]:
                        self.state = JSONState.IN_OBJECT_EXPECT_COLON
                        return True
                    elif self.state in [JSONState.IN_OBJECT_EXPECT_VALUE]:
                        self.state = JSONState.IN_OBJECT_EXPECT_COMMA_OR_CLOSE
                        return True
                elif top == '[':
                    self.state = JSONState.IN_ARRAY_EXPECT_COMMA_OR_CLOSE
                    return True
                return False
            # Regular string character
            return True

        # 2. Literal constant matching (true, false, null)
        if self.literal_expected:
            if ch == self.literal_expected[self.literal_index]:
                self.literal_index += 1
                if self.literal_index == len(self.literal_expected):
                    self.literal_expected = ""
                    self.literal_index = 0
                    if not self.stack:
                        self.state = JSONState.COMPLETE
                    elif self.stack[-1] == '{':
                        self.state = JSONState.IN_OBJECT_EXPECT_COMMA_OR_CLOSE
                    elif self.stack[-1] == '[':
                        self.state = JSONState.IN_ARRAY_EXPECT_COMMA_OR_CLOSE
                return True
            return False

        # 3. Number handling
        if self.state == JSONState.IN_NUMBER:
            if ch in '0123456789.eE+-':
                return True
            else:
                # Transition out of number upon seeing delimiter or whitespace
                self.state = JSONState.IN_OBJECT_EXPECT_COMMA_OR_CLOSE if (self.stack and self.stack[-1] == '{') else (
                    JSONState.IN_ARRAY_EXPECT_COMMA_OR_CLOSE if (self.stack and self.stack[-1] == '[') else JSONState.COMPLETE
                )
                # Fall through to process delimiter

        # 4. Whitespace is ignored outside strings (unless already complete)
        if ch in ' \t\n\r':
            if self.state == JSONState.COMPLETE:
                return False
            return True

        # 5. Top-level or container transitions
        if self.state == JSONState.EXPECT_VALUE:
            if ch == '{':
                if len(self.stack) >= self.max_depth: return False
                self.stack.append('{')
                self.state = JSONState.IN_OBJECT_EXPECT_KEY_OR_CLOSE
                return True
            elif ch == '[':
                if len(self.stack) >= self.max_depth: return False
                self.stack.append('[')
                self.state = JSONState.IN_ARRAY_EXPECT_VALUE_OR_CLOSE
                return True
            elif ch == '"':
                self.in_string = True
                return True
            elif ch in '-0123456789':
                self.state = JSONState.IN_NUMBER
                return True
            elif ch == 't':
                self.literal_expected = 'true'
                self.literal_index = 1
                return True
            elif ch == 'f':
                self.literal_expected = 'false'
                self.literal_index = 1
                return True
            elif ch == 'n':
                self.literal_expected = 'null'
                self.literal_index = 1
                return True
            return False

        if self.state == JSONState.IN_OBJECT_EXPECT_KEY_OR_CLOSE:
            if ch == '}':
                self.stack.pop()
                self.state = JSONState.COMPLETE if not self.stack else (
                    JSONState.IN_OBJECT_EXPECT_COMMA_OR_CLOSE if self.stack[-1] == '{' else JSONState.IN_ARRAY_EXPECT_COMMA_OR_CLOSE
                )
                return True
            elif ch == '"':
                self.in_string = True
                return True
            return False

        if self.state == JSONState.IN_OBJECT_EXPECT_KEY:
            if ch == '"':
                self.in_string = True
                return True
            return False

        if self.state == JSONState.IN_OBJECT_EXPECT_COLON:
            if ch == ':':
                self.state = JSONState.IN_OBJECT_EXPECT_VALUE
                return True
            return False

        if self.state == JSONState.IN_OBJECT_EXPECT_VALUE:
            if ch == '{':
                if len(self.stack) >= self.max_depth: return False
                self.stack.append('{')
                self.state = JSONState.IN_OBJECT_EXPECT_KEY_OR_CLOSE
                return True
            elif ch == '[':
                if len(self.stack) >= self.max_depth: return False
                self.stack.append('[')
                self.state = JSONState.IN_ARRAY_EXPECT_VALUE_OR_CLOSE
                return True
            elif ch == '"':
                self.in_string = True
                return True
            elif ch in '-0123456789':
                self.state = JSONState.IN_NUMBER
                return True
            elif ch == 't':
                self.literal_expected = 'true'
                self.literal_index = 1
                return True
            elif ch == 'f':
                self.literal_expected = 'false'
                self.literal_index = 1
                return True
            elif ch == 'n':
                self.literal_expected = 'null'
                self.literal_index = 1
                return True
            return False

        if self.state == JSONState.IN_OBJECT_EXPECT_COMMA_OR_CLOSE:
            if ch == ',':
                self.state = JSONState.IN_OBJECT_EXPECT_KEY
                return True
            elif ch == '}':
                self.stack.pop()
                self.state = JSONState.COMPLETE if not self.stack else (
                    JSONState.IN_OBJECT_EXPECT_COMMA_OR_CLOSE if self.stack[-1] == '{' else JSONState.IN_ARRAY_EXPECT_COMMA_OR_CLOSE
                )
                return True
            return False

        if self.state == JSONState.IN_ARRAY_EXPECT_VALUE_OR_CLOSE:
            if ch == ']':
                self.stack.pop()
                self.state = JSONState.COMPLETE if not self.stack else (
                    JSONState.IN_OBJECT_EXPECT_COMMA_OR_CLOSE if self.stack[-1] == '{' else JSONState.IN_ARRAY_EXPECT_COMMA_OR_CLOSE
                )
                return True
            elif ch == '{':
                if len(self.stack) >= self.max_depth: return False
                self.stack.append('{')
                self.state = JSONState.IN_OBJECT_EXPECT_KEY_OR_CLOSE
                return True
            elif ch == '[':
                if len(self.stack) >= self.max_depth: return False
                self.stack.append('[')
                self.state = JSONState.IN_ARRAY_EXPECT_VALUE_OR_CLOSE
                return True
            elif ch == '"':
                self.in_string = True
                return True
            elif ch in '-0123456789':
                self.state = JSONState.IN_NUMBER
                return True
            elif ch == 't':
                self.literal_expected = 'true'
                self.literal_index = 1
                return True
            elif ch == 'f':
                self.literal_expected = 'false'
                self.literal_index = 1
                return True
            elif ch == 'n':
                self.literal_expected = 'null'
                self.literal_index = 1
                return True
            return False

        if self.state == JSONState.IN_ARRAY_EXPECT_VALUE:
            if ch == '{':
                if len(self.stack) >= self.max_depth: return False
                self.stack.append('{')
                self.state = JSONState.IN_OBJECT_EXPECT_KEY_OR_CLOSE
                return True
            elif ch == '[':
                if len(self.stack) >= self.max_depth: return False
                self.stack.append('[')
                self.state = JSONState.IN_ARRAY_EXPECT_VALUE_OR_CLOSE
                return True
            elif ch == '"':
                self.in_string = True
                return True
            elif ch in '-0123456789':
                self.state = JSONState.IN_NUMBER
                return True
            elif ch == 't':
                self.literal_expected = 'true'
                self.literal_index = 1
                return True
            elif ch == 'f':
                self.literal_expected = 'false'
                self.literal_index = 1
                return True
            elif ch == 'n':
                self.literal_expected = 'null'
                self.literal_index = 1
                return True
            return False

        if self.state == JSONState.IN_ARRAY_EXPECT_COMMA_OR_CLOSE:
            if ch == ',':
                self.state = JSONState.IN_ARRAY_EXPECT_VALUE
                return True
            elif ch == ']':
                self.stack.pop()
                self.state = JSONState.COMPLETE if not self.stack else (
                    JSONState.IN_OBJECT_EXPECT_COMMA_OR_CLOSE if self.stack[-1] == '{' else JSONState.IN_ARRAY_EXPECT_COMMA_OR_CLOSE
                )
                return True
            return False

        if self.state == JSONState.COMPLETE:
            return False

        return False

    def is_valid_string_extension(self, s: str) -> bool:
        """Simulates feeding string s to a cloned validator state."""
        import copy
        sim = copy.deepcopy(self)
        for ch in s:
            if not sim.feed_char(ch):
                return False
        return True

    def is_complete(self) -> bool:
        return self.state == JSONState.COMPLETE and len(self.stack) == 0 and not self.in_string and not self.literal_expected


def generate_json_constrained(
    model: Any,
    tokenizer: Any,
    prompt_tokens: List[int],
    max_new_tokens: int = 48,
    eos_id: int = 1,
    repetition_penalty: float = 1.0,
) -> Dict[str, Any]:
    """Greedy generation with grammar-constrained logit masking."""
    validator = IncrementalJSONValidator()
    gen_tokens: List[int] = []
    curr = list(prompt_tokens)
    stop_reason = "MAX_TOKENS"

    # Pre-cache decoded string representation of all tokens in vocabulary
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

        # If validator is complete, EOS is allowed (and preferred)
        if validator.is_complete():
            stop_reason = "COMPLETE_JSON"
            gen_tokens.append(eos_id)
            break

        # Apply repetition penalty to already generated tokens if configured
        if repetition_penalty > 1.0:
            for prev_tok in set(gen_tokens):
                if logits[prev_tok] > 0:
                    logits[prev_tok] /= repetition_penalty
                else:
                    logits[prev_tok] *= repetition_penalty

        # Find allowed tokens: test top candidates
        # To remain fast on CPU, test candidate tokens sorted by unconstrained logits
        topk_candidates = torch.topk(logits, min(128, vocab_size)).indices.tolist()

        chosen_token = None
        for cand_tok in topk_candidates:
            if cand_tok == eos_id:
                if validator.is_complete():
                    chosen_token = cand_tok
                    break
                else:
                    continue  # EOS disallowed when JSON is incomplete

            cand_str = decoded_vocab[cand_tok]
            if validator.is_valid_string_extension(cand_str):
                chosen_token = cand_tok
                break

        # Fallback: if none in top-128 is valid, scan entire vocabulary
        if chosen_token is None:
            sorted_all = torch.argsort(logits, descending=True).tolist()
            for cand_tok in sorted_all:
                if cand_tok == eos_id and validator.is_complete():
                    chosen_token = cand_tok
                    break
                cand_str = decoded_vocab[cand_tok]
                if validator.is_valid_string_extension(cand_str):
                    chosen_token = cand_tok
                    break

        if chosen_token is None:
            stop_reason = "GRAMMAR_DEADLOCK"
            break

        if chosen_token == eos_id:
            stop_reason = "STOP_TOKEN"
            break

        tok_str = decoded_vocab[chosen_token]
        # Feed characters to master validator
        for ch in tok_str:
            validator.feed_char(ch)

        gen_tokens.append(chosen_token)
        curr.append(chosen_token)

    try:
        out_text = tokenizer.decode(gen_tokens, errors="replace")
    except Exception:
        out_text = tokenizer.decode(gen_tokens)

    # Validate output with standard strict json parser
    is_valid_json = False
    parsed_obj = None
    try:
        parsed_obj = json.loads(out_text.strip())
        is_valid_json = isinstance(parsed_obj, (dict, list))
    except Exception:
        is_valid_json = False

    return {
        "generated_tokens": gen_tokens,
        "token_count": len(gen_tokens),
        "text": out_text,
        "stop_reason": stop_reason,
        "is_valid_json": is_valid_json,
        "parsed_object": parsed_obj,
    }
