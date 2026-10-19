"""Deterministic tests and fake-model harness for Schema Prefix Trie and Decoder."""

import json
from pathlib import Path
import pytest
import torch

from experiments.batch3111_3150.schema_prefix_trie import (
    JSONSchema,
    JSONType,
    PrefixTrie,
    SchemaConstrainedJSONValidator,
    generate_schema_constrained,
)


class FakeTokenizer:
    """Deterministic miniature tokenizer mapping IDs to strings."""
    def __init__(self, vocab_list):
        self.id_to_token = {i: tok for i, tok in enumerate(vocab_list)}
        self.token_to_id = {tok: i for i, tok in enumerate(vocab_list)}
        self.vocab_size = len(vocab_list)

    def encode(self, text: str):
        # simple character/token lookup
        res = []
        i = 0
        while i < len(text):
            matched = False
            for length in range(min(10, len(text) - i), 0, -1):
                sub = text[i:i+length]
                if sub in self.token_to_id:
                    res.append(self.token_to_id[sub])
                    i += length
                    matched = True
                    break
            if not matched:
                raise ValueError(f"Unknown char: {text[i]}")
        return res

    def decode(self, token_ids, errors="replace"):
        return "".join(self.id_to_token[t] for t in token_ids)


class FakeModel:
    """Deterministic fake model that returns scripted top logits."""
    def __init__(self, script_log_dict, vocab_size):
        self.script = script_log_dict
        self.vocab_size = vocab_size

    def __call__(self, x):
        tokens = tuple(x[0].tolist())
        logits = torch.zeros((1, 1, self.vocab_size))
        # Default flat logits
        if tokens in self.script:
            preferred = self.script[tokens]
            for tid, val in preferred.items():
                logits[0, 0, tid] = val
        return logits


def test_fake_model_schema_constrained_generation():
    vocab_list = [
        "<pad>", "<eos>", "{", "}", "\"", ": ", ", ", "status", "code", "queued", "202", " ", "\n", "__main__"
    ]
    tok = FakeTokenizer(vocab_list)
    eos_id = 1
    # Build a scripted scenario where:
    # prompt is [0]
    # At [0], model wants '{' (id 2)
    # At [0, 2], model strongly prefers '__main__' (id 13) with logit 10, '"' (id 4) with logit 5
    # Schema should force '"'
    # Then 'status' -> ': ' -> '"' -> 'queued' -> '"' -> ', ' -> '"' -> 'code' -> ': ' -> '202' -> '}'
    script = {
        (0,): {2: 10.0, 13: 8.0},
        (0, 2): {13: 15.0, 4: 8.0},  # '__main__' is favored, but schema must block it!
        (0, 2, 4): {13: 15.0, 7: 10.0, 8: 6.0}, # favors 'status' (id 7) over 'code' (id 8)
        (0, 2, 4, 7): {4: 10.0},
        (0, 2, 4, 7, 4): {5: 10.0}, # ': '
        (0, 2, 4, 7, 4, 5): {4: 10.0}, # '"'
        (0, 2, 4, 7, 4, 5, 4): {9: 10.0}, # 'queued'
        (0, 2, 4, 7, 4, 5, 4, 9): {4: 10.0}, # '"'
        (0, 2, 4, 7, 4, 5, 4, 9, 4): {6: 10.0}, # ', '
        (0, 2, 4, 7, 4, 5, 4, 9, 4, 6): {4: 10.0}, # '"'
        (0, 2, 4, 7, 4, 5, 4, 9, 4, 6, 4): {8: 10.0}, # 'code'
        (0, 2, 4, 7, 4, 5, 4, 9, 4, 6, 4, 8): {4: 10.0}, # '"'
        (0, 2, 4, 7, 4, 5, 4, 9, 4, 6, 4, 8, 4): {5: 10.0}, # ': '
        (0, 2, 4, 7, 4, 5, 4, 9, 4, 6, 4, 8, 4, 5): {10: 10.0}, # '202'
        (0, 2, 4, 7, 4, 5, 4, 9, 4, 6, 4, 8, 4, 5, 10): {3: 10.0}, # '}'
    }
    model = FakeModel(script, vocab_size=len(vocab_list))
    schema = JSONSchema.from_keys(
        required_keys=["status", "code"],
        type_map={"status": JSONType.STRING, "code": JSONType.INTEGER}
    )

    out = generate_schema_constrained(
        model=model,
        tokenizer=tok,
        prompt_tokens=[0],
        schema=schema,
        max_new_tokens=30,
        eos_id=eos_id,
    )

    assert out["stop_reason"] == "COMPLETE_JSON"
    assert out["is_valid_json"] is True
    assert out["schema_match"] is True
    assert out["parsed_object"] == {"status": "queued", "code": 202}
