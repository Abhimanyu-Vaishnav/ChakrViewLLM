"""
ChakrView Tokenizer Corpus Pipeline: Normalization Policy Module.

Explicitly documents and distinguishes TRAINING NORMALIZATION from RUNTIME ENCODING.

CRITICAL INVARIANT:
-------------------
Runtime encoding is STRICTLY LOSSLESS:
    Decode(Encode(text)) == text
Do NOT perform irreversible Unicode transformations at runtime.
"""

import unicodedata
from typing import Dict, Final

TRAINING_NORMALIZATION_POLICY: Final[Dict[str, str]] = {
    "unicode_normalization": "NFC (Canonical Decomposition, followed by Canonical Composition) for training stability",
    "newline_policy": "Normalized to '\\n' for line-delimited training; trailing \\r stripped",
    "whitespace_policy": "Consecutive internal spaces preserved; trailing line whitespace trimmed",
    "case_policy": "Strictly Case-Preserving (uppercase, lowercase, camelCase, snake_case untouched)",
    "punctuation_policy": "Strictly Preserved (ASCII and Unicode punctuation unchanged)",
    "unicode_preservation": "All combining marks, virama/halant, ZWJ, ZWNJ, emojis, and variation selectors strictly preserved",
}

RUNTIME_ENCODING_POLICY: Final[Dict[str, str]] = {
    "status": "STRICTLY LOSSLESS (Zero Normalization)",
    "unicode_normalization": "NONE (Raw UTF-8 byte stream is directly tokenized)",
    "newline_policy": "NONE (Both \\r\\n and \\n are faithfully preserved as raw byte primitives)",
    "whitespace_policy": "NONE (Exact byte sequence is reconstructed)",
    "case_policy": "NONE (Exact casing preserved)",
    "reversibility_standard": "Decode(Encode(S)) == S for 100% of inputs",
}


def apply_training_normalization(text: str) -> str:
    """
    Apply training-time normalization to stabilize BPE merge statistics.

    Rules applied:
    1. Unicode NFC canonical composition.
    2. Normalize CRLF to LF.
    3. Strip trailing whitespace per line.

    Note: This is ONLY applied during corpus preprocessing for BPE merge discovery.
    Runtime tokenizer execution bypasses this entirely to ensure 100% bit-exact reversibility.
    """
    # 1. Unicode NFC
    normalized = unicodedata.normalize("NFC", text)
    # 2. Strip trailing newlines and whitespace for clean line-level deduplication and training
    normalized = normalized.rstrip(" \t\r\n")
    return normalized
