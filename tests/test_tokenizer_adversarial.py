"""
ChakrView Tokenizer Tests: Explicit Adversarial & Stress Testing Suite (Phase I).

Verifies round-trip exactness against all Category I adversarial test cases:
- TC-ADV-01: Hindi Combining Marks
- TC-ADV-02: Sanskrit Conjuncts
- TC-ADV-03: ZWJ (U+200D) and ZWNJ (U+200C)
- TC-ADV-04: Complex Emojis & ZWJ sequences
- TC-ADV-05: Uncommon Unicode (Runes, Tibetan, Coptic)
- TC-ADV-06: Mixed Scripts Intertwined
- TC-ADV-07: Hindi + English + Numbers Intertwined
- TC-ADV-08: Code Indentation Stress
- TC-ADV-09: Tabs & Mixed Whitespace
- TC-ADV-10: CRLF and LF Alternation
- TC-ADV-11: Complex URLs & Query Strings
- TC-ADV-12: Windows Paths
- TC-ADV-13: Linux Paths
- TC-ADV-14: Mathematical Symbols
- TC-ADV-15: Malformed / Arbitrary Raw UTF-8 Byte Sequences
- Variation Selectors (U+FE0E, U+FE0F)

Rule: Do NOT normalize or modify the input merely to make the tests pass.
Reconstruction must be 100% bit-exact.
"""

import pytest

from chakrview.tokenizer.decoder import decode_bytes, decode_tokens
from chakrview.tokenizer.encoder import encode_bytes, encode_text
from chakrview.tokenizer.tokenizer import BPETokenizer


ADVERSARIAL_TEST_CASES = {
    "TC-ADV-01 (Hindi Combining Marks)": "कौशाम्बी, त्र्यंबकेश्वर, कुंडलियाँ, पूँछ, अँधेरा, ऋग्वेद, ॐ",
    "TC-ADV-02 (Sanskrit Conjuncts)": "सत्त्व, दग्ध, बुद्ध, उष्ट्र, स्त्रोत, कार्त्तिकेय, वाग्देवी",
    "TC-ADV-03 (ZWJ & ZWNJ)": "क्‍ + य = क्‍य (ZWJ: \u200d), क + ् + ‌ + य = क्‌य (ZWNJ: \u200c)",
    "TC-ADV-04 (Complex Emojis & ZWJ Sequences)": "Family: 👨‍👩‍👧‍👦, Flag: 🇮🇳, Modifiers: 👍🏽, 👩‍💻",
    "TC-ADV-05 (Uncommon Unicode)": "Runes: ᚠᚢᚦᚨᚱᚲ, Tibetan: ཨོཾ་མ་ཎི་པདྨེ་ཧཱུྃ, Coptic: ⲁⲃⲅⲇ",
    "TC-ADV-06 (Mixed Scripts Intertwined)": "ChakrView-चक्रव्यूह_core-v0.1",
    "TC-ADV-07 (Hindi + English + Numbers Intertwined)": "model-2026 mein ₹50000 ka 13th version, latency < 15.4ms",
    "TC-ADV-08 (Code Indentation Stress)": "    def f():\n        if True:\n            return 1",
    "TC-ADV-09 (Tabs & Mixed Whitespace)": "\tdef foo():\n\t    x = 10\n  \t  y = 20",
    "TC-ADV-10 (CRLF and LF Alternation)": "Line1\r\nLine2\nLine3\r\nLine4",
    "TC-ADV-11 (Complex URLs & Query Strings)": "https://api.chakrview.ai/v1/search?q=%E0%A4%AD%E0%A4%BE%E0%A4%B0%E0%A4%A4&limit=10&auth=true#top",
    "TC-ADV-12 (Windows Paths)": "d:\\Project\\ChakrView\\brain\\..\\data\\raw\\corpus.jsonl",
    "TC-ADV-13 (Linux Paths)": "/var/log/syslog.1.gz; rm -rf /tmp/test*",
    "TC-ADV-14 (Mathematical Symbols)": "∀x ∈ A: ∃y s.t. x ⊕ y = ∅ ∧ x ≢ y",
    "TC-ADV-VS (Variation Selectors)": "Heart without VS: \u2764, Heart with VS-16: \u2764\ufe0f, Text VS-15: \u2600\ufe0e",
}


@pytest.mark.parametrize("case_name, text", ADVERSARIAL_TEST_CASES.items())
def test_adversarial_round_trip(case_name: str, text: str):
    """
    Verify 100% exact round-trip preservation without normalization alteration.
    """
    tokens = encode_text(text)
    decoded = decode_tokens(tokens)

    assert decoded == text, f"Failed on adversarial test case: {case_name}"
    # Verify exact codepoints
    assert [ord(c) for c in decoded] == [ord(c) for c in text]


@pytest.mark.parametrize("case_name, text", ADVERSARIAL_TEST_CASES.items())
def test_adversarial_round_trip_with_bpe_merges(case_name: str, text: str):
    """
    Verify that learned BPE merges do not compromise exact adversarial reconstruction.
    """
    tokenizer = BPETokenizer()
    # Train toy merges on the test string itself + context
    tokenizer.train_toy(text + " " + text, max_merges=5, min_frequency=2)

    tokens = tokenizer.encode(text)
    decoded = tokenizer.decode(tokens)

    assert decoded == text, f"Failed with merges on adversarial test case: {case_name}"
    assert [ord(c) for c in decoded] == [ord(c) for c in text]


def test_adv_15_raw_invalid_utf8_byte_sequences():
    """
    TC-ADV-15: Test malformed, truncated, or illegal UTF-8 byte sequences.
    Must be faithfully processed by encode_bytes / decode_bytes without throwing exceptions.
    """
    malformed_byte_sequences = [
        b"\x80",  # Orphan continuation byte
        b"\xe0\xa4",  # Truncated 3-byte Devanagari prefix (missing third byte)
        b"\xf0\x9f\x98",  # Truncated 4-byte emoji prefix
        b"\xff\xfe\xfd",  # Completely illegal UTF-8 bytes
        b"\xc0\x80",  # Modified UTF-8 / overlong NUL
        b"\xed\xa0\x80",  # High surrogate in UTF-8
        b"Valid prefix " + b"\x80\xff" + b" valid suffix",
    ]

    for raw in malformed_byte_sequences:
        tokens = encode_bytes(raw)
        reconstructed = decode_bytes(tokens)
        assert reconstructed == raw, f"Failed raw byte round-trip on malformed sequence {raw!r}"
