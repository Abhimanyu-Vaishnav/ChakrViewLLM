"""
ChakrView Tokenizer Tests: Lossless UTF-8 Reconstruction (Phase C).

Verifies the primary architectural invariant:
  decode_tokens(encode_text(text)) == text
across all required linguistic, symbolic, code, and operating system test inputs.
"""

import pytest

from chakrview.tokenizer.decoder import decode_tokens
from chakrview.tokenizer.encoder import encode_text
from chakrview.tokenizer.tokenizer import BPETokenizer


REQUIRED_TEST_STRINGS = [
    # 1. Empty string
    "",
    # 2. English
    "Hello world",
    # 3. Hindi
    "नमस्ते दुनिया",
    # 4. Hinglish
    "mujhe coding seekhni hai",
    # 5. Mixed
    "mujhe Python mein एक function banana hai",
    # 6. Sanskrit
    "ॐ नमः शिवाय",
    # 7. Emoji
    "भारत 🇮🇳 ❤️ 🚀",
    # 8. Technical
    "RTX A1000 6GB",
    # 9. Mathematics
    "∀x ∈ A",
    # 10. URLs
    "https://example.com/a?q=नमस्ते",
    # 11. Windows path
    "D:\\Project\\ChakrView\\data",
    # 12. Linux path
    "/home/user/ChakrView/data",
    # 13. Code
    "def add(a, b):\n    return a + b",
    # 14. Whitespace, Tabs, LF, CRLF
    "Line1\r\nLine2\n\tIndented    Spaces",
    # 15. Numbers & Currency
    "123456 ₹50000 3.14159 -42 2026-09-26",
    # 16. Long multi-paragraph bilingual text
    (
        "ChakrView is an indigenous AI system designed for edge efficiency.\n"
        "चक्रव्यूह एक स्वदेशी एआई अनुसंधान परियोजना है जो कम संसाधनों वाले कंप्यूटरों पर चलती है।\n"
        "Here is an arithmetic function:\n"
        "def compute_entropy(p: float) -> float:\n"
        "    import math\n"
        "    return -p * math.log2(p) if p > 0.0 else 0.0\n"
        "Accuracy benchmark: 99.8% on RTX A1000.\n"
    ),
]


@pytest.mark.parametrize("text", REQUIRED_TEST_STRINGS)
def test_lossless_round_trip_functional(text: str):
    """
    Verify exact lossless round-trip using functional API:
    decode_tokens(encode_text(text)) == text
    """
    tokens = encode_text(text)
    reconstructed = decode_tokens(tokens)
    assert reconstructed == text, f"Mismatch on input: {text!r}"


@pytest.mark.parametrize("text", REQUIRED_TEST_STRINGS)
def test_lossless_round_trip_bpe_tokenizer_instance(text: str):
    """
    Verify exact lossless round-trip using BPETokenizer class instance.
    """
    tokenizer = BPETokenizer()
    tokens = tokenizer.encode(text)
    reconstructed = tokenizer.decode(tokens)
    assert reconstructed == text, f"Mismatch on input: {text!r}"


def test_type_safety():
    """Verify that non-string inputs raise TypeError."""
    with pytest.raises(TypeError):
        encode_text(12345)  # type: ignore
    with pytest.raises(TypeError):
        encode_text(None)  # type: ignore
    with pytest.raises(TypeError):
        decode_tokens("not a list")  # type: ignore
