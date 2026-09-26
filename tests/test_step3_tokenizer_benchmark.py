r"""
Unit and Integration Tests for Step 3: Empirical Tokenizer Benchmark & Validation.

Verifies:
1. 23-category control corpus completeness, metadata integrity, and hashes.
2. Numeric tokenization strategies (Candidate A, B, and C) on arithmetic test suite.
3. Indic and Unicode stress tests:
   - Devanagari Hindi (कौशाम्बी, त्र्यंबकेश्वर, कुंडलियाँ, पूँछ, अँधेरा, ऋग्वेद, ॐ)
   - Classical Sanskrit (सत्त्व, दग्ध, बुद्ध, उष्ट्र, कार्त्तिकेय, वाग्देवी)
   - ZWJ (U+200D) and ZWNJ (U+200C) ligature half-forms
   - Multi-codepoint composite emojis (👨👩👧👦, 👩💻, 👍🏽, 🇮🇳)
   - Combining characters, variation selectors, zero-width spaces, uncommon scripts
4. Raw arbitrary 8-bit octet sequences and invalid/truncated UTF-8 bytes:
   decode_bytes(encode_bytes(data)) == data for all cases.
5. Deterministic reproducibility across repeated tokenization runs.
6. Special token bounds [0, 2] and base byte primitives [3, 258].
"""

import json
from pathlib import Path
import pytest

from chakrview.tokenizer.bytes import BYTE_OFFSET, NUM_BYTE_TOKENS
from chakrview.tokenizer.special_tokens import BOS_ID, EOS_ID, PAD_ID
from chakrview.tokenizer.tokenizer import BPETokenizer
from experiments.tokenizer.candidates.numeric_adapters import NumericTokenizationAdapter
from experiments.tokenizer.fixtures.fixtures import (
    NUMERIC_BENCHMARK_ITEMS,
    UNICODE_STRESS_ITEMS,
    get_raw_byte_test_cases,
)


@pytest.fixture(scope="module")
def default_tokenizer() -> BPETokenizer:
    """Fixture providing a baseline BPETokenizer with fundamental bytes."""
    return BPETokenizer()


@pytest.fixture(scope="module")
def control_corpus_dir() -> Path:
    """Path to the 23-category control corpus."""
    return Path(__file__).resolve().parents[1] / "experiments" / "tokenizer" / "corpus"


def test_23_category_control_corpus_integrity(control_corpus_dir: Path):
    """Verify that all 23 required corpus files and metadata.json exist and are non-empty."""
    metadata_path = control_corpus_dir / "metadata.json"
    assert metadata_path.is_file(), "metadata.json missing in control corpus"

    with metadata_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    assert meta["total_categories"] == 23, f"Expected 23 categories, got {meta['total_categories']}"
    assert meta["total_bytes"] > 10000, "Corpus bytes suspiciously small"

    expected_files = [
        "english.txt",
        "hindi.txt",
        "sanskrit.txt",
        "hinglish.txt",
        "indian_names.txt",
        "technical.txt",
        "python_code.txt",
        "json_data.txt",
        "urls.txt",
        "windows_paths.txt",
        "linux_paths.txt",
        "mathematics.txt",
        "numbers.txt",
        "currencies.txt",
        "dates.txt",
        "scientific_notation.txt",
        "emojis.txt",
        "mixed_unicode.txt",
        "devanagari_conjuncts.txt",
        "zwj_zwnj.txt",
        "whitespace.txt",
        "line_endings.txt",
        "raw_bytes.txt",
    ]

    for fname in expected_files:
        fpath = control_corpus_dir / fname
        assert fpath.is_file(), f"Missing required category file: {fname}"
        content = fpath.read_text(encoding="utf-8")
        assert len(content.strip()) > 0, f"File {fname} is unexpectedly empty"


@pytest.mark.parametrize("strategy", ["A", "B", "C"])
def test_numeric_strategies_lossless(default_tokenizer: BPETokenizer, strategy: str):
    """Verify all 3 numeric strategies satisfy Decode(Encode(x)) == x on arithmetic benchmark."""
    adapter = NumericTokenizationAdapter(default_tokenizer, strategy=strategy)
    for item in NUMERIC_BENCHMARK_ITEMS:
        is_ok, tokens, decoded = adapter.round_trip_check(item)
        assert is_ok, f"Strategy {strategy} failed on {item!r}: got {decoded!r}"
        assert len(tokens) > 0, f"Empty tokens on {item!r}"


def test_numeric_expansion_monotonicity(default_tokenizer: BPETokenizer):
    """Verify sequence expansion relationship: tokens(A) >= tokens(B) >= tokens(C)."""
    adapter_a = NumericTokenizationAdapter(default_tokenizer, strategy="A")
    adapter_b = NumericTokenizationAdapter(default_tokenizer, strategy="B")
    adapter_c = NumericTokenizationAdapter(default_tokenizer, strategy="C")

    total_a = sum(len(adapter_a.encode(item)) for item in NUMERIC_BENCHMARK_ITEMS)
    total_b = sum(len(adapter_b.encode(item)) for item in NUMERIC_BENCHMARK_ITEMS)
    total_c = sum(len(adapter_c.encode(item)) for item in NUMERIC_BENCHMARK_ITEMS)

    assert total_a >= total_b >= total_c, (
        f"Expected total_a ({total_a}) >= total_b ({total_b}) >= total_c ({total_c})"
    )


@pytest.mark.parametrize("category, items", list(UNICODE_STRESS_ITEMS.items()))
def test_unicode_and_indic_stress(default_tokenizer: BPETokenizer, category: str, items: list):
    """Verify exact round-trip reconstruction for all Indic, Sanskrit, ZWJ/ZWNJ, and Emoji tests."""
    for text in items:
        tokens = default_tokenizer.encode(text)
        decoded = default_tokenizer.decode(tokens)
        assert decoded == text, f"Lossless reconstruction failed in {category} for {text!r}"


def test_raw_byte_arbitrary_sequences(default_tokenizer: BPETokenizer):
    """Verify decode_bytes(encode_bytes(b)) == b on arbitrary, random, and malformed bytes."""
    cases = get_raw_byte_test_cases(seed=42)
    assert len(cases) >= 10

    for case in cases:
        raw_data = case["bytes"]
        tokens = default_tokenizer.encode_bytes(raw_data)
        reconstructed = default_tokenizer.decode_bytes(tokens)
        assert reconstructed == raw_data, f"Raw byte test {case['name']} failed"


def test_repeated_tokenization_determinism(default_tokenizer: BPETokenizer):
    """Verify that identical input always produces bit-exact identical token IDs."""
    test_strings = [
        "ChakrView चक्रव्यूह v0.1 indigenous architecture 🚀",
        "सत्यमेव जयते नानृतम् ॐ 2026-09-26",
        "123 + 456 = 579, ₹50,000, 1.23e-10",
        "👨👩👧👦 👩💻 👍🏽 🇮🇳",
    ]
    for text in test_strings:
        first_tokens = default_tokenizer.encode(text)
        for _ in range(5):
            tokens = default_tokenizer.encode(text)
            assert tokens == first_tokens, f"Non-deterministic encoding on {text!r}"
            decoded = default_tokenizer.decode(tokens)
            assert decoded == text, f"Non-deterministic decoding on {text!r}"


def test_special_tokens_contract():
    """Verify hard special token and base byte invariants."""
    assert BOS_ID == 0
    assert EOS_ID == 1
    assert PAD_ID == 2
    assert BYTE_OFFSET == 3
    assert NUM_BYTE_TOKENS == 256
