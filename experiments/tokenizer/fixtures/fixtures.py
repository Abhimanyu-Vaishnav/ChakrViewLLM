"""
ChakrView Tokenizer Experiment Fixtures.

Contains deterministic test sets for:
- Phase 3.6 Numeric & Arithmetic Benchmark
- Phase 3.7 Unicode, Indic, Devanagari, Sanskrit, ZWJ/ZWNJ & Emoji Stress Tests
- Phase 3.8 Arbitrary Raw Byte & Invalid/Truncated UTF-8 Payloads
"""

import random
from typing import Dict, List

# Phase 3.6: Dedicated Numeric Benchmark Strings
NUMERIC_BENCHMARK_ITEMS: List[str] = [
    "0",
    "1",
    "9",
    "10",
    "11",
    "12",
    "99",
    "100",
    "101",
    "999",
    "1000",
    "2026",
    "123456789",
    "3.1415926",
    "0.000001",
    "1.23e-10",
    "₹50000",
    "$1000",
    "+91-9876543210",
    "2026-09-26",
    "12:45:59",
    "123 + 456",
    "9999 × 8888",
    "123456789 × 987654321",
    "(123 + 456) / 7",
]

# Phase 3.7: Unicode and Indic Stress Test Strings
UNICODE_STRESS_ITEMS: Dict[str, List[str]] = {
    "hindi": [
        "कौशाम्बी",
        "त्र्यंबकेश्वर",
        "कुंडलियाँ",
        "पूँछ",
        "अँधेरा",
        "ऋग्वेद",
        "ॐ",
    ],
    "sanskrit": [
        "सत्त्व",
        "दग्ध",
        "बुद्ध",
        "उष्ट्र",
        "कार्त्तिकेय",
        "वाग्देवी",
    ],
    "zwj_zwnj": [
        "\u200D",  # ZWJ
        "\u200C",  # ZWNJ
        "क्\u200Dष",  # Half-form ka + ssa with ZWJ
        "क्\u200Cष",  # Virama ka + ssa with ZWNJ
        "श्री\u200Dमान्",
    ],
    "emojis": [
        "👨👩👧👦",  # Family sequence (multiple codepoints)
        "👩💻",      # Woman technologist
        "👍🏽",       # Thumbs up medium skin tone
        "🇮🇳",       # Flag: India
        "🚀❤️🔥",
        "🏳️‍🌈",     # Rainbow flag with ZWJ
    ],
    "combining_and_special": [
        "e\u0301",          # e with combining acute accent
        "a\u0308\u0304",    # a with combining diaeresis and macron
        "\u2764\ufe0f",     # Red heart with variation selector-16
        "\u2600\ufe0e",     # Sun with variation selector-15
        "\u200b",           # Zero-width space
        "\ufeff",           # Zero-width no-break space (BOM)
        "\u2060",           # Word joiner
    ],
    "uncommon_scripts": [
        "வணக்கம்",          # Tamil
        "নমস্কার",           # Bengali
        "నమస్కారం",         # Telugu
        "مرحبا",            # Arabic
        "Привет",           # Cyrillic
        "こんにちは",        # Japanese Hiragana
    ],
    "mixed_script": [
        "ChakrView चक्रव्यूह v0.1: Indigenous AI model ₹50,000 🚀",
        "def नमस्ते_world(): return 'सत्त्व' + 123",
        "संस्कृतम् (Sanskrit) & हिंदी (Hindi) benchmarks in 2026.",
        "Query: ' त्र्यंबकेश्वर मंदिर ' at 12:45:59 on 2026-09-26",
    ],
}


def get_raw_byte_test_cases(seed: int = 42) -> List[Dict[str, bytes]]:
    """
    Generate deterministic raw byte payloads for Phase 3.8.
    """
    rng = random.Random(seed)
    cases: List[Dict[str, bytes]] = [
        {"name": "single_zero_byte", "bytes": b"\x00"},
        {"name": "single_ff_byte", "bytes": b"\xff"},
        {"name": "alternating_00_ff_00", "bytes": b"\x00\xff\x00"},
        {"name": "all_256_bytes_ascending", "bytes": bytes(range(256))},
        {"name": "all_256_bytes_descending", "bytes": bytes(range(255, -1, -1))},
        {
            "name": "random_256_bytes",
            "bytes": bytes(rng.randint(0, 255) for _ in range(256)),
        },
        {
            "name": "random_1024_bytes",
            "bytes": bytes(rng.randint(0, 255) for _ in range(1024)),
        },
        {
            "name": "invalid_utf8_isolated_high_bytes",
            "bytes": b"\x80\x81\x82\xfe\xff\xc0\xaf",
        },
        {
            "name": "truncated_utf8_hindi_prefix",
            # b"\xe0\xa4" is first 2 bytes of 3-byte Devanagari Ka (\xe0\xa4\x95)
            "bytes": b"\xe0\xa4",
        },
        {
            "name": "truncated_utf8_emoji_prefix",
            # b"\xf0\x9f\x98" is first 3 bytes of 4-byte grinning face (\xf0\x9f\x98\x80)
            "bytes": b"\xf0\x9f\x98",
        },
        {
            "name": "mixed_valid_and_invalid_bytes",
            "bytes": b"ChakrView\x80\xff\x00AI\xe0\xa4Core\xfe",
        },
    ]
    return cases
