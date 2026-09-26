"""
ChakrView Tokenizer Tests: Performance Baseline (Phase J).

Measures simple unoptimized baseline latency, token counts, and byte counts
across representative benchmark texts.

Notice: This is a diagnostic research baseline for the prototype, NOT an
optimized production benchmark. No performance claims are made.
"""

import time
from typing import Dict, List, Tuple

from chakrview.tokenizer.tokenizer import BPETokenizer


BENCHMARK_SAMPLES: List[Tuple[str, str]] = [
    ("English (Short)", "ChakrView is an indigenous neural language model designed for efficiency."),
    ("Hindi (Short)", "नमस्ते! चक्रव्यूह एक स्वदेशी एआई अनुसंधान परियोजना है।"),
    ("Hinglish (Short)", "mujhe Python mein ek function banana hai jo memory optimize kare."),
    ("Source Code", "def add_vectors(a: list, b: list) -> list:\n    return [x + y for x, y in zip(a, b)]"),
    ("Arithmetic & Symbols", "₹50000 at 7.5% per annum for 13 years: Total = ₹98,750.45. ∀x ∈ ℝ."),
]


def test_performance_baseline_measurement():
    """
    Measure baseline encoding/decoding metrics for representative inputs.
    Verifies that operations execute deterministically within expected bounds.
    """
    tokenizer = BPETokenizer()

    # Pre-train a few toy merges so the merge engine path is active
    sample_corpus = " ".join([text for _, text in BENCHMARK_SAMPLES])
    tokenizer.train_toy(sample_corpus, max_merges=10, min_frequency=2)

    results: List[Dict[str, object]] = []

    iterations = 50

    for name, text in BENCHMARK_SAMPLES:
        raw_bytes = text.encode("utf-8")
        byte_len = len(raw_bytes)

        # Measure encoding latency
        t0 = time.perf_counter()
        for _ in range(iterations):
            tokens = tokenizer.encode(text)
        t_enc = (time.perf_counter() - t0) / iterations

        # Measure decoding latency
        t0 = time.perf_counter()
        for _ in range(iterations):
            decoded = tokenizer.decode(tokens)
        t_dec = (time.perf_counter() - t0) / iterations

        assert decoded == text
        tokens_produced = len(tokens)

        results.append({
            "name": name,
            "bytes": byte_len,
            "tokens": tokens_produced,
            "compression_ratio": round(byte_len / tokens_produced, 2) if tokens_produced > 0 else 0.0,
            "encode_latency_us": round(t_enc * 1_000_000, 2),
            "decode_latency_us": round(t_dec * 1_000_000, 2),
        })

    # Assertions to verify baseline validity
    for r in results:
        assert r["bytes"] > 0
        assert r["tokens"] > 0
        assert r["encode_latency_us"] > 0.0
        assert r["decode_latency_us"] > 0.0

    print("\n--- ChakrView Tokenizer Research Prototype Baseline Metrics ---")
    print(f"{'Sample Name':<24} | {'Bytes':<6} | {'Tokens':<6} | {'Bytes/Tok':<10} | {'Enc (µs)':<10} | {'Dec (µs)':<10}")
    print("-" * 76)
    for r in results:
        print(
            f"{r['name']:<24} | {r['bytes']:<6} | {r['tokens']:<6} | "
            f"{r['compression_ratio']:<10} | {r['encode_latency_us']:<10} | {r['decode_latency_us']:<10}"
        )
    print("-" * 76)
