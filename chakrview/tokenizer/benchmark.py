"""
ChakrView Tokenizer: Comprehensive Benchmark & Empirical Selection Engine.

Implements all required evaluations for Step 3:
1. Multi-Vocabulary Candidates (V = 2048, 4096, 8192, 16384) trained strictly on
   data/processed/train/ and evaluated on unseen data/validation/ and data/processed/test/.
2. 3-Way Numeric Strategy (Candidate A: single digits, Candidate B: 2-digit chunks, Candidate C: normal BPE)
   across arithmetic, long numbers, decimals, negatives, percentages, currency, dates, timestamps, versions, IPs.
3. Grapheme-Aware Evaluation: Variant A (raw byte BPE) vs Variant B (grapheme-aware pre-tokenization + BPE)
   on Hindi, Sanskrit, English, Hinglish, Code.
4. Latency Distribution: Mean, Median, P95, Min, Max for both encode and decode.
5. Memory Footprint: Static vocab/merge tables, embedding parameters and memory at d_model=192.
6. Lossless Invariant Verification: Decode(Encode(x)) == x and decode_bytes(encode_bytes(b)) == b.
"""

import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from chakrview.tokenizer.corpus import CorpusItem, load_corpus_split
from chakrview.tokenizer.metrics import (
    LatencyStats,
    MemoryBreakdown,
    TokenMetrics,
    compute_latency_stats,
    compute_sequence_metrics,
    measure_tokenizer_latency,
    measure_tokenizer_memory,
)
from chakrview.tokenizer.tokenizer import BPETokenizer
from chakrview.tokenizer.trainer import BPETrainer

# Standard test fixtures for language and numeric evaluation
LANGUAGE_BENCHMARK_FIXTURES: Dict[str, List[str]] = {
    "hindi": [
        "नमस्ते! चक्रव्यूह एक स्वदेशी कृत्रिम बुद्धिमत्ता अनुसंधान परियोजना है।",
        "कौशाम्बी, त्र्यंबकेश्वर, कुंडलियाँ, पूँछ, अँधेरा और ऋग्वेद जैसे जटिल शब्द देवनागरी की विशिष्टता को दर्शाते हैं।",
        "संयुक्त व्यंजनों का निर्माण हलंत के माध्यम से होता है, जहां दो या दो से अधिक व्यंजन मिलकर एक नया रूप धारण करते हैं।",
        "कम मेमोरी और कम बिजली खपत वाले उपकरणों पर भाषा मॉडल चलाना एक महत्वपूर्ण वैज्ञानिक चुनौती है।",
    ],
    "sanskrit": [
        "ॐ सह नाववतु। सह नौ भुनक्तु। सह वीर्यं करवावहै। तेजस्वि नावधीतमस्तु मा विद्विषावहै। ॐ शान्तिः शान्तिः शान्तिः॥",
        "सत्यमेव जयते नानृतं सत्येन पन्था विततो देवयानः। येनाक्रमन्त्यृषयो ह्याप्तकामा यत्र तत् सत्यस्य परमं निधानम्॥",
        "सत्त्व, दग्ध, बुद्ध, उष्ट्र, कार्त्तिकेय, वाग्देवी, अष्टाध्यायी, माहेश्वरसूत्राणि।",
        "विद्या ददाति विनयं विनयाद् याति पात्रताम्। पात्रत्वाद् धनमाप्नोति धनाद् धर्मं ततः सुखम्॥",
    ],
    "english": [
        "ChakrView is an indigenous artificial intelligence research initiative created to build a lightweight neural architecture.",
        "When a transformer model evaluates next-token probabilities, every weight tensor must be retrieved from main memory into registers.",
        "Vectorized single-instruction multiple-data (SIMD) instruction sets such as AVX2, AVX-512, and ARM NEON allow parallel arithmetic.",
        "The trade-off between vocabulary size, token fertility, and embedding parameter footprint is a central axis of micro-model design.",
    ],
    "hinglish": [
        "Aaj hum ChakrView ka neural core build kar rahe hain.",
        "bhai mujhe ek lightweight indigenous AI model develop karna hai jo low-spec CPU par smoothly run kare.",
        "kya ChakrView bina discrete GPU ke sirf Intel ya AMD processor par inferencing kar sakta hai?",
        "RAM consumption aur CPU clock cycles optimize karke hum purane 28nm processors par testing karenge.",
    ],
    "code": [
        "def compute_entropy(p: float) -> float:\n    import math\n    return -p * math.log2(p) if p > 0.0 else 0.0",
        '{\n  "project": "ChakrView",\n  "version": "0.1.0",\n  "vocab_size": 4096,\n  "d_model": 192\n}',
        "class SimpleMultiHeadAttention:\n    def __init__(self, d_model: int = 192, n_heads: int = 6) -> None:\n        self.d_head = d_model // n_heads",
        "python -m pytest tests/ -v --tb=short",
    ],
    "mathematics": [
        "∀x ∈ A: x + y = z",
        "∫ f(x) dx = F(x) + C",
        "E = mc²",
        "e^{iπ} + 1 = 0",
        "RoPE(q, m) = [q_0 * cos(m * θ) - q_1 * sin(m * θ), q_0 * sin(m * θ) + q_1 * cos(m * θ)]",
    ],
    "numbers": [
        "0 42 2026 3.1415926 -98765",
        "₹50000 13.56% 192.168.1.1 2026-09-26 12:45:31",
        "123 + 456 = 579, 9999 * 8888 = 88871112",
        "v0.1.0-rc1, 1.23e-10, 6.022e+23, +91-9876543210",
    ],
}

NUMERIC_TEST_ITEMS = [
    "0",
    "42",
    "2026",
    "3.1415926",
    "-98765",
    "₹50000",
    "13.56%",
    "192.168.1.1",
    "2026-09-26",
    "12:45:31",
    "1234567890",
    "1.23e-10",
    "v0.1.0",
    "123 + 456 = 579",
]


def calculate_vocab_memory(tokenizer: BPETokenizer) -> int:
    """Compute approximate memory footprint in bytes for vocab and merge tables."""
    mem = sys.getsizeof(tokenizer.vocab) + sys.getsizeof(tokenizer.merges)
    for tid, b in tokenizer.vocab.items():
        mem += sys.getsizeof(tid) + sys.getsizeof(b)
    for pair, tid in tokenizer.merges.items():
        mem += sys.getsizeof(pair) + sys.getsizeof(tid)
    return mem


def encode_candidate_digits(text: str, tokenizer: BPETokenizer, strategy: str) -> Tuple[List[int], str]:
    r"""
    Encode text using one of the three numeric strategies:
      - 'A': individual digits (\d)
      - 'B': common two-digit chunks (\d{1,2})
      - 'C': normal frequency-driven BPE
    """
    if strategy == "C":
        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)
        return tokens, decoded

    if strategy == "A":
        pattern = r"(\d|\D+)"
    elif strategy == "B":
        pattern = r"(\d{1,2}|\D+)"
    else:
        raise ValueError(f"Unknown numeric strategy: {strategy}")

    chunks = [m for m in re.findall(pattern, text) if m]
    all_tokens: List[int] = []
    for chunk in chunks:
        all_tokens.extend(tokenizer.encode(chunk))

    decoded = tokenizer.decode(all_tokens)
    return all_tokens, decoded


def run_numeric_experiment(
    tokenizer: BPETokenizer,
    sample_strings: Optional[List[str]] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Evaluate candidate numeric strategies (A: single digits, B: 2-digit chunks, C: normal BPE).
    """
    items = sample_strings or NUMERIC_TEST_ITEMS
    results: Dict[str, Dict[str, Any]] = {}

    for text in items:
        tok_a, dec_a = encode_candidate_digits(text, tokenizer, "A")
        tok_b, dec_b = encode_candidate_digits(text, tokenizer, "B")
        tok_c, dec_c = encode_candidate_digits(text, tokenizer, "C")

        results[text] = {
            "bytes": len(text.encode("utf-8")),
            "strategy_A_individual_digits": {
                "tokens": len(tok_a),
                "reconstruction_lossless": (dec_a == text),
            },
            "strategy_B_two_digit_chunks": {
                "tokens": len(tok_b),
                "reconstruction_lossless": (dec_b == text),
            },
            "strategy_C_normal_bpe": {
                "tokens": len(tok_c),
                "reconstruction_lossless": (dec_c == text),
            },
        }

    return results


def run_grapheme_experiment(
    corpus_lines: List[str],
    test_lines: List[str],
    vocab_size: int = 1024,
) -> Dict[str, Any]:
    """
    Benchmark Variant A (raw byte BPE) vs Variant B (grapheme-aware pre-tokenization + byte BPE).
    """
    # 1. Train Variant A (raw byte BPE)
    trainer_a = BPETrainer(min_frequency=1, pretokenization="byte")
    merges_a, vocab_a, _ = trainer_a.train(corpus_lines, target_vocab_size=vocab_size)
    tok_a = BPETokenizer(merges=merges_a, vocab=vocab_a)

    # 2. Train Variant B (grapheme-aware BPE)
    trainer_b = BPETrainer(min_frequency=1, pretokenization="grapheme")
    merges_b, vocab_b, _ = trainer_b.train(corpus_lines, target_vocab_size=vocab_size)
    tok_b = BPETokenizer(merges=merges_b, vocab=vocab_b)

    def evaluate_variant(tokenizer: BPETokenizer, pretokenization: str) -> Dict[str, Any]:
        total_tokens = 0
        total_chars = 0
        total_words = 0
        total_bytes = 0
        fails = 0

        t0 = time.perf_counter()
        for line in test_lines:
            b = line.encode("utf-8")
            if pretokenization == "grapheme":
                segments = trainer_b.split_segments(line)
                tokens: List[int] = []
                for s in segments:
                    tokens.extend(tokenizer.encode(s))
            else:
                tokens = tokenizer.encode(line)

            dec = tokenizer.decode(tokens)
            if dec != line:
                fails += 1

            total_tokens += len(tokens)
            total_chars += len(line)
            total_words += max(1, len(line.split()))
            total_bytes += len(b)

        elapsed = time.perf_counter() - t0
        avg_enc_us = (elapsed / max(1, len(test_lines))) * 1_000_000

        return {
            "tokens": total_tokens,
            "bytes": total_bytes,
            "chars": total_chars,
            "tokens_per_char": round(total_tokens / total_chars, 4) if total_chars > 0 else 0.0,
            "tokens_per_word": round(total_tokens / total_words, 4) if total_words > 0 else 0.0,
            "bytes_per_token": round(total_bytes / total_tokens, 4) if total_tokens > 0 else 0.0,
            "failed_reconstructions": fails,
            "lossless": (fails == 0),
            "encode_latency_us": round(avg_enc_us, 2),
        }

    return {
        "variant_A_raw_byte_bpe": evaluate_variant(tok_a, "byte"),
        "variant_B_grapheme_aware_bpe": evaluate_variant(tok_b, "grapheme"),
    }


def evaluate_tokenizer_on_corpus(
    tokenizer: BPETokenizer,
    val_items: List[CorpusItem],
    latency_repetitions: int = 10,
) -> Dict[str, Any]:
    """
    Evaluate a tokenizer candidate on validation corpus items.
    """
    total_bytes = 0
    total_chars = 0
    total_words = 0
    total_tokens = 0
    failed_reconstructions = 0

    category_buckets: Dict[str, Dict[str, int]] = {}

    for item in val_items:
        text = item.text
        raw_bytes = text.encode("utf-8")
        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)

        if decoded != text:
            failed_reconstructions += 1

        b_len = len(raw_bytes)
        c_len = len(text)
        w_len = max(1, len(text.split()))
        t_len = len(tokens)

        total_bytes += b_len
        total_chars += c_len
        total_words += w_len
        total_tokens += t_len

        cat = item.category
        if cat not in category_buckets:
            category_buckets[cat] = {"bytes": 0, "chars": 0, "words": 0, "tokens": 0}
        category_buckets[cat]["bytes"] += b_len
        category_buckets[cat]["chars"] += c_len
        category_buckets[cat]["words"] += w_len
        category_buckets[cat]["tokens"] += t_len

    category_metrics: Dict[str, Dict[str, float]] = {}
    for cat, b in sorted(category_buckets.items()):
        c_tokens = b["tokens"]
        c_bytes = b["bytes"]
        c_chars = b["chars"]
        c_words = b["words"]
        category_metrics[cat] = {
            "tokens": c_tokens,
            "bytes": c_bytes,
            "tokens_per_char": round(c_tokens / c_chars, 4) if c_chars > 0 else 0.0,
            "tokens_per_word": round(c_tokens / c_words, 4) if c_words > 0 else 0.0,
            "bytes_per_token": round(c_bytes / c_tokens, 4) if c_tokens > 0 else 0.0,
            "compression_ratio": round(c_bytes / c_tokens, 4) if c_tokens > 0 else 0.0,
        }

    # Measure latency
    t0 = time.perf_counter()
    for _ in range(latency_repetitions):
        for item in val_items:
            _ = tokenizer.encode(item.text)
    total_enc_time = time.perf_counter() - t0
    avg_enc_us = (total_enc_time / (max(1, len(val_items)) * latency_repetitions)) * 1_000_000

    val_token_sequences = [tokenizer.encode(item.text) for item in val_items]
    t0 = time.perf_counter()
    for _ in range(latency_repetitions):
        for seq in val_token_sequences:
            _ = tokenizer.decode(seq)
    total_dec_time = time.perf_counter() - t0
    avg_dec_us = (total_dec_time / (max(1, len(val_items)) * latency_repetitions)) * 1_000_000

    overall_tpc = round(total_tokens / total_chars, 4) if total_chars > 0 else 0.0
    overall_tpw = round(total_tokens / total_words, 4) if total_words > 0 else 0.0
    overall_bpt = round(total_bytes / total_tokens, 4) if total_tokens > 0 else 0.0
    compression_ratio = overall_bpt

    mem_bytes = calculate_vocab_memory(tokenizer)

    return {
        "total_val_items": len(val_items),
        "total_bytes": total_bytes,
        "total_chars": total_chars,
        "total_words": total_words,
        "total_tokens": total_tokens,
        "failed_reconstructions": failed_reconstructions,
        "reconstruction_lossless": (failed_reconstructions == 0),
        "tokens_per_char": overall_tpc,
        "tokens_per_word": overall_tpw,
        "bytes_per_token": overall_bpt,
        "compression_ratio": compression_ratio,
        "encode_latency_us_per_doc": round(avg_enc_us, 2),
        "decode_latency_us_per_doc": round(avg_dec_us, 2),
        "vocab_memory_bytes": mem_bytes,
        "vocab_memory_kb": round(mem_bytes / 1024, 2),
        "category_metrics": category_metrics,
    }


def run_candidate_vocab_suite(
    train_lines: List[str],
    eval_items: List[CorpusItem],
    vocab_sizes: Optional[List[int]] = None,
    d_model: int = 192,
) -> Dict[str, Any]:
    """
    Train candidates on train_lines ONLY, and benchmark on unseen eval_items.
    """
    targets = vocab_sizes or [2048, 4096, 8192, 16384]
    results: Dict[str, Any] = {}

    trainer = BPETrainer(min_frequency=1, tie_breaking_rule="(-frequency, pair[0], pair[1])")

    for target_v in targets:
        print(f"Training Candidate V={target_v}...")
        merges, vocab, train_stats = trainer.train(train_lines, target_vocab_size=target_v)
        tokenizer = BPETokenizer(merges=merges, vocab=vocab)

        eval_stats = evaluate_tokenizer_on_corpus(tokenizer, eval_items, latency_repetitions=10)
        mem = measure_tokenizer_memory(tokenizer, d_model=d_model)

        # Extended latency benchmark with p95, min, max
        sample_texts = [it.text for it in eval_items[:30]]
        lat_dict = measure_tokenizer_latency(tokenizer, sample_texts, repetitions=15, warmup=3)

        results[f"V{target_v}"] = {
            "requested_vocab_size": target_v,
            "actual_vocab_size": train_stats["actual_vocab_size"],
            "merges_performed": train_stats["merges_performed"],
            "training_time_seconds": train_stats["training_time_seconds"],
            "tokens_per_char": eval_stats["tokens_per_char"],
            "tokens_per_word": eval_stats["tokens_per_word"],
            "bytes_per_token": eval_stats["bytes_per_token"],
            "compression_ratio": eval_stats["compression_ratio"],
            "lossless_verified": eval_stats["reconstruction_lossless"],
            "memory": {
                "vocab_memory_bytes": mem.vocab_memory_bytes,
                "merge_table_bytes": mem.merge_table_bytes,
                "total_static_memory_kb": mem.total_static_memory_kb,
                "embedding_params": mem.embedding_params,
                "embedding_param_percentage": mem.embedding_param_percentage,
            },
            "latency": {
                "encode_mean_us": lat_dict["encode"].mean_us,
                "encode_median_us": lat_dict["encode"].median_us,
                "encode_p95_us": lat_dict["encode"].p95_us,
                "decode_mean_us": lat_dict["decode"].mean_us,
                "decode_median_us": lat_dict["decode"].median_us,
                "decode_p95_us": lat_dict["decode"].p95_us,
            },
            "category_metrics": eval_stats["category_metrics"],
        }

    return results
