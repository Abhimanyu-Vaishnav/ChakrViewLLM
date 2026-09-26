"""
ChakrView Tokenizer Performance & Memory Benchmark (Phases 3.9 & 3.10).

Measures:
- Cold encode vs Warm encode latency
- Cold decode vs Warm decode latency
- Scale benchmarks across: 32B, 128B, 512B, 2KB, 8KB, 32KB, 128KB
- Latency distribution statistics: median, p95, min, max
- Throughput metrics: tokens/second, MB/second
- Memory profiling: static merge/vocab tables vs runtime process memory

Target Environment: Current development workstation (Windows 11, Intel Core i9-13900H).
"""

import gc
import math
import sys
import time
from typing import Any, Dict, List

from chakrview.tokenizer.benchmark import calculate_vocab_memory
from chakrview.tokenizer.tokenizer import BPETokenizer

# Standard test sizes in bytes
SIZE_SPECS: List[Tuple[str, int]] = [
    ("32_bytes", 32),
    ("128_bytes", 128),
    ("512_bytes", 512),
    ("2_KB", 2048),
    ("8_KB", 8192),
    ("32_KB", 32768),
    ("128_KB", 131072),
]


def generate_benchmark_payload(target_bytes: int) -> str:
    """
    Generate a deterministic, realistic multilingual text payload of approximately target_bytes.
    """
    base_text = (
        "ChakrView (चक्रव्यूह) is an indigenous AI research initiative. "
        "सत्यमेव जयते नानृतम्। "
        "def compute_attention(q, k, v):\n    return softmax(q @ k.T / math.sqrt(d_k)) @ v\n"
        "Testing numbers: ₹50,000, 1.23e-10, 2026-09-26, 12:45:59. "
        "Colloquial Hinglish: mujhe lightweight model banana hai. "
        "Emojis: 🚀 🧠 💻 🇮🇳 ⚡. "
    )
    base_bytes = len(base_text.encode("utf-8"))
    repeats = max(1, math.ceil(target_bytes / base_bytes))
    full_text = (base_text * repeats)

    # Trim to exact byte length safely on character boundary
    encoded = full_text.encode("utf-8")
    if len(encoded) > target_bytes:
        trimmed_bytes = encoded[:target_bytes]
        # Decode ignoring trailing broken UTF-8 byte
        return trimmed_bytes.decode("utf-8", errors="ignore")
    return full_text


def compute_percentiles(times_s: List[float]) -> Dict[str, float]:
    """Compute min, median, p95, max in milliseconds and microseconds."""
    sorted_times = sorted(times_s)
    n = len(sorted_times)
    min_val = sorted_times[0]
    max_val = sorted_times[-1]
    median_val = sorted_times[n // 2]
    p95_idx = int(0.95 * (n - 1))
    p95_val = sorted_times[p95_idx]

    return {
        "min_us": round(min_val * 1_000_000, 2),
        "median_us": round(median_val * 1_000_000, 2),
        "p95_us": round(p95_val * 1_000_000, 2),
        "max_us": round(max_val * 1_000_000, 2),
        "min_ms": round(min_val * 1_000, 3),
        "median_ms": round(median_val * 1_000, 3),
        "p95_ms": round(p95_val * 1_000, 3),
        "max_ms": round(max_val * 1_000, 3),
    }


def run_performance_benchmark(
    tokenizer: BPETokenizer,
    iterations_per_size: int = 15,
) -> Dict[str, Any]:
    """
    Execute performance benchmark across 7 input sizes measuring cold/warm latency and throughput.
    """
    size_results: Dict[str, Any] = {}

    for size_label, target_bytes in SIZE_SPECS:
        payload = generate_benchmark_payload(target_bytes)
        actual_bytes = len(payload.encode("utf-8"))

        # 1. Cold encode
        gc.collect()
        t0 = time.perf_counter()
        cold_tokens = tokenizer.encode(payload)
        cold_enc_s = time.perf_counter() - t0

        # 2. Warm encode iterations
        warm_enc_times: List[float] = []
        for _ in range(iterations_per_size):
            t0 = time.perf_counter()
            tokens = tokenizer.encode(payload)
            warm_enc_times.append(time.perf_counter() - t0)

        # 3. Cold decode
        gc.collect()
        t0 = time.perf_counter()
        cold_decoded = tokenizer.decode(cold_tokens)
        cold_dec_s = time.perf_counter() - t0
        assert cold_decoded == payload, f"Lossless check failed for {size_label}"

        # 4. Warm decode iterations
        warm_dec_times: List[float] = []
        for _ in range(iterations_per_size):
            t0 = time.perf_counter()
            decoded = tokenizer.decode(cold_tokens)
            warm_dec_times.append(time.perf_counter() - t0)
            assert decoded == payload

        token_count = len(cold_tokens)
        enc_stats = compute_percentiles(warm_enc_times)
        dec_stats = compute_percentiles(warm_dec_times)

        # Throughput calculations based on median latency
        median_enc_s = warm_enc_times[len(warm_enc_times) // 2]
        median_dec_s = warm_dec_times[len(warm_dec_times) // 2]

        enc_tok_per_s = round(token_count / median_enc_s, 2) if median_enc_s > 0 else 0.0
        enc_mb_per_s = round((actual_bytes / (1024 * 1024)) / median_enc_s, 2) if median_enc_s > 0 else 0.0

        dec_tok_per_s = round(token_count / median_dec_s, 2) if median_dec_s > 0 else 0.0
        dec_mb_per_s = round((actual_bytes / (1024 * 1024)) / median_dec_s, 2) if median_dec_s > 0 else 0.0

        size_results[size_label] = {
            "target_bytes": target_bytes,
            "actual_bytes": actual_bytes,
            "token_count": token_count,
            "bytes_per_token": round(actual_bytes / token_count, 3) if token_count > 0 else 0.0,
            "cold_encode_us": round(cold_enc_s * 1_000_000, 2),
            "cold_decode_us": round(cold_dec_s * 1_000_000, 2),
            "warm_encode": enc_stats,
            "warm_decode": dec_stats,
            "encode_throughput": {
                "tokens_per_second": enc_tok_per_s,
                "mb_per_second": enc_mb_per_s,
            },
            "decode_throughput": {
                "tokens_per_second": dec_tok_per_s,
                "mb_per_second": dec_mb_per_s,
            },
        }

    return {
        "benchmark_name": "Phase_3_9_Performance_Benchmark",
        "iterations_per_size": iterations_per_size,
        "results_by_size": size_results,
    }


def run_memory_benchmark(tokenizer: BPETokenizer) -> Dict[str, Any]:
    """
    Profile static tokenizer data structures and runtime memory usage.
    """
    gc.collect()

    vocab_dict_size = sys.getsizeof(tokenizer.vocab)
    vocab_entries_size = sum(sys.getsizeof(k) + sys.getsizeof(v) for k, v in tokenizer.vocab.items())
    total_vocab_bytes = vocab_dict_size + vocab_entries_size

    merges_dict_size = sys.getsizeof(tokenizer.merges)
    merges_entries_size = sum(sys.getsizeof(k) + sys.getsizeof(v) for k, v in tokenizer.merges.items())
    total_merges_bytes = merges_dict_size + merges_entries_size

    total_static_bytes = total_vocab_bytes + total_merges_bytes

    return {
        "benchmark_name": "Phase_3_10_Memory_Benchmark",
        "vocab_size": tokenizer.vocab_size,
        "num_merges": tokenizer.num_merges,
        "static_tokenizer_data": {
            "vocab_table_bytes": total_vocab_bytes,
            "vocab_table_kb": round(total_vocab_bytes / 1024, 2),
            "merge_table_bytes": total_merges_bytes,
            "merge_table_kb": round(total_merges_bytes / 1024, 2),
            "total_static_bytes": total_static_bytes,
            "total_static_kb": round(total_static_bytes / 1024, 2),
            "total_static_mb": round(total_static_bytes / (1024 * 1024), 3),
        },
        "classification": "STATIC_TOKENIZER_DATA_ONLY",
        "runtime_notes": "Excludes Python interpreter runtime slab overhead, process virtual memory, and garbage collector tables.",
    }
