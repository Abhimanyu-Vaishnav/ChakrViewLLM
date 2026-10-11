"""
ChakrView Tokenizer: Metrics & Accounting Engine.

Provides unified, rigorous measurement functions for:
1. Sequence efficiency: tokens/char, tokens/word, bytes/token, compression ratio.
2. Latency statistics: mean, median, p95, min, max, std (microseconds) for encode and decode.
3. Memory accounting: static vocabulary memory, merge table memory, embedding table parameters.
4. Category breakdowns: Hindi, English, Hinglish, Sanskrit, Code, Mathematics, Numbers, Mixed.
"""

import math
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

from chakrview.tokenizer.tokenizer import BPETokenizer


@dataclass(frozen=True)
class TokenMetrics:
    """Quantitative tokenization efficiency metrics."""
    tokens: int
    characters: int
    words: int
    raw_bytes: int
    tokens_per_char: float
    tokens_per_word: float
    bytes_per_token: float
    compression_ratio: float


@dataclass(frozen=True)
class LatencyStats:
    """Latency distribution measurements in microseconds (us)."""
    mean_us: float
    median_us: float
    p95_us: float
    min_us: float
    max_us: float
    std_us: float
    iterations: int


@dataclass(frozen=True)
class MemoryBreakdown:
    """Memory consumption of tokenizer data structures and downstream embedding table."""
    vocab_size: int
    vocab_memory_bytes: int
    merge_table_bytes: int
    total_static_memory_bytes: int
    total_static_memory_kb: float
    d_model: int
    embedding_params: int
    embedding_bytes_fp16: int
    embedding_bytes_int8: int
    embedding_param_percentage: float  # Percentage of 3.44M Chakr-Micro parameters


def compute_sequence_metrics(
    text: str,
    tokens: List[int],
    raw_bytes: Optional[bytes] = None,
) -> TokenMetrics:
    """Compute efficiency metrics for a single text and token sequence."""
    b = raw_bytes if raw_bytes is not None else text.encode("utf-8")
    b_len = len(b)
    c_len = len(text)
    w_len = max(1, len(text.split()))
    t_len = len(tokens)

    tpc = round(t_len / c_len, 4) if c_len > 0 else 0.0
    tpw = round(t_len / w_len, 4) if w_len > 0 else 0.0
    bpt = round(b_len / t_len, 4) if t_len > 0 else 0.0
    comp = bpt

    return TokenMetrics(
        tokens=t_len,
        characters=c_len,
        words=w_len,
        raw_bytes=b_len,
        tokens_per_char=tpc,
        tokens_per_word=tpw,
        bytes_per_token=bpt,
        compression_ratio=comp,
    )


def compute_latency_stats(times_us: List[float]) -> LatencyStats:
    """Calculate summary statistics from a series of execution times in microseconds."""
    if not times_us:
        return LatencyStats(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0)

    sorted_times = sorted(times_us)
    n = len(sorted_times)
    mean_v = statistics.fmean(sorted_times)
    median_v = statistics.median(sorted_times)
    p95_idx = int(0.95 * n)
    p95_v = sorted_times[min(p95_idx, n - 1)]
    min_v = sorted_times[0]
    max_v = sorted_times[-1]
    std_v = statistics.stdev(sorted_times) if n > 1 else 0.0

    return LatencyStats(
        mean_us=round(mean_v, 2),
        median_us=round(median_v, 2),
        p95_us=round(p95_v, 2),
        min_us=round(min_v, 2),
        max_us=round(max_v, 2),
        std_us=round(std_v, 2),
        iterations=n,
    )


def measure_tokenizer_memory(
    tokenizer: BPETokenizer,
    d_model: int = 192,
    total_model_params: int = 3_443_712,
) -> MemoryBreakdown:
    """
    Measure static in-memory footprint of vocabulary and merge tables,
    plus calculate downstream neural-core embedding parameter weight.
    """
    vocab_mem = sys.getsizeof(tokenizer.vocab)
    for tid, b in tokenizer.vocab.items():
        vocab_mem += sys.getsizeof(tid) + sys.getsizeof(b)

    merges_mem = sys.getsizeof(tokenizer.merges)
    for pair, tid in tokenizer.merges.items():
        merges_mem += sys.getsizeof(pair) + sys.getsizeof(tid)

    total_static = vocab_mem + merges_mem
    v_size = tokenizer.vocab_size
    emb_params = v_size * d_model
    emb_fp16 = emb_params * 2
    emb_int8 = emb_params * 1
    pct = round((emb_params / total_model_params) * 100, 2)

    return MemoryBreakdown(
        vocab_size=v_size,
        vocab_memory_bytes=vocab_mem,
        merge_table_bytes=merges_mem,
        total_static_memory_bytes=total_static,
        total_static_memory_kb=round(total_static / 1024, 2),
        d_model=d_model,
        embedding_params=emb_params,
        embedding_bytes_fp16=emb_fp16,
        embedding_bytes_int8=emb_int8,
        embedding_param_percentage=pct,
    )


def measure_tokenizer_latency(
    tokenizer: BPETokenizer,
    texts: List[str],
    repetitions: int = 20,
    warmup: int = 3,
) -> Dict[str, LatencyStats]:
    """
    Benchmark encode and decode latency separately across repetitions.
    Includes explicit warmup to stabilize CPU branch predictors and caches.
    """
    # 1. Warmup
    for _ in range(warmup):
        for text in texts:
            tokens = tokenizer.encode(text)
            _ = tokenizer.decode(tokens)

    # 2. Benchmark Encode
    enc_times_us: List[float] = []
    for _ in range(repetitions):
        t0 = time.perf_counter()
        for text in texts:
            _ = tokenizer.encode(text)
        t_el = time.perf_counter() - t0
        enc_times_us.append((t_el / len(texts)) * 1_000_000)

    # 3. Pre-encode tokens for decode test
    all_tokens = [tokenizer.encode(text) for text in texts]

    # 4. Benchmark Decode
    dec_times_us: List[float] = []
    for _ in range(repetitions):
        t0 = time.perf_counter()
        for seq in all_tokens:
            _ = tokenizer.decode(seq)
        t_el = time.perf_counter() - t0
        dec_times_us.append((t_el / len(texts)) * 1_000_000)

    return {
        "encode": compute_latency_stats(enc_times_us),
        "decode": compute_latency_stats(dec_times_us),
    }


def evaluate_split_metrics(
    tokenizer: BPETokenizer,
    split_categories: Dict[str, List[str]],
    measure_latencies: bool = True,
    d_model: int = 192,
) -> Dict[str, Any]:
    """
    Evaluate tokenizer across an entire split organized by category.
    """
    total_tokens = 0
    total_chars = 0
    total_words = 0
    total_bytes = 0
    failed_reconstructions = 0

    cat_metrics: Dict[str, Dict[str, Any]] = {}
    all_texts: List[str] = []

    for cat, texts in sorted(split_categories.items()):
        c_tok = 0
        c_char = 0
        c_word = 0
        c_byte = 0
        c_fails = 0

        for text in texts:
            all_texts.append(text)
            b = text.encode("utf-8")
            tokens = tokenizer.encode(text)
            dec = tokenizer.decode(tokens)

            if dec != text:
                c_fails += 1
                failed_reconstructions += 1

            t_len = len(tokens)
            c_len = len(text)
            w_len = max(1, len(text.split()))
            b_len = len(b)

            c_tok += t_len
            c_char += c_len
            c_word += w_len
            c_byte += b_len

            total_tokens += t_len
            total_chars += c_len
            total_words += w_len
            total_bytes += b_len

        cat_metrics[cat] = {
            "items": len(texts),
            "tokens": c_tok,
            "bytes": c_byte,
            "chars": c_char,
            "words": c_word,
            "tokens_per_char": round(c_tok / c_char, 4) if c_char > 0 else 0.0,
            "tokens_per_word": round(c_tok / c_word, 4) if c_word > 0 else 0.0,
            "bytes_per_token": round(c_byte / c_tok, 4) if c_tok > 0 else 0.0,
            "compression_ratio": round(c_byte / c_tok, 4) if c_tok > 0 else 0.0,
            "failed_reconstructions": c_fails,
        }

    overall_tpc = round(total_tokens / total_chars, 4) if total_chars > 0 else 0.0
    overall_tpw = round(total_tokens / total_words, 4) if total_words > 0 else 0.0
    overall_bpt = round(total_bytes / total_tokens, 4) if total_tokens > 0 else 0.0

    mem = measure_tokenizer_memory(tokenizer, d_model=d_model)

    result: Dict[str, Any] = {
        "vocab_size": tokenizer.vocab_size,
        "total_items": len(all_texts),
        "total_bytes": total_bytes,
        "total_chars": total_chars,
        "total_words": total_words,
        "total_tokens": total_tokens,
        "tokens_per_char": overall_tpc,
        "tokens_per_word": overall_tpw,
        "bytes_per_token": overall_bpt,
        "compression_ratio": overall_bpt,
        "failed_reconstructions": failed_reconstructions,
        "lossless_verified": (failed_reconstructions == 0),
        "memory": asdict(mem),
        "categories": cat_metrics,
    }

    if measure_latencies and all_texts:
        lat = measure_tokenizer_latency(tokenizer, all_texts)
        result["latency"] = {
            "encode": asdict(lat["encode"]),
            "decode": asdict(lat["decode"]),
        }

    return result
