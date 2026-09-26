"""
ChakrView Tokenizer: Benchmark & Evaluation Engine.

Provides unified evaluation routines across candidate BPE vocabularies:
- Lossless reconstruction verification (Decode(Encode(text)) == text)
- Compression ratio, tokens/char, tokens/word, bytes/token
- Category-specific efficiencies (Hindi, English, Hinglish, Code, Numbers, Math, Unicode)
- Encode/decode latencies
- Memory footprint
- Numeric strategy evaluation (Individual Digits vs Two-Digit Chunks vs Normal BPE)
"""

import re
import sys
import time
from typing import Any, Dict, List, Tuple

from chakrview.tokenizer_corpus.loader import CorpusItem
from chakrview.tokenizer.tokenizer import BPETokenizer


def calculate_vocab_memory(tokenizer: BPETokenizer) -> int:
    """Compute approximate memory footprint in bytes for vocab and merge tables."""
    mem = sys.getsizeof(tokenizer.vocab) + sys.getsizeof(tokenizer.merges)
    for tid, b in tokenizer.vocab.items():
        mem += sys.getsizeof(tid) + sys.getsizeof(b)
    for pair, tid in tokenizer.merges.items():
        mem += sys.getsizeof(pair) + sys.getsizeof(tid)
    return mem


def evaluate_tokenizer_on_corpus(
    tokenizer: BPETokenizer,
    val_items: List[CorpusItem],
    latency_repetitions: int = 10,
) -> Dict[str, Any]:
    """
    Evaluate a tokenizer candidate on a designated validation corpus.
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

        # Lossless reconstruction check
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

    # Category breakdowns
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

    # Measure latency across entire validation corpus
    t0 = time.perf_counter()
    for _ in range(latency_repetitions):
        for item in val_items:
            _ = tokenizer.encode(item.text)
    total_enc_time = time.perf_counter() - t0
    avg_enc_us = (total_enc_time / (len(val_items) * latency_repetitions)) * 1_000_000

    # Pre-encode all tokens for decoding measurement
    val_token_sequences = [tokenizer.encode(item.text) for item in val_items]
    t0 = time.perf_counter()
    for _ in range(latency_repetitions):
        for seq in val_token_sequences:
            _ = tokenizer.decode(seq)
    total_dec_time = time.perf_counter() - t0
    avg_dec_us = (total_dec_time / (len(val_items) * latency_repetitions)) * 1_000_000

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


def encode_candidate_digits(text: str, tokenizer: BPETokenizer, strategy: str) -> Tuple[List[int], str]:
    r"""
    Encode text using one of the three numeric strategies:
      - 'A': individual digits (pre-tokenize digits into single digits \d)
      - 'B': two-digit chunks (\d{1,2})
      - 'C': normal BPE (standard BPE without digit splitting)
    Returns:
        tokens, reconstructed_text
    """
    if strategy == "C":
        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)
        return tokens, decoded

    # For strategies A and B, split input into numeric chunks and non-numeric chunks
    if strategy == "A":
        # Match single digits or non-digit chunks
        pattern = r"(\d|\D+)"
    elif strategy == "B":
        # Match 2 digits, 1 digit, or non-digits
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
    sample_strings: List[str],
) -> Dict[str, Dict[str, Any]]:
    """
    Evaluate candidate numeric strategies (A: single digits, B: 2-digit chunks, C: normal BPE).
    """
    results: Dict[str, Dict[str, Any]] = {}

    for text in sample_strings:
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
