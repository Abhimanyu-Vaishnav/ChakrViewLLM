"""
ChakrView Vocabulary Candidate Evaluator (Phase 3.4 & 3.5).

Evaluates candidate vocabularies (V = 2048, 4096, 8192, 16384) on the controlled corpus.
Measures the 20 mandatory Phase 3.5 metrics:
1. Total token count
2. Total UTF-8 byte count
3. Tokens / character
4. Tokens / word
5. Bytes / token
6. Compression ratio (bytes / tokens)
7. Hindi token efficiency
8. English token efficiency
9. Hinglish token efficiency
10. Sanskrit token efficiency
11. Code token efficiency
12. Mathematics token efficiency
13. Numeric token efficiency
14. Unicode token efficiency
15. Emoji token efficiency
16. Encode latency
17. Decode latency
18. Process memory footprint
19. Tokenizer model / merge-table memory
20. Number of learned merges
"""

import sys
import time
from typing import Any, Dict, List, Tuple

from chakrview.tokenizer.benchmark import calculate_vocab_memory
from chakrview.tokenizer.tokenizer import BPETokenizer
from chakrview.tokenizer_corpus.loader import CorpusItem


def evaluate_candidate_on_corpus(
    candidate_name: str,
    tokenizer: BPETokenizer,
    corpus_items: List[CorpusItem],
    latency_repetitions: int = 5,
) -> Dict[str, Any]:
    """
    Evaluate a single tokenizer candidate on the full benchmark corpus.
    """
    total_bytes = 0
    total_chars = 0
    total_words = 0
    total_tokens = 0
    failed_reconstructions = 0

    category_stats: Dict[str, Dict[str, Any]] = {}

    for item in corpus_items:
        text = item.text
        raw_b = len(text.encode("utf-8"))
        c_len = len(text)
        w_len = max(1, len(text.split()))

        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)

        if decoded != text:
            failed_reconstructions += 1

        t_len = len(tokens)
        total_bytes += raw_b
        total_chars += c_len
        total_words += w_len
        total_tokens += t_len

        cat = item.category
        if cat not in category_stats:
            category_stats[cat] = {
                "bytes": 0,
                "chars": 0,
                "words": 0,
                "tokens": 0,
                "item_count": 0,
            }

        category_stats[cat]["bytes"] += raw_b
        category_stats[cat]["chars"] += c_len
        category_stats[cat]["words"] += w_len
        category_stats[cat]["tokens"] += t_len
        category_stats[cat]["item_count"] += 1

    # Detailed category breakdown
    category_metrics: Dict[str, Dict[str, float]] = {}
    for cat, data in sorted(category_stats.items()):
        b = data["bytes"]
        c = data["chars"]
        w = data["words"]
        t = data["tokens"]
        category_metrics[cat] = {
            "item_count": data["item_count"],
            "bytes": b,
            "chars": c,
            "words": w,
            "tokens": t,
            "tokens_per_char": round(t / c, 4) if c > 0 else 0.0,
            "tokens_per_word": round(t / w, 4) if w > 0 else 0.0,
            "bytes_per_token": round(b / t, 4) if t > 0 else 0.0,
            "compression_ratio": round(b / t, 4) if t > 0 else 0.0,
        }

    # Measure Encode Latency
    t0 = time.perf_counter()
    for _ in range(latency_repetitions):
        for item in corpus_items:
            _ = tokenizer.encode(item.text)
    total_enc_time = time.perf_counter() - t0
    avg_enc_us = (total_enc_time / (len(corpus_items) * latency_repetitions)) * 1_000_000

    # Measure Decode Latency
    pre_encoded = [tokenizer.encode(item.text) for item in corpus_items]
    t0 = time.perf_counter()
    for _ in range(latency_repetitions):
        for seq in pre_encoded:
            _ = tokenizer.decode(seq)
    total_dec_time = time.perf_counter() - t0
    avg_dec_us = (total_dec_time / (len(corpus_items) * latency_repetitions)) * 1_000_000

    # Overall metrics
    overall_tpc = round(total_tokens / total_chars, 4) if total_chars > 0 else 0.0
    overall_tpw = round(total_tokens / total_words, 4) if total_words > 0 else 0.0
    overall_bpt = round(total_bytes / total_tokens, 4) if total_tokens > 0 else 0.0
    mem_bytes = calculate_vocab_memory(tokenizer)

    # Specific Category Efficiencies requested in Phase 3.5
    def get_efficiency(key_substr: str) -> Dict[str, float]:
        matches = [m for cat, m in category_metrics.items() if key_substr.lower() in cat.lower()]
        if not matches:
            return {"tokens_per_word": 0.0, "bytes_per_token": 0.0}
        avg_tpw = sum(m["tokens_per_word"] for m in matches) / len(matches)
        avg_bpt = sum(m["bytes_per_token"] for m in matches) / len(matches)
        return {
            "tokens_per_word": round(avg_tpw, 3),
            "bytes_per_token": round(avg_bpt, 3),
        }

    metrics_20: Dict[str, Any] = {
        "candidate_name": candidate_name,
        "1_total_token_count": total_tokens,
        "2_total_utf8_byte_count": total_bytes,
        "3_tokens_per_character": overall_tpc,
        "4_tokens_per_word": overall_tpw,
        "5_bytes_per_token": overall_bpt,
        "6_compression_ratio": overall_bpt,
        "7_hindi_efficiency": get_efficiency("hindi"),
        "8_english_efficiency": get_efficiency("english"),
        "9_hinglish_efficiency": get_efficiency("hinglish"),
        "10_sanskrit_efficiency": get_efficiency("sanskrit"),
        "11_code_efficiency": get_efficiency("code"),
        "12_math_efficiency": get_efficiency("math"),
        "13_numeric_efficiency": get_efficiency("number"),
        "14_unicode_efficiency": get_efficiency("unicode"),
        "15_emoji_efficiency": get_efficiency("emoji"),
        "16_encode_latency_us_per_item": round(avg_enc_us, 2),
        "17_decode_latency_us_per_item": round(avg_dec_us, 2),
        "18_process_memory_kb": round(mem_bytes / 1024, 2),
        "19_tokenizer_merge_table_bytes": mem_bytes,
        "20_number_of_learned_merges": tokenizer.num_merges,
        "actual_vocab_size": tokenizer.vocab_size,
        "failed_reconstructions": failed_reconstructions,
        "is_lossless": (failed_reconstructions == 0),
        "category_breakdown": category_metrics,
    }

    return metrics_20
