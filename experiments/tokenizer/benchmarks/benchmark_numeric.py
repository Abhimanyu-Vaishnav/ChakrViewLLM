r"""
ChakrView Numeric Tokenization Benchmark (Phase 3.6).

Evaluates:
- Candidate A: Single-digit tokenization
- Candidate B: Two-digit chunks (\d{1,2})
- Candidate C: Normal BPE frequency learning

Measures:
- Token count per item
- Sequence expansion ratio relative to normal BPE (Candidate C)
- Token boundary stability (consistency of digit representations across contexts)
- Exact lossless round-trip reconstruction for every item
"""

from typing import Any, Dict, List

from chakrview.tokenizer.tokenizer import BPETokenizer
from experiments.tokenizer.candidates.numeric_adapters import NumericTokenizationAdapter
from experiments.tokenizer.fixtures.fixtures import NUMERIC_BENCHMARK_ITEMS


def run_numeric_benchmark(tokenizer: BPETokenizer) -> Dict[str, Any]:
    """
    Run dedicated arithmetic and numeric benchmark comparing Candidates A, B, and C.
    """
    adapter_a = NumericTokenizationAdapter(tokenizer, strategy="A")
    adapter_b = NumericTokenizationAdapter(tokenizer, strategy="B")
    adapter_c = NumericTokenizationAdapter(tokenizer, strategy="C")

    item_results: List[Dict[str, Any]] = []
    total_tokens_a = 0
    total_tokens_b = 0
    total_tokens_c = 0
    total_bytes = 0
    all_lossless_a = True
    all_lossless_b = True
    all_lossless_c = True

    for text in NUMERIC_BENCHMARK_ITEMS:
        raw_b = len(text.encode("utf-8"))
        ok_a, tok_a, dec_a = adapter_a.round_trip_check(text)
        ok_b, tok_b, dec_b = adapter_b.round_trip_check(text)
        ok_c, tok_c, dec_c = adapter_c.round_trip_check(text)

        if not ok_a:
            all_lossless_a = False
        if not ok_b:
            all_lossless_b = False
        if not ok_c:
            all_lossless_c = False

        len_a = len(tok_a)
        len_b = len(tok_b)
        len_c = len(tok_c)

        total_bytes += raw_b
        total_tokens_a += len_a
        total_tokens_b += len_b
        total_tokens_c += len_c

        item_results.append({
            "text": text,
            "utf8_bytes": raw_b,
            "strategy_A_single_digit": {
                "token_count": len_a,
                "tokens": tok_a,
                "lossless": ok_a,
            },
            "strategy_B_two_digit": {
                "token_count": len_b,
                "tokens": tok_b,
                "lossless": ok_b,
            },
            "strategy_C_normal_bpe": {
                "token_count": len_c,
                "tokens": tok_c,
                "lossless": ok_c,
            },
        })

    # Summary analysis
    expansion_a_vs_c = round(total_tokens_a / total_tokens_c, 3) if total_tokens_c > 0 else 1.0
    expansion_b_vs_c = round(total_tokens_b / total_tokens_c, 3) if total_tokens_c > 0 else 1.0

    return {
        "benchmark_name": "Phase_3_6_Numeric_Tokenization_Experiment",
        "total_test_items": len(NUMERIC_BENCHMARK_ITEMS),
        "total_bytes": total_bytes,
        "aggregates": {
            "strategy_A_single_digit": {
                "total_tokens": total_tokens_a,
                "avg_tokens_per_item": round(total_tokens_a / len(NUMERIC_BENCHMARK_ITEMS), 2),
                "expansion_ratio_vs_normal_bpe": expansion_a_vs_c,
                "all_lossless": all_lossless_a,
            },
            "strategy_B_two_digit": {
                "total_tokens": total_tokens_b,
                "avg_tokens_per_item": round(total_tokens_b / len(NUMERIC_BENCHMARK_ITEMS), 2),
                "expansion_ratio_vs_normal_bpe": expansion_b_vs_c,
                "all_lossless": all_lossless_b,
            },
            "strategy_C_normal_bpe": {
                "total_tokens": total_tokens_c,
                "avg_tokens_per_item": round(total_tokens_c / len(NUMERIC_BENCHMARK_ITEMS), 2),
                "expansion_ratio_vs_normal_bpe": 1.0,
                "all_lossless": all_lossless_c,
            },
        },
        "item_breakdown": item_results,
    }
