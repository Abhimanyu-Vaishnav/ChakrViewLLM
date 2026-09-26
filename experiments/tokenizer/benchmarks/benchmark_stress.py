"""
ChakrView Unicode, Indic & Raw Byte Stress Test Benchmark (Phases 3.7 & 3.8).

Evaluates:
- Phase 3.7: Devanagari Hindi, classical Sanskrit, ZWJ/ZWNJ half-forms, emoji sequences,
  combining marks, variation selectors, uncommon scripts, and mixed-script text.
- Phase 3.8: Arbitrary raw byte sequences (00, FF, 00 FF 00, random 256 bytes,
  random 1024 bytes, invalid UTF-8 octets, truncated multi-byte UTF-8 prefixes).

Mandatory Invariants:
- Decode(Encode(text)) == text
- decode_bytes(encode_bytes(data)) == data
"""

from typing import Any, Dict, List

from chakrview.tokenizer.tokenizer import BPETokenizer
from experiments.tokenizer.fixtures.fixtures import (
    UNICODE_STRESS_ITEMS,
    get_raw_byte_test_cases,
)


def run_unicode_stress_test(tokenizer: BPETokenizer) -> Dict[str, Any]:
    """
    Run Phase 3.7 Unicode & Indic stress tests against the tokenizer.
    """
    results: Dict[str, Any] = {
        "benchmark_name": "Phase_3_7_Unicode_Indic_Stress_Test",
        "categories": {},
        "all_lossless": True,
        "total_items_tested": 0,
    }

    total_items = 0
    all_ok = True

    for cat_name, items in UNICODE_STRESS_ITEMS.items():
        cat_results: List[Dict[str, Any]] = []
        for text in items:
            total_items += 1
            tokens = tokenizer.encode(text)
            decoded = tokenizer.decode(tokens)
            is_lossless = (decoded == text)
            if not is_lossless:
                all_ok = False

            cat_results.append({
                "text": text,
                "utf8_bytes": len(text.encode("utf-8")),
                "token_count": len(tokens),
                "token_ids": tokens,
                "lossless": is_lossless,
            })

        results["categories"][cat_name] = {
            "item_count": len(items),
            "all_lossless": all(r["lossless"] for r in cat_results),
            "items": cat_results,
        }

    results["total_items_tested"] = total_items
    results["all_lossless"] = all_ok
    return results


def run_raw_byte_stress_test(tokenizer: BPETokenizer, seed: int = 42) -> Dict[str, Any]:
    """
    Run Phase 3.8 Raw Byte stress tests against the tokenizer.
    """
    cases = get_raw_byte_test_cases(seed=seed)
    case_results: List[Dict[str, Any]] = []
    all_ok = True

    for case in cases:
        raw_data = case["bytes"]
        tokens = tokenizer.encode_bytes(raw_data)
        reconstructed = tokenizer.decode_bytes(tokens)
        is_lossless = (reconstructed == raw_data)
        if not is_lossless:
            all_ok = False

        case_results.append({
            "name": case["name"],
            "byte_length": len(raw_data),
            "token_count": len(tokens),
            "lossless": is_lossless,
        })

    return {
        "benchmark_name": "Phase_3_8_Raw_Byte_Stress_Test",
        "random_seed": seed,
        "total_cases_tested": len(cases),
        "all_lossless": all_ok,
        "cases": case_results,
    }
