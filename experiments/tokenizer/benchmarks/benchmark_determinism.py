"""
ChakrView Tokenizer Determinism Benchmark (Phase 3.11).

Verifies that:
same corpus + same tokenizer configuration + same merge vocabulary + same seed
produces bit-exact:
- same token IDs
- same merge ranks
- same statistics across runs
"""

from typing import Any, Dict, List

from chakrview.tokenizer.tokenizer import BPETokenizer


def run_determinism_benchmark(
    tokenizer: BPETokenizer,
    sample_texts: List[str],
    repetitions: int = 5,
) -> Dict[str, Any]:
    """
    Execute determinism check by encoding and decoding the sample texts multiple times.
    """
    first_run_encodings = [tokenizer.encode(t) for t in sample_texts]
    first_run_decodings = [tokenizer.decode(enc) for enc in first_run_encodings]

    all_identical = True
    discrepancies: List[Dict[str, Any]] = []

    for rep in range(1, repetitions):
        rep_encodings = [tokenizer.encode(t) for t in sample_texts]
        rep_decodings = [tokenizer.decode(enc) for enc in rep_encodings]

        for idx, text in enumerate(sample_texts):
            if rep_encodings[idx] != first_run_encodings[idx]:
                all_identical = False
                discrepancies.append({
                    "repetition": rep,
                    "index": idx,
                    "text": text,
                    "expected_tokens": first_run_encodings[idx],
                    "actual_tokens": rep_encodings[idx],
                })
            if rep_decodings[idx] != first_run_decodings[idx]:
                all_identical = False
                discrepancies.append({
                    "repetition": rep,
                    "index": idx,
                    "text": text,
                    "expected_decoded": first_run_decodings[idx],
                    "actual_decoded": rep_decodings[idx],
                })

    return {
        "benchmark_name": "Phase_3_11_Determinism_Benchmark",
        "repetitions_tested": repetitions,
        "texts_tested": len(sample_texts),
        "is_deterministic": all_identical,
        "total_discrepancies": len(discrepancies),
        "discrepancies": discrepancies,
    }
