"""
ChakrView Step 3: Comprehensive Empirical Tokenizer & Corpus Benchmark Runner.

Executes all 15 Phases of Step 3:
- Phase 2, 3, 4: Corpus validation, statistics, manifest, deduplication, 80/10/10 split
- Phase 5, 8: Vocabulary experiments (V = 2048, 4096, 8192, 16384) with artifacts in data/experiments/vocab_{V}/
- Phase 6: Numeric tokenization experiment (Candidate A vs B vs C)
- Phase 7: Indic and Devanagari stress evaluation
- Phase 9: Vocabulary bias analysis (linguistic/functional classification of learned tokens)
- Phase 10: Performance latency and throughput benchmarks (mean, median, p95, min, max)
- Phase 11: Edge hardware memory and parameter scaling analysis
- Phase 12: Machine-readable data/statistics/step3_tokenizer_benchmark.json
"""

import json
import os
import platform
import re
import sys
import time
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from chakrview.corpus import (
    compute_corpus_statistics,
    generate_corpus_manifest,
    load_corpus_tree,
    partition_corpus,
    save_statistics_report,
    validate_corpus_collection,
)
from chakrview.tokenizer.benchmark import (
    NUMERIC_TEST_ITEMS,
    encode_candidate_digits,
    evaluate_tokenizer_on_corpus,
    run_grapheme_experiment,
    run_numeric_experiment,
)
from chakrview.tokenizer.metrics import (
    measure_tokenizer_latency,
    measure_tokenizer_memory,
)
from chakrview.tokenizer.serialization import save_tokenizer_artifacts
from chakrview.tokenizer.tokenizer import BPETokenizer
from chakrview.tokenizer.trainer import BPETrainer
from experiments.tokenizer.fixtures.fixtures import (
    NUMERIC_BENCHMARK_ITEMS,
    UNICODE_STRESS_ITEMS,
    get_raw_byte_test_cases,
)

DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
VAL_DIR = DATA_DIR / "validation"
MANIFESTS_DIR = DATA_DIR / "manifests"
STATS_DIR = DATA_DIR / "statistics"
EXPERIMENTS_DIR = DATA_DIR / "experiments"

CANDIDATE_VOCAB_SIZES = [2048, 4096, 8192, 16384]


def classify_token_bytes(b: bytes, tid: int) -> str:
    """Classify a token's byte sequence into linguistic / functional category."""
    if tid in (0, 1, 2):
        return "Special_Token"
    if tid < 259:
        return "Base_Byte_Primitive"

    try:
        s = b.decode("utf-8")
    except UnicodeDecodeError:
        return "Non_UTF8_Byte_Sequence"

    has_deva = any(0x0900 <= ord(c) <= 0x097F for c in s)
    has_latin = any(("a" <= c <= "z") or ("A" <= c <= "Z") for c in s)
    has_digit = any("0" <= c <= "9" for c in s)
    has_punct = any(unicodedata.category(c).startswith("P") or c in "{}[]():;=+\\-*/<>_" for c in s)

    if has_deva and not has_latin and not has_digit:
        return "Devanagari"
    if has_latin and not has_deva and not has_digit:
        return "English_Latin"
    if has_digit and not has_deva and not has_latin:
        return "Numeric"
    if has_punct and not has_deva and not has_latin and not has_digit:
        return "Punctuation_Syntax"
    if has_deva and has_latin:
        return "Mixed_Indic_Latin"
    if has_latin and (has_digit or has_punct):
        return "Code_Alphanumeric"
    return "Mixed_Other"


def analyze_vocabulary_bias(tokenizer: BPETokenizer) -> Dict[str, Any]:
    """Analyze the distribution and bias of learned vocabulary tokens."""
    categories: Counter = Counter()
    token_lengths: List[int] = []

    for tid, b in tokenizer.vocab.items():
        cat = classify_token_bytes(b, tid)
        categories[cat] += 1
        token_lengths.append(len(b))

    total = len(tokenizer.vocab)
    pcts = {cat: round((cnt / total) * 100, 2) for cat, cnt in categories.most_common()}

    return {
        "total_tokens_analyzed": total,
        "category_counts": dict(categories.most_common()),
        "category_percentages": pcts,
        "avg_token_byte_length": round(sum(token_lengths) / max(1, total), 2),
        "max_token_byte_length": max(token_lengths) if token_lengths else 0,
        "min_token_byte_length": min(token_lengths) if token_lengths else 0,
    }


def run_comprehensive_benchmark() -> Dict[str, Any]:
    MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)
    STATS_DIR.mkdir(parents=True, exist_ok=True)
    VAL_DIR.mkdir(parents=True, exist_ok=True)
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=================================================================")
    print("CHAKRVIEW STEP 3: COMPREHENSIVE RESEARCH & EMPIRICAL BENCHMARK")
    print("=================================================================")

    # 1. Load and Validate Raw Corpus
    print("\n[Phase 2-4] Loading and Validating Raw Corpus...")
    raw_tree = load_corpus_tree(RAW_DIR)
    all_raw_docs = [doc for cat_docs in raw_tree.values() for doc in cat_docs]

    valid_docs, val_report = validate_corpus_collection(
        all_raw_docs,
        output_report_path=VAL_DIR / "corpus_validation_report.json",
    )
    print(f"  Loaded {len(all_raw_docs)} documents.")
    print(f"  Valid: {len(valid_docs)} | Discarded: {val_report['discarded_documents']} | Duplicates: {val_report['duplicate_documents']}")

    # 2. Compute Corpus Statistics & Manifest
    corpus_stats = compute_corpus_statistics(valid_docs, duplicate_count=val_report["duplicate_documents"])
    save_statistics_report(corpus_stats, STATS_DIR / "corpus_statistics.json")
    corpus_manifest = generate_corpus_manifest(RAW_DIR, output_path=MANIFESTS_DIR / "corpus_manifest.json")
    print(f"  Corpus Characters: {corpus_stats['summary']['total_characters']:,} | Bytes: {corpus_stats['summary']['total_bytes']:,}")
    print(f"  Script distribution: Devanagari {corpus_stats['script_distribution']['percentages'].get('Devanagari', 0)}%, Latin {corpus_stats['script_distribution']['percentages'].get('Latin_ASCII', 0)}%")

    # 3. Partition Corpus into 80% Train, 10% Validation, 10% Test
    print("\n[Phase 2] Partitioning Corpus into 80/10/10 Deterministic Disjoint Splits...")
    docs_by_cat: Dict[str, List[Any]] = {}
    for d in valid_docs:
        docs_by_cat.setdefault(d.category, []).append(d)

    splits = partition_corpus(docs_by_cat, train_ratio=0.80, val_ratio=0.10, test_ratio=0.10, seed=42)

    train_lines: List[str] = [d.text for cat_docs in splits["train"].values() for d in cat_docs]
    val_docs = [d for cat_docs in splits["validation"].values() for d in cat_docs]
    test_docs = [d for cat_docs in splits["test"].values() for d in cat_docs]

    print(f"  Train: {len(train_lines)} lines | Val: {len(val_docs)} lines | Test: {len(test_docs)} lines")

    # 4. Vocabulary Size Candidates Benchmark (2048, 4096, 8192, 16384)
    print("\n[Phase 5, 8, 9] Training and Benchmarking Vocabulary Candidates...")
    vocab_experiments: Dict[str, Any] = {}
    tokenizers_by_v: Dict[int, BPETokenizer] = {}
    trainer = BPETrainer(min_frequency=1, tie_breaking_rule="(-frequency, pair[0], pair[1])")

    for v in CANDIDATE_VOCAB_SIZES:
        exp_dir = EXPERIMENTS_DIR / f"vocab_{v}"
        exp_dir.mkdir(parents=True, exist_ok=True)

        t0 = time.perf_counter()
        merges, vocab_map, train_stats = trainer.train(train_lines, target_vocab_size=v, verbose=False)
        train_duration = time.perf_counter() - t0
        tok = BPETokenizer(merges=merges, vocab=vocab_map)
        tokenizers_by_v[v] = tok

        # Save experiment artifact
        save_tokenizer_artifacts(tok, exp_dir, metadata={"requested_v": v, "train_stats": train_stats})

        # Evaluate on unseen Validation Split
        from chakrview.tokenizer.corpus import CorpusItem
        val_items = [CorpusItem(d.text, d.category, d.source_file, d.line_number) for d in val_docs]
        test_items = [CorpusItem(d.text, d.category, d.source_file, d.line_number) for d in test_docs]

        val_metrics = evaluate_tokenizer_on_corpus(tok, val_items, latency_repetitions=10)
        test_metrics = evaluate_tokenizer_on_corpus(tok, test_items, latency_repetitions=10)

        # Latency distribution on validation set
        val_texts = [d.text for d in val_docs]
        lat_stats = measure_tokenizer_latency(tok, val_texts, repetitions=20, warmup=3)

        # Memory breakdown and model scaling
        mem_stats = measure_tokenizer_memory(tok, d_model=192)

        # Vocabulary bias analysis
        bias_stats = analyze_vocabulary_bias(tok)

        # Extract top 5 and bottom 5 merges safely
        sorted_merges = sorted(merges.items(), key=lambda x: x[1])

        def format_token_label(tid: int) -> str:
            b = tok.vocab.get(tid, b"")
            try:
                s = b.decode("utf-8")
                return repr(s)
            except UnicodeDecodeError:
                return repr(b)

        top_merges = [f"{format_token_label(m[0][0])} + {format_token_label(m[0][1])} -> ID {m[1]} ({format_token_label(m[1])})" for m in sorted_merges[:5]]
        bottom_merges = [f"{format_token_label(m[0][0])} + {format_token_label(m[0][1])} -> ID {m[1]} ({format_token_label(m[1])})" for m in sorted_merges[-5:]]

        vocab_experiments[f"V{v}"] = {
            "requested_vocab_size": v,
            "actual_vocab_size": train_stats["actual_vocab_size"],
            "merges_performed": train_stats["merges_performed"],
            "exhausted_merges": (train_stats["actual_vocab_size"] < v),
            "training_time_seconds": round(train_duration, 4),
            "sample_top_merges": top_merges,
            "sample_bottom_merges": bottom_merges,
            "validation_metrics": val_metrics,
            "test_metrics": test_metrics,
            "latency": {
                "encode_mean_us": lat_stats["encode"].mean_us,
                "encode_median_us": lat_stats["encode"].median_us,
                "encode_p95_us": lat_stats["encode"].p95_us,
                "encode_min_us": lat_stats["encode"].min_us,
                "encode_max_us": lat_stats["encode"].max_us,
                "decode_mean_us": lat_stats["decode"].mean_us,
                "decode_median_us": lat_stats["decode"].median_us,
                "decode_p95_us": lat_stats["decode"].p95_us,
                "decode_min_us": lat_stats["decode"].min_us,
                "decode_max_us": lat_stats["decode"].max_us,
            },
            "memory": {
                "vocab_memory_bytes": mem_stats.vocab_memory_bytes,
                "merge_table_bytes": mem_stats.merge_table_bytes,
                "total_static_memory_kb": mem_stats.total_static_memory_kb,
                "embedding_params": mem_stats.embedding_params,
                "embedding_bytes_fp16": mem_stats.embedding_bytes_fp16,
                "embedding_param_percentage": mem_stats.embedding_param_percentage,
            },
            "vocabulary_bias": bias_stats,
        }

        print(f"  V={v} -> Actual: {train_stats['actual_vocab_size']} merges | Val Comp: {val_metrics['compression_ratio']:.4f} b/tok | Hindi: {val_metrics['category_metrics']['hindi']['tokens_per_word']:.2f} tok/w | Static RAM: {mem_stats.total_static_memory_kb:.1f} KB | Model Params: {mem_stats.embedding_param_percentage}%")

    # 5. Numeric Tokenization Experiment
    print("\n[Phase 6] Running Numeric Tokenization Strategies (A vs B vs C)...")
    base_tok = tokenizers_by_v[4096]
    test_numeric_suite = [
        "123456789",
        "2026",
        "13.56",
        "₹50000",
        "99.99%",
        "10:45:32",
        "192.168.1.1",
        "v0.1.4",
        "3.1415926535",
        "123 + 456 = 579",
        "9999 * 8888 = 88871112",
        "-98765",
        "1.23e-10",
        "2026-09-26",
    ]
    numeric_results = run_numeric_experiment(base_tok, test_numeric_suite)
    tot_a = sum(res["strategy_A_individual_digits"]["tokens"] for res in numeric_results.values())
    tot_b = sum(res["strategy_B_two_digit_chunks"]["tokens"] for res in numeric_results.values())
    tot_c = sum(res["strategy_C_normal_bpe"]["tokens"] for res in numeric_results.values())

    print(f"  Numeric Token Sums across test suite: A={tot_a} | B={tot_b} | C={tot_c}")
    print(f"  Candidate A expands sequence by {tot_a / tot_c:.2f}x; Candidate B expands by {tot_b / tot_c:.2f}x")

    # 6. Indic & Devanagari Validation
    print("\n[Phase 7] Running Indic & Devanagari Stress Validation...")
    indic_results: Dict[str, Any] = {}
    for cat, items in UNICODE_STRESS_ITEMS.items():
        all_passed = True
        tok_counts = []
        for text in items:
            tokens = base_tok.encode(text)
            decoded = base_tok.decode(tokens)
            if decoded != text:
                all_passed = False
            tok_counts.append(len(tokens))
        indic_results[cat] = {
            "items_tested": len(items),
            "lossless_verified": all_passed,
            "avg_tokens_per_item": round(sum(tok_counts) / max(1, len(tok_counts)), 2),
        }
        print(f"  Indic Category '{cat}': {len(items)} items -> Lossless: {all_passed}")

    # 7. Raw Byte & Adversarial Octet Tests
    print("\n[Adversarial] Running Raw Octet and Binary Fallback Tests...")
    raw_cases = get_raw_byte_test_cases(seed=42)
    raw_passed = True
    for case in raw_cases:
        tokens = base_tok.encode_bytes(case["bytes"])
        rec = base_tok.decode_bytes(tokens)
        if rec != case["bytes"]:
            raw_passed = False
    print(f"  Arbitrary raw octet test suite ({len(raw_cases)} cases): Lossless: {raw_passed}")

    # 8. Grapheme-Aware Pre-tokenization Experiment
    print("\n[Grapheme] Running Grapheme Pre-tokenization Variant A vs Variant B...")
    grapheme_comp = run_grapheme_experiment(
        corpus_lines=train_lines,
        test_lines=[d.text for d in val_docs],
        vocab_size=2048,
    )
    print(f"  Variant A (Raw Byte BPE): {grapheme_comp['variant_A_raw_byte_bpe']['tokens_per_word']} tok/word, {grapheme_comp['variant_A_raw_byte_bpe']['encode_latency_us']:.1f} us/doc")
    print(f"  Variant B (Grapheme BPE): {grapheme_comp['variant_B_grapheme_aware_bpe']['tokens_per_word']} tok/word, {grapheme_comp['variant_B_grapheme_aware_bpe']['encode_latency_us']:.1f} us/doc")

    # 9. Compile Complete Benchmark Report
    full_report: Dict[str, Any] = {
        "execution_metadata": {
            "title": "ChakrView Step 3 Tokenizer & Corpus Empirical Benchmark",
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "python_version": platform.python_version(),
            "os": f"{platform.system()} {platform.release()}",
            "machine": platform.machine(),
            "processor": platform.processor(),
            "seed": 42,
        },
        "corpus_statistics": corpus_stats,
        "corpus_validation": {
            "total_documents": len(all_raw_docs),
            "valid_documents": len(valid_docs),
            "discarded_documents": val_report["discarded_documents"],
            "duplicate_documents": val_report["duplicate_documents"],
            "splits": {
                "train_lines": len(train_lines),
                "val_lines": len(val_docs),
                "test_lines": len(test_docs),
            },
        },
        "vocabulary_experiments": vocab_experiments,
        "numeric_experiment": {
            "candidate_sums": {"tokens_A": tot_a, "tokens_B": tot_b, "tokens_C": tot_c},
            "expansion_ratios": {"A_vs_C": round(tot_a / tot_c, 2), "B_vs_C": round(tot_b / tot_c, 2)},
            "items": numeric_results,
        },
        "indic_validation": indic_results,
        "raw_byte_validation": {"cases_tested": len(raw_cases), "lossless": raw_passed},
        "grapheme_experiment": grapheme_comp,
    }

    report_path = STATS_DIR / "step3_tokenizer_benchmark.json"
    report_path.write_text(json.dumps(full_report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[OK] Step 3 Benchmark JSON report written to {report_path}")

    return full_report


if __name__ == "__main__":
    run_comprehensive_benchmark()
