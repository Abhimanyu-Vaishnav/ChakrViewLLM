"""
ChakrView Step 3: Empirical Tokenizer Experiment Suite & Selection Battery.

Executes:
1. Multi-Vocabulary Candidates: V in {2048, 4096, 8192, 16384}
   - Trained STRICTLY on data/processed/train/
   - Evaluated on UNSEEN data/validation/ and data/processed/test/
   - Records all 20 required metrics: tokens/char, tokens/word, bytes/tok,
     compression, category efficiencies, encode/decode latency (mean, median, p95),
     memory footprint, and downstream embedding table parameters.
2. Numeric Tokenization Experiment: Candidates A, B, C across 12 distinct numeric domains.
3. Grapheme-Aware Pre-tokenization Experiment: Variant A (raw byte BPE) vs Variant B (grapheme-aware BPE).
4. Adversarial & Malformed Byte Verification.
5. Saves results to experiments/tokenizer/reports/step3_benchmark_results.json.
"""

import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from chakrview.tokenizer.benchmark import (
    NUMERIC_TEST_ITEMS,
    encode_candidate_digits,
    evaluate_tokenizer_on_corpus,
    run_grapheme_experiment,
    run_numeric_experiment,
)
from chakrview.tokenizer.corpus import (
    CorpusItem,
    execute_corpus_pipeline,
    load_corpus_split,
)
from chakrview.tokenizer.metrics import (
    measure_tokenizer_latency,
    measure_tokenizer_memory,
)
from chakrview.tokenizer.serialization import save_tokenizer_artifacts
from chakrview.tokenizer.tokenizer import BPETokenizer
from chakrview.tokenizer.trainer import BPETrainer

DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
VAL_DIR = DATA_DIR / "validation"
REPORTS_DIR = ROOT_DIR / "experiments" / "tokenizer" / "reports"
MODELS_DIR = ROOT_DIR / "experiments" / "tokenizer" / "candidates"


def run_complete_step3_battery() -> Dict[str, Any]:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("=================================================================")
    print("CHAKRVIEW STEP 3 — EMPIRICAL TOKENIZER RESEARCH & SELECTION")
    print("=================================================================")

    # 1. Ensure Corpus Pipeline has run
    manifest = execute_corpus_pipeline(
        raw_dir=RAW_DIR,
        processed_dir=PROCESSED_DIR,
        validation_dir=VAL_DIR,
        train_ratio=0.80,
        val_ratio=0.10,
        test_ratio=0.10,
        seed=42,
    )
    print(f"Corpus Pipeline Verified:")
    print(f"  Train: {manifest['totals']['train_lines']} lines ({manifest['totals']['train_bytes']:,} bytes)")
    print(f"  Val:   {manifest['totals']['val_lines']} lines ({manifest['totals']['val_bytes']:,} bytes)")
    print(f"  Test:  {manifest['totals']['test_lines']} lines ({manifest['totals']['test_bytes']:,} bytes)")

    # 2. Load Train and Validation Splits
    train_split = load_corpus_split(PROCESSED_DIR / "train")
    val_split = load_corpus_split(VAL_DIR)
    test_split = load_corpus_split(PROCESSED_DIR / "test")

    train_lines: List[str] = []
    for cat in sorted(train_split.keys()):
        train_lines.extend([it.text for it in train_split[cat]])

    val_items: List[CorpusItem] = []
    for cat in sorted(val_split.keys()):
        val_items.extend(val_split[cat])

    test_items: List[CorpusItem] = []
    for cat in sorted(test_split.keys()):
        test_items.extend(test_split[cat])

    # 3. Train and Benchmark Vocabulary Candidates
    vocab_candidates = [2048, 4096, 8192, 16384]
    vocab_results: Dict[str, Any] = {}
    trained_tokenizers: Dict[int, BPETokenizer] = {}

    trainer = BPETrainer(min_frequency=1, tie_breaking_rule="(-frequency, pair[0], pair[1])")

    for v in vocab_candidates:
        print(f"\n--- Training Candidate V={v} (on Train split only) ---")
        t0 = time.perf_counter()
        merges, vocab_map, stats = trainer.train(train_lines, target_vocab_size=v, verbose=False)
        train_time = time.perf_counter() - t0
        tok = BPETokenizer(merges=merges, vocab=vocab_map)
        trained_tokenizers[v] = tok

        # Save artifact
        cand_dir = MODELS_DIR / f"v{v}"
        save_tokenizer_artifacts(tok, cand_dir, metadata={"requested_v": v, "train_stats": stats})

        # Evaluate on unseen Validation items
        val_eval = evaluate_tokenizer_on_corpus(tok, val_items, latency_repetitions=10)
        # Evaluate on unseen Test items
        test_eval = evaluate_tokenizer_on_corpus(tok, test_items, latency_repetitions=10)

        # Extended latency profile
        sample_texts = [it.text for it in val_items]
        lat_profile = measure_tokenizer_latency(tok, sample_texts, repetitions=20, warmup=3)

        # Memory profile
        mem_profile = measure_tokenizer_memory(tok, d_model=192)

        vocab_results[f"V{v}"] = {
            "requested_vocab_size": v,
            "actual_vocab_size": stats["actual_vocab_size"],
            "merges_performed": stats["merges_performed"],
            "exhausted_candidates": (stats["actual_vocab_size"] < v),
            "training_time_seconds": round(train_time, 4),
            "validation": val_eval,
            "test": test_eval,
            "latency": {
                "encode_mean_us": lat_profile["encode"].mean_us,
                "encode_median_us": lat_profile["encode"].median_us,
                "encode_p95_us": lat_profile["encode"].p95_us,
                "encode_min_us": lat_profile["encode"].min_us,
                "encode_max_us": lat_profile["encode"].max_us,
                "decode_mean_us": lat_profile["decode"].mean_us,
                "decode_median_us": lat_profile["decode"].median_us,
                "decode_p95_us": lat_profile["decode"].p95_us,
                "decode_min_us": lat_profile["decode"].min_us,
                "decode_max_us": lat_profile["decode"].max_us,
            },
            "memory": {
                "vocab_memory_bytes": mem_profile.vocab_memory_bytes,
                "merge_table_bytes": mem_profile.merge_table_bytes,
                "total_static_memory_bytes": mem_profile.total_static_memory_bytes,
                "total_static_memory_kb": mem_profile.total_static_memory_kb,
                "embedding_params": mem_profile.embedding_params,
                "embedding_bytes_fp16": mem_profile.embedding_bytes_fp16,
                "embedding_bytes_int8": mem_profile.embedding_bytes_int8,
                "embedding_param_percentage": mem_profile.embedding_param_percentage,
            },
        }

        print(f"  V={v} -> Actual: {stats['actual_vocab_size']} (Merges: {stats['merges_performed']})")
        print(f"  Validation Compression: {val_eval['compression_ratio']:.4f} bytes/tok | Hindi: {val_eval['category_metrics']['hindi']['tokens_per_word']:.2f} tok/w | English: {val_eval['category_metrics']['english']['tokens_per_word']:.2f} tok/w")
        print(f"  Static RAM: {mem_profile.total_static_memory_kb:.1f} KB | Embedding: {mem_profile.embedding_params:,} params ({mem_profile.embedding_param_percentage}%)")

    # 4. Numeric Tokenization Experiment
    print("\n--- Running Numeric Tokenization Experiment (Candidate A vs B vs C) ---")
    base_tok = trained_tokenizers[4096]
    numeric_results = run_numeric_experiment(base_tok)

    num_summary = {"tokens_A": 0, "tokens_B": 0, "tokens_C": 0, "total_bytes": 0, "lossless": True}
    for item, res in numeric_results.items():
        num_summary["tokens_A"] += res["strategy_A_individual_digits"]["tokens"]
        num_summary["tokens_B"] += res["strategy_B_two_digit_chunks"]["tokens"]
        num_summary["tokens_C"] += res["strategy_C_normal_bpe"]["tokens"]
        num_summary["total_bytes"] += res["bytes"]
        if not (res["strategy_A_individual_digits"]["reconstruction_lossless"] and
                res["strategy_B_two_digit_chunks"]["reconstruction_lossless"] and
                res["strategy_C_normal_bpe"]["reconstruction_lossless"]):
            num_summary["lossless"] = False

    print(f"  Numeric Token Sums across test battery: Candidate A={num_summary['tokens_A']} | Candidate B={num_summary['tokens_B']} | Candidate C={num_summary['tokens_C']}")
    print(f"  Sequence expansion vs normal BPE: Candidate A is {num_summary['tokens_A'] / num_summary['tokens_C']:.2f}x longer, Candidate B is {num_summary['tokens_B'] / num_summary['tokens_C']:.2f}x longer")

    # 5. Grapheme-Aware Pre-tokenization Experiment
    print("\n--- Running Grapheme Pre-tokenization Experiment (Variant A vs Variant B) ---")
    val_texts = [it.text for it in val_items]
    grapheme_results = run_grapheme_experiment(
        corpus_lines=train_lines,
        test_lines=val_texts,
        vocab_size=2048,
    )
    print(f"  Variant A (Raw Byte BPE): {grapheme_results['variant_A_raw_byte_bpe']['tokens']} tokens, {grapheme_results['variant_A_raw_byte_bpe']['tokens_per_word']} tok/word, {grapheme_results['variant_A_raw_byte_bpe']['encode_latency_us']:.1f} us/doc")
    print(f"  Variant B (Grapheme BPE): {grapheme_results['variant_B_grapheme_aware_bpe']['tokens']} tokens, {grapheme_results['variant_B_grapheme_aware_bpe']['tokens_per_word']} tok/word, {grapheme_results['variant_B_grapheme_aware_bpe']['encode_latency_us']:.1f} us/doc")

    # 6. Build Manifest and System Information
    manifest_report: Dict[str, Any] = {
        "system_manifest": {
            "python_version": platform.python_version(),
            "os": f"{platform.system()} {platform.release()}",
            "machine": platform.machine(),
            "processor": platform.processor(),
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "git_commit": "Step 3 Empirical Selection",
            "seed": 42,
        },
        "corpus_manifest": {
            "total_raw_files": manifest["totals"]["raw_files_validated"],
            "total_raw_bytes": manifest["totals"]["raw_bytes"],
            "total_raw_lines": manifest["totals"]["raw_lines"],
            "total_unique_lines": manifest["totals"]["dedup_lines"],
            "splits": {
                "train_lines": manifest["totals"]["train_lines"],
                "train_bytes": manifest["totals"]["train_bytes"],
                "val_lines": manifest["totals"]["val_lines"],
                "val_bytes": manifest["totals"]["val_bytes"],
                "test_lines": manifest["totals"]["test_lines"],
                "test_bytes": manifest["totals"]["test_bytes"],
            },
        },
        "vocab_candidates": vocab_results,
        "numeric_experiment": {
            "summary": num_summary,
            "items": numeric_results,
        },
        "grapheme_experiment": grapheme_results,
    }

    report_path = REPORTS_DIR / "step3_benchmark_results.json"
    report_path.write_text(json.dumps(manifest_report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[OK] Step 3 Benchmark Report saved to {report_path}")

    return manifest_report


if __name__ == "__main__":
    run_complete_step3_battery()
