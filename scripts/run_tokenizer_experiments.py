"""
ChakrView Step 2.3: Vocabulary Experiment & Benchmark Script.

Trains candidate BPE vocabularies (2048, 4096, 8192, 16384) on the research corpus
and benchmarks each candidate under identical conditions.
"""

import json
import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from chakrview.tokenizer_corpus import (
    load_corpus,
    validate_corpus,
    deterministic_train_val_split,
    compute_corpus_statistics,
    format_statistics_report,
)
from chakrview.tokenizer import (
    BPETrainer,
    BPETokenizer,
    save_experiment_artifacts,
)
from chakrview.tokenizer.benchmark import (
    evaluate_tokenizer_on_corpus,
    run_numeric_experiment,
)

NUMERIC_TEST_SAMPLES = [
    "1234567890",
    "20260926",
    "₹50000",
    "13.14159",
    "13900H",
    "2026-09-26",
    "12:45:30",
    "def calc_total(items: int = 1500) -> float:\n    return items * 19.99",
    "\\sigma(z)_i = \\frac{e^{z_i}}{\\sum_{j=1}^{10} e^{z_j}}",
]


def main():
    print("=" * 80)
    print("ChakrView Step 2.3 — Corpus Engineering & BPE Vocabulary Experiment")
    print("=" * 80)

    corpus_dir = root_dir / "data" / "tokenizer_corpus"
    experiments_dir = root_dir / "data" / "tokenizer_experiments"
    experiments_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load and Validate Corpus
    print(f"\n[1/5] Loading and validating corpus from {corpus_dir}...")
    corpus = load_corpus(corpus_dir)
    validate_corpus(corpus)
    stats = compute_corpus_statistics(corpus)
    print(f"Loaded {stats['total_documents']} documents ({stats['total_bytes']:,} UTF-8 bytes)")

    # 2. Partition Train/Validation split
    print("\n[2/5] Partitioning corpus with deterministic 80/20 train/val split...")
    train_items, val_items = deterministic_train_val_split(corpus, val_ratio=0.20)
    print(f"Train split: {len(train_items)} items ({sum(len(x.text.encode('utf-8')) for x in train_items):,} bytes)")
    print(f"Val split:   {len(val_items)} items ({sum(len(x.text.encode('utf-8')) for x in val_items):,} bytes)")

    train_lines = [item.text for item in train_items]

    # 3. Train Candidate Vocabularies
    candidate_vocab_sizes = [2048, 4096, 8192, 16384]
    eval_results = {}
    numeric_results = {}

    trainer = BPETrainer(min_frequency=1)

    for target_v in candidate_vocab_sizes:
        print(f"\n[3/5] Training BPE candidate: V = {target_v}...")
        merges, vocab, train_stats = trainer.train(
            train_lines,
            target_vocab_size=target_v,
            verbose=False,
        )

        actual_v = train_stats["actual_vocab_size"]
        num_merges = train_stats["merges_performed"]
        t_time = train_stats["training_time_seconds"]
        print(f"  Completed {num_merges} merges in {t_time:.2f}s (Actual V: {actual_v})")

        # Save artifacts
        exp_dir = experiments_dir / f"v{target_v}"
        save_experiment_artifacts(
            output_dir=exp_dir,
            target_vocab_size=target_v,
            merges=merges,
            vocab=vocab,
            train_stats=train_stats,
            metadata={
                "train_items_count": len(train_items),
                "val_items_count": len(val_items),
            },
        )
        print(f"  Artifacts saved to {exp_dir}")

        # 4. Evaluate on Validation Split
        tokenizer = BPETokenizer(merges=merges, vocab=vocab)
        print(f"  Evaluating candidate V = {target_v} on {len(val_items)} validation items...")
        eval_metrics = evaluate_tokenizer_on_corpus(tokenizer, val_items)
        eval_results[target_v] = eval_metrics

        # Save eval stats
        with (exp_dir / "eval_stats.json").open("w", encoding="utf-8") as f:
            json.dump(eval_metrics, f, indent=2)

        is_lossless = eval_metrics["reconstruction_lossless"]
        cr = eval_metrics["compression_ratio"]
        tpw = eval_metrics["tokens_per_word"]
        print(f"  Result: Lossless={is_lossless}, Compression Ratio={cr:.2f} bytes/tok, Tokens/Word={tpw:.2f}")

        # Run numeric experiment for 4096 and 8192
        if target_v in (4096, 8192):
            print(f"  Running 3-way numeric experiment on candidate V = {target_v}...")
            num_res = run_numeric_experiment(tokenizer, NUMERIC_TEST_SAMPLES)
            numeric_results[target_v] = num_res
            with (exp_dir / "numeric_experiment.json").open("w", encoding="utf-8") as f:
                json.dump(num_res, f, indent=2)

    # 5. Print Consolidated Results Table
    print("\n" + "=" * 80)
    print("CONSOLIDATED BPE VOCABULARY BENCHMARK RESULTS")
    print("=" * 80)
    print(
        f"{'Candidate':<8} | {'Merges':<7} | {'Tokens/Char':<11} | {'Tokens/Word':<11} | "
        f"{'Bytes/Token':<11} | {'Lossless':<8} | {'Enc (µs)':<9} | {'Dec (µs)':<9} | {'Vocab RAM':<10}"
    )
    print("-" * 96)
    for v in candidate_vocab_sizes:
        r = eval_results[v]
        v_name = f"V={v}"
        merges_str = str(r["total_tokens"])  # token count on val set
        tpc = f"{r['tokens_per_char']:.4f}"
        tpw = f"{r['tokens_per_word']:.2f}"
        bpt = f"{r['bytes_per_token']:.2f}"
        lossless = "PASS" if r["reconstruction_lossless"] else "FAIL"
        enc = f"{r['encode_latency_us_per_doc']:.1f}"
        dec = f"{r['decode_latency_us_per_doc']:.1f}"
        ram = f"{r['vocab_memory_kb']:.1f} KB"
        print(
            f"{v_name:<8} | {r['total_tokens']:<7} | {tpc:<11} | {tpw:<11} | "
            f"{bpt:<11} | {lossless:<8} | {enc:<9} | {dec:<9} | {ram:<10}"
        )
    print("-" * 96)

    # Save consolidated summary
    consolidated_summary = {
        "candidate_results": eval_results,
        "numeric_experiment": numeric_results,
    }
    with (experiments_dir / "consolidated_benchmark.json").open("w", encoding="utf-8") as f:
        json.dump(consolidated_summary, f, indent=2)
    print(f"\nConsolidated benchmark saved to {experiments_dir / 'consolidated_benchmark.json'}")
    print("Execution complete.")


if __name__ == "__main__":
    main()
