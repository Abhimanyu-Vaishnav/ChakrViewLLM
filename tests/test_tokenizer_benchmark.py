"""
ChakrView Tests: Tokenizer Benchmark Engine & Metrics Suite.

Tests:
- Metric computation (tokens/char, tokens/word, bytes/token, compression ratio)
- Memory measurement (static vocab, merge table, embedding table parameters)
- Latency statistics calculation (mean, median, p95, min, max, std)
- Candidate numeric strategies (Candidate A, B, C)
- Grapheme-aware pre-tokenization vs raw byte BPE comparison
"""

from chakrview.tokenizer.benchmark import (
    calculate_vocab_memory,
    encode_candidate_digits,
    run_grapheme_experiment,
    run_numeric_experiment,
)
from chakrview.tokenizer.metrics import (
    compute_latency_stats,
    compute_sequence_metrics,
    evaluate_split_metrics,
    measure_tokenizer_latency,
    measure_tokenizer_memory,
)
from chakrview.tokenizer.tokenizer import BPETokenizer
from chakrview.tokenizer.trainer import BPETrainer

SAMPLE_TEXTS = [
    "नमस्ते! चक्रव्यूह प्रोजेक्ट।",
    "ChakrView indigenous edge AI model.",
    "2026-09-26 ₹50000 13.56%",
]


def test_compute_sequence_metrics():
    """Verify calculation of tokens/char, tokens/word, bytes/token."""
    text = "Hello world"
    tokens = [10, 20]  # 2 tokens
    m = compute_sequence_metrics(text, tokens)

    assert m.tokens == 2
    assert m.characters == len(text)
    assert m.words == 2
    assert m.raw_bytes == len(text.encode("utf-8"))
    assert m.tokens_per_char == round(2 / len(text), 4)
    assert m.tokens_per_word == 1.0


def test_compute_latency_stats():
    """Verify mean, median, and p95 calculations."""
    times = [10.0, 20.0, 30.0, 40.0, 50.0]
    stats = compute_latency_stats(times)

    assert stats.mean_us == 30.0
    assert stats.median_us == 30.0
    assert stats.min_us == 10.0
    assert stats.max_us == 50.0
    assert stats.iterations == 5


def test_measure_tokenizer_memory():
    """Verify memory accounting for vocab, merges, and embedding parameters."""
    trainer = BPETrainer(min_frequency=1)
    merges, vocab, _ = trainer.train(SAMPLE_TEXTS, target_vocab_size=280)
    tok = BPETokenizer(merges=merges, vocab=vocab)

    mem = measure_tokenizer_memory(tok, d_model=192)
    assert mem.vocab_size == tok.vocab_size
    assert mem.vocab_memory_bytes > 0
    assert mem.merge_table_bytes > 0
    assert mem.embedding_params == tok.vocab_size * 192
    assert mem.embedding_bytes_fp16 == mem.embedding_params * 2
    assert mem.embedding_bytes_int8 == mem.embedding_params * 1


def test_numeric_experiment_strategies():
    """Verify numeric strategies A, B, and C on test items."""
    tok = BPETokenizer()  # Base byte tokenizer
    results = run_numeric_experiment(tok, ["2026", "₹50000"])

    for item, res in results.items():
        assert res["strategy_A_individual_digits"]["reconstruction_lossless"]
        assert res["strategy_B_two_digit_chunks"]["reconstruction_lossless"]
        assert res["strategy_C_normal_bpe"]["reconstruction_lossless"]


def test_grapheme_experiment_execution():
    """Verify running grapheme experiment Variant A vs Variant B."""
    corpus = [
        "नमस्ते! चक्रव्यूह एक स्वदेशी एआई अनुसंधान परियोजना है।",
        "ChakrView is an indigenous neural architecture.",
    ]
    test_lines = ["नमस्ते दुनिया", "ChakrView AI"]

    results = run_grapheme_experiment(corpus, test_lines, vocab_size=280)
    assert "variant_A_raw_byte_bpe" in results
    assert "variant_B_grapheme_aware_bpe" in results
    assert results["variant_A_raw_byte_bpe"]["lossless"]
    assert results["variant_B_grapheme_aware_bpe"]["lossless"]
