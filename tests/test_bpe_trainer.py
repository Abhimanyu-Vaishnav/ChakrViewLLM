"""
ChakrView Tests: BPE Trainer & Experiment Suite.

Tests:
- BPE trainer determinism across independent runs
- Vocabulary-size correctness and limits
- Serialization and deserialization of experiment artifacts
- Tokenizer round-trip after applying trained merges
- 3-way numeric strategy verification
"""

import tempfile
from pathlib import Path

from chakrview.tokenizer import (
    BPETrainer,
    BPETokenizer,
    load_experiment_artifacts,
    save_experiment_artifacts,
)
from chakrview.tokenizer.benchmark import (
    encode_candidate_digits,
    evaluate_tokenizer_on_corpus,
    run_numeric_experiment,
)
from chakrview.tokenizer_corpus.loader import CorpusItem


SAMPLE_CORPUS = [
    "ChakrView is an indigenous neural model designed for edge efficiency.",
    "नमस्ते! चक्रव्यूह एक स्वदेशी एआई अनुसंधान परियोजना है।",
    "def add(a: int, b: int) -> int:\n    return a + b",
    "₹50000 balance at 7.5% interest rate. Date: 2026-09-26.",
    "∀x ∈ ℝ: x² ≥ 0 and ॐ नमः शिवाय.",
]


def test_bpe_trainer_determinism():
    """Verify that training twice on the same corpus produces identical merges and vocab."""
    trainer = BPETrainer(min_frequency=1)

    merges1, vocab1, stats1 = trainer.train(SAMPLE_CORPUS, target_vocab_size=300)
    merges2, vocab2, stats2 = trainer.train(SAMPLE_CORPUS, target_vocab_size=300)

    assert merges1 == merges2
    assert vocab1 == vocab2
    assert stats1["actual_vocab_size"] == stats2["actual_vocab_size"]
    assert stats1["merges_performed"] == stats2["merges_performed"]


def test_vocabulary_size_correctness():
    """Verify trainer respects target vocabulary size and reports exact merge count."""
    trainer = BPETrainer(min_frequency=1)
    target_v = 300
    merges, vocab, stats = trainer.train(SAMPLE_CORPUS, target_vocab_size=target_v)

    # Base vocab is 259 (3 special + 256 bytes)
    expected_merges = target_v - 259
    assert len(merges) <= expected_merges
    assert stats["actual_vocab_size"] == 259 + len(merges)
    assert len(vocab) == 256 + len(merges)


def test_artifact_serialization_round_trip():
    """Verify that saving and reloading an experiment produces an identical BPETokenizer."""
    trainer = BPETrainer(min_frequency=1)
    merges, vocab, train_stats = trainer.train(SAMPLE_CORPUS, target_vocab_size=280)

    with tempfile.TemporaryDirectory() as tmpdir:
        exp_dir = Path(tmpdir) / "v280"
        save_experiment_artifacts(
            output_dir=exp_dir,
            target_vocab_size=280,
            merges=merges,
            vocab=vocab,
            train_stats=train_stats,
        )

        assert (exp_dir / "merges.json").exists()
        assert (exp_dir / "vocab.json").exists()
        assert (exp_dir / "train_stats.json").exists()
        assert (exp_dir / "metadata.json").exists()

        loaded_tok = load_experiment_artifacts(exp_dir)

        assert loaded_tok.merges == merges
        assert loaded_tok.vocab == vocab

        # Verify encoding and decoding match between original and reloaded
        test_text = "नमस्ते ChakrView!"
        original_tok = BPETokenizer(merges=merges, vocab=vocab)
        tokens_orig = original_tok.encode(test_text)
        tokens_loaded = loaded_tok.encode(test_text)

        assert tokens_orig == tokens_loaded
        assert loaded_tok.decode(tokens_loaded) == test_text


def test_tokenizer_round_trip_after_trained_merges():
    """Verify that a tokenizer with trained merges strictly preserves lossless reconstruction."""
    trainer = BPETrainer(min_frequency=1)
    merges, vocab, _ = trainer.train(SAMPLE_CORPUS, target_vocab_size=350)
    tok = BPETokenizer(merges=merges, vocab=vocab)

    for text in SAMPLE_CORPUS:
        tokens = tok.encode(text)
        decoded = tok.decode(tokens)
        assert decoded == text, f"Failed round-trip on: {text!r}"


def test_numeric_experiment_lossless():
    """Verify that numeric strategies A, B, and C all maintain 100% lossless reconstruction."""
    trainer = BPETrainer(min_frequency=1)
    merges, vocab, _ = trainer.train(SAMPLE_CORPUS, target_vocab_size=300)
    tok = BPETokenizer(merges=merges, vocab=vocab)

    test_strings = [
        "1234567890",
        "20260926",
        "₹50000",
        "13.14159",
        "13900H",
        "2026-09-26",
        "12:45:30",
    ]

    for s in test_strings:
        tok_a, dec_a = encode_candidate_digits(s, tok, "A")
        tok_b, dec_b = encode_candidate_digits(s, tok, "B")
        tok_c, dec_c = encode_candidate_digits(s, tok, "C")

        assert dec_a == s
        assert dec_b == s
        assert dec_c == s
        # Individual digits length must equal digit count
        assert len(tok_a) >= len(tok_b)
