"""
ChakrView Tokenizer Tests: Strict Determinism (Phase F).

Verifies that:
1. encode(text) == encode(text) identically across repeated invocations.
2. decode(tokens) == decode(tokens) identically across repeated invocations.
3. Training on identical corpora produces identical merge tables and token IDs.
"""

from chakrview.tokenizer.bpe import train_toy_bpe
from chakrview.tokenizer.tokenizer import BPETokenizer


def test_repeated_encoding_determinism():
    """Verify encode(text) == encode(text) across 50 iterations."""
    sample_text = (
        "ChakrView v0.1: स्वदेशी एआई रिसर्च प्रोटोकॉल.\n"
        "Testing determinism across multiple iterations with symbols: ∀x ∈ A, ₹50000, 🚀."
    )
    tokenizer = BPETokenizer()
    tokenizer.train_toy(sample_text, max_merges=10, min_frequency=2)

    baseline_tokens = tokenizer.encode(sample_text)

    for i in range(50):
        tokens = tokenizer.encode(sample_text)
        assert tokens == baseline_tokens, f"Non-deterministic encoding at iteration {i}"


def test_repeated_decoding_determinism():
    """Verify decode(tokens) == decode(tokens) across 50 iterations."""
    sample_text = "चक्रव्यूह टोकनाइज़र रिसर्च प्रोटोटाइप"
    tokenizer = BPETokenizer()
    tokenizer.train_toy(sample_text, max_merges=5, min_frequency=2)

    tokens = tokenizer.encode(sample_text)
    baseline_decoded = tokenizer.decode(tokens)

    for i in range(50):
        decoded = tokenizer.decode(tokens)
        assert decoded == baseline_decoded, f"Non-deterministic decoding at iteration {i}"


def test_merge_training_determinism():
    """Verify that independent training runs on the same data yield identical merge tables."""
    data = (
        b"abracadabra abracadabra abracadabra "
        b"\xe0\xa4\x95\xe0\xa4\xbe \xe0\xa4\x95\xe0\xa4\xbe \xe0\xa4\x95\xe0\xa4\xbe"
    )

    merges1, vocab1 = train_toy_bpe(data, max_merges=8, min_frequency=2)
    merges2, vocab2 = train_toy_bpe(data, max_merges=8, min_frequency=2)

    assert merges1 == merges2
    assert list(merges1.keys()) == list(merges2.keys())
    assert list(merges1.values()) == list(merges2.values())
    assert vocab1 == vocab2
