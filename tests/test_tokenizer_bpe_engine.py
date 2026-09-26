"""
ChakrView Tokenizer Tests: Minimal BPE Merge Engine (Phase E).

Verifies the deterministic BPE training, merge application, encoding,
and exact lossless decoding with learned multi-byte merges.
"""

from chakrview.tokenizer.bpe import (
    apply_merge,
    count_pairs,
    create_base_vocab,
    encode_with_merges,
    select_best_pair,
    train_toy_bpe,
)
from chakrview.tokenizer.bytes import byte_seq_to_token_ids, byte_to_token_id
from chakrview.tokenizer.decoder import decode_tokens
from chakrview.tokenizer.encoder import encode_text
from chakrview.tokenizer.tokenizer import BPETokenizer


def test_base_vocab_construction():
    """Verify base vocabulary contains exactly 256 byte primitives with single-byte values."""
    vocab = create_base_vocab()
    assert len(vocab) == 256
    for b in range(256):
        tid = byte_to_token_id(b)
        assert tid in vocab
        assert vocab[tid] == bytes([b])


def test_pair_frequency_counting():
    """Verify adjacent pair frequency counting."""
    tokens = [10, 20, 10, 20, 30, 10, 20]
    counts = count_pairs(tokens)
    assert counts[(10, 20)] == 3
    assert counts[(20, 10)] == 1
    assert counts[(20, 30)] == 1
    assert counts[(30, 10)] == 1


def test_deterministic_tie_breaking():
    """
    Verify tie-breaking rule:
    When frequencies are tied, select pair with lowest token IDs:
    (-freq, pair[0], pair[1])
    """
    # Two pairs both have frequency 2
    counts = {
        (50, 60): 2,
        (40, 70): 2,
    }
    # (40, 70) should be selected because 40 < 50
    best = select_best_pair(counts)
    assert best == (40, 70)

    # When first element is same, break tie on second element
    counts = {
        (40, 80): 2,
        (40, 70): 2,
    }
    best = select_best_pair(counts)
    assert best == (40, 70)


def test_apply_merge_single_step():
    """Verify single-pass non-overlapping merge replacement."""
    tokens = [10, 20, 30, 10, 20, 40]
    merged = apply_merge(tokens, pair=(10, 20), new_token_id=500)
    assert merged == [500, 30, 500, 40]

    # Non-overlapping behavior: [10, 10, 10] merging (10, 10) -> [500, 10]
    overlapping = [10, 10, 10]
    merged_ov = apply_merge(overlapping, pair=(10, 10), new_token_id=500)
    assert merged_ov == [500, 10]


def test_toy_bpe_training_and_compression():
    """
    Phase E: Train toy BPE on 'hello hello hello world hello'
    Verify:
      1. Merges are formed.
      2. Token count decreases.
      3. Exact reconstruction holds with merges.
    """
    corpus = "hello hello hello world hello"
    corpus_bytes = corpus.encode("utf-8")
    initial_tokens = byte_seq_to_token_ids(corpus_bytes)
    initial_len = len(initial_tokens)

    merges, vocab = train_toy_bpe(corpus_bytes, max_merges=5, min_frequency=2)

    assert len(merges) > 0
    # First token ID assigned for merge is 259
    assert 259 in vocab
    assert len(vocab) == 256 + len(merges)

    # Encode with learned merges
    compressed_tokens = encode_with_merges(initial_tokens, merges)
    assert len(compressed_tokens) < initial_len

    # Decode with merges and verify 100% exact text
    reconstructed = decode_tokens(compressed_tokens, vocab=vocab)
    assert reconstructed == corpus


def test_bpe_tokenizer_with_devanagari_merges():
    """
    Verify toy BPE merges on repetitive Hindi text:
    'नमस्ते नमस्ते नमस्ते'
    Multi-byte UTF-8 sequences must merge cleanly and decode losslessly.
    """
    hindi_text = "नमस्ते नमस्ते नमस्ते नमस्ते"
    tokenizer = BPETokenizer()

    # Pre-merge token count (each Hindi char is 3 UTF-8 bytes)
    pre_tokens = tokenizer.encode(hindi_text)
    pre_len = len(pre_tokens)

    # Train 5 toy merges
    tokenizer.train_toy(hindi_text, max_merges=5, min_frequency=2)
    assert tokenizer.num_merges > 0

    post_tokens = tokenizer.encode(hindi_text)
    post_len = len(post_tokens)

    # Token count must be strictly compressed
    assert post_len < pre_len

    # Exact lossless reconstruction must remain 100% true
    reconstructed = tokenizer.decode(post_tokens)
    assert reconstructed == hindi_text
