"""
ChakrView Tests: Tokenizer-to-Neural-Core Interface Contract (Step 2.4).

Tests:
- Token ID range validation ([0, 4095] for V=4096)
- Special token ID contract (BOS=0, EOS=1, PAD=2)
- Output semantics for explicit test strings ("", " ", "\\n", "क", "ॐ", emojis, numbers, "<BOS>")
- Neural-core handoff test (token IDs -> [seq_len, 192] embedding tensor)
- Batch preparation contract ([batch, seq_len], attention_mask)
- Truncation and chunking runtime policies
- Reproducibility across identical training runs
- Memory accounting verification (static parameter storage only)
"""

from pathlib import Path
import pytest

from chakrview.tokenizer import (
    BOS_ID,
    EOS_ID,
    PAD_ID,
    D_MODEL,
    MAX_CONTEXT,
    PROVISIONAL_VOCAB_SIZE,
    BPETokenizer,
    BPETrainer,
    calculate_static_parameter_memory,
    chunk_tokens,
    fake_embedding_lookup,
    load_experiment_artifacts,
    prepare_batch,
    truncate_tokens,
    validate_token_ids,
)


@pytest.fixture
def v4096_tokenizer() -> BPETokenizer:
    exp_dir = Path(__file__).resolve().parent.parent / "data" / "tokenizer_experiments" / "v4096"
    assert exp_dir.is_dir(), f"v4096 experiment directory missing at: {exp_dir}"
    return load_experiment_artifacts(exp_dir)


def test_provisional_vocabulary_size(v4096_tokenizer: BPETokenizer):
    """Verify provisional vocabulary size is exactly 4096."""
    assert v4096_tokenizer.vocab_size == PROVISIONAL_VOCAB_SIZE
    assert PROVISIONAL_VOCAB_SIZE == 4096


def test_special_tokens_frozen_contract():
    """Verify frozen special token IDs."""
    assert BOS_ID == 0
    assert EOS_ID == 1
    assert PAD_ID == 2


def test_token_id_range_contract(v4096_tokenizer: BPETokenizer):
    """Verify that all token IDs produced by v4096 fall strictly in [0, 4095]."""
    test_inputs = [
        "Hello World",
        "नमस्ते चक्रव्यूह 2026",
        "def compute(): return 42",
        "∀x ∈ ℝ: x² ≥ 0",
        "D:\\Project\\ChakrView\\data",
        "भारत 🇮🇳 ❤️ 🚀",
        "1234567890 ₹50000",
    ]

    for text in test_inputs:
        tokens = v4096_tokenizer.encode(text, add_bos=True, add_eos=True)
        validate_token_ids(tokens, vocab_size=PROVISIONAL_VOCAB_SIZE)
        for tid in tokens:
            assert 0 <= tid < 4096, f"Token ID {tid} out of valid range [0, 4095]"


def test_token_id_range_validation_errors():
    """Verify validate_token_ids raises ValueError on out-of-range tokens."""
    validate_token_ids([0, 1, 2, 258, 4095], vocab_size=4096)

    with pytest.raises(ValueError):
        validate_token_ids([-1], vocab_size=4096)

    with pytest.raises(ValueError):
        validate_token_ids([4096], vocab_size=4096)

    with pytest.raises(TypeError):
        validate_token_ids(["0"], vocab_size=4096)  # type: ignore


def test_tokenizer_output_semantics(v4096_tokenizer: BPETokenizer):
    """
    Test Section 10 explicit strings:
    "", " ", "\\n", "क", "ॐ", emojis, "1234567890", "<BOS>", "<EOS>", "<PAD>"
    """
    # 1. Empty string
    tokens_empty = v4096_tokenizer.encode("")
    assert tokens_empty == []
    assert v4096_tokenizer.decode(tokens_empty) == ""

    # With BOS/EOS
    tokens_empty_boseos = v4096_tokenizer.encode("", add_bos=True, add_eos=True)
    assert tokens_empty_boseos == [BOS_ID, EOS_ID]
    assert v4096_tokenizer.decode(tokens_empty_boseos, skip_special_tokens=True) == ""

    # 2. Single space
    tokens_space = v4096_tokenizer.encode(" ")
    assert len(tokens_space) > 0
    assert v4096_tokenizer.decode(tokens_space) == " "

    # 3. Newline
    tokens_nl = v4096_tokenizer.encode("\n")
    assert len(tokens_nl) > 0
    assert v4096_tokenizer.decode(tokens_nl) == "\n"

    # 4. Devanagari "क"
    tokens_ka = v4096_tokenizer.encode("क")
    assert len(tokens_ka) > 0
    assert v4096_tokenizer.decode(tokens_ka) == "क"

    # 5. Sacred symbol "ॐ"
    tokens_om = v4096_tokenizer.encode("ॐ")
    assert len(tokens_om) > 0
    assert v4096_tokenizer.decode(tokens_om) == "ॐ"

    # 6. Emoji sequence "👨👩👧👦"
    tokens_emoji = v4096_tokenizer.encode("👨👩👧👦")
    assert len(tokens_emoji) > 0
    assert v4096_tokenizer.decode(tokens_emoji) == "👨👩👧👦"

    # 7. Numeric string "1234567890"
    tokens_num = v4096_tokenizer.encode("1234567890")
    assert len(tokens_num) > 0
    assert v4096_tokenizer.decode(tokens_num) == "1234567890"

    # 8. Literal special strings must NOT emit special token IDs
    for literal in ("<BOS>", "<EOS>", "<PAD>"):
        toks = v4096_tokenizer.encode(literal, add_bos=False, add_eos=False)
        assert BOS_ID not in toks
        assert EOS_ID not in toks
        assert PAD_ID not in toks
        assert v4096_tokenizer.decode(toks) == literal


def test_neural_core_handoff(v4096_tokenizer: BPETokenizer):
    """
    Section 13: Simulate text -> tokenizer -> token IDs -> embedding lookup -> [seq_len, d_model].
    Verify output shape and dimensional correctness for d_model = 192.
    """
    sample_text = "ChakrView indigenous neural architecture v0.1"
    tokens = v4096_tokenizer.encode(sample_text, add_bos=True, add_eos=True)
    seq_len = len(tokens)

    embeddings = fake_embedding_lookup(tokens, d_model=D_MODEL)

    # Verify tensor shape
    assert len(embeddings) == seq_len
    for vec in embeddings:
        assert len(vec) == D_MODEL
        assert all(isinstance(v, float) for v in vec)


def test_batch_contract():
    """Section 6: Verify batch creation with padding and attention mask."""
    seq1 = [10, 20, 30]
    seq2 = [40, 50]
    seq3 = [60]

    input_ids, attention_mask = prepare_batch([seq1, seq2, seq3], pad_id=PAD_ID)

    assert len(input_ids) == 3
    assert len(attention_mask) == 3

    # All padded to batch max length = 3
    assert input_ids[0] == [10, 20, 30]
    assert attention_mask[0] == [1, 1, 1]

    assert input_ids[1] == [40, 50, PAD_ID]
    assert attention_mask[1] == [1, 1, 0]

    assert input_ids[2] == [60, PAD_ID, PAD_ID]
    assert attention_mask[2] == [1, 0, 0]


def test_long_sequence_policy():
    """Section 5: Verify truncation and chunking runtime policies."""
    long_seq = list(range(600))  # Exceeds MAX_CONTEXT = 512

    # Truncation
    truncated = truncate_tokens(long_seq, max_length=512)
    assert len(truncated) == 512
    assert truncated == long_seq[:512]

    # Truncation with EOS
    truncated_eos = truncate_tokens(long_seq, max_length=512, add_eos_if_truncated=True)
    assert len(truncated_eos) == 512
    assert truncated_eos[-1] == EOS_ID
    assert truncated_eos[:-1] == long_seq[:511]

    # Chunking
    chunks = chunk_tokens(long_seq, chunk_size=512, overlap=64)
    assert len(chunks) == 2
    assert len(chunks[0]) == 512
    # Second chunk step: 512 - 64 = 448; start index 448 -> 600 - 448 = 152 items
    assert len(chunks[1]) == 152


def test_training_reproducibility():
    """Section 12: Verify that two training runs with identical configuration produce identical artifacts."""
    corpus = ["apple banana cherry", "banana cherry date", "cherry date elderberry"]
    trainer = BPETrainer(min_frequency=1)

    m1, v1, s1 = trainer.train(corpus, target_vocab_size=270)
    m2, v2, s2 = trainer.train(corpus, target_vocab_size=270)

    assert m1 == m2
    assert v1 == v2
    assert s1["actual_vocab_size"] == s2["actual_vocab_size"]

    tok1 = BPETokenizer(merges=m1, vocab=v1)
    tok2 = BPETokenizer(merges=m2, vocab=v2)

    sample = "apple cherry date"
    assert tok1.encode(sample) == tok2.encode(sample)


def test_memory_accounting():
    """Section 14: Verify static parameter memory accounting."""
    mem_report = calculate_static_parameter_memory(vocab_size=4096, d_model=192)

    assert mem_report["embedding_parameters"] == 786432
    assert mem_report["tied_unembedding_parameters"] == 0
    assert mem_report["total_embedding_parameters"] == 786432

    # Verify byte values
    assert mem_report["memory_fp32_bytes"] == 3145728  # 3.0 MB
    assert mem_report["memory_fp16_bytes"] == 1572864  # 1.5 MB
    assert mem_report["memory_int8_bytes"] == 786432   # 0.75 MB
    assert mem_report["memory_int4_bytes"] == 393216   # 0.375 MB

    assert mem_report["scope"] == "STATIC_PARAMETER_STORAGE_ONLY"
    assert "activations" in mem_report["exclusions"]
    assert "kv_cache" in mem_report["exclusions"]
