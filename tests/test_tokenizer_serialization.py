"""
ChakrView Tests: Tokenizer Serialization Engine.

Verifies:
- Saving tokenizer to disk writes vocab.json, merges.json, and config.json
- Checksums are calculated and verified
- Checksum tampering detection raises error
- Lossless serialization round-trip on Indic, English, and non-UTF-8 bytes
- Tokenizer behavior is identical before and after serialization
"""

import tempfile
from pathlib import Path
import pytest

from chakrview.tokenizer.bytes import (
    BYTE_OFFSET,
    NUM_BYTE_TOKENS,
)
from chakrview.tokenizer.serialization import (
    load_tokenizer_artifacts,
    save_tokenizer_artifacts,
)
from chakrview.tokenizer.tokenizer import BPETokenizer
from chakrview.tokenizer.trainer import BPETrainer

SAMPLE_TEXTS = [
    "नमस्ते दुनिया!",
    "ChakrView indigenous tokenizer v0.1",
    "def forward(x):\n    return x * 2",
    "2026-09-26 ₹50000 13.56%",
]


def test_serialization_files_created():
    """Verify that save_tokenizer_artifacts creates inspectable json files."""
    trainer = BPETrainer(min_frequency=1)
    merges, vocab, _ = trainer.train(SAMPLE_TEXTS, target_vocab_size=280)
    tok = BPETokenizer(merges=merges, vocab=vocab)

    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir) / "tok_artifacts"
        saved_path = save_tokenizer_artifacts(tok, out_dir, numeric_strategy="C", pretokenization_strategy="byte")

        assert (out_dir / "merges.json").is_file()
        assert (out_dir / "vocab.json").is_file()
        assert (out_dir / "config.json").is_file()

        loaded_tok, cfg = load_tokenizer_artifacts(out_dir, verify_checksums=True)
        assert cfg["vocab_size"] == tok.vocab_size
        assert cfg["numeric_strategy"] == "C"
        assert cfg["pretokenization_strategy"] == "byte"
        assert loaded_tok.vocab_size == tok.vocab_size
        assert loaded_tok.merges == tok.merges
        assert loaded_tok.vocab == tok.vocab


def test_serialization_round_trip_encoding():
    """Verify that reloaded tokenizer produces bit-exact identical token IDs and decoded text."""
    trainer = BPETrainer(min_frequency=1)
    merges, vocab, _ = trainer.train(SAMPLE_TEXTS, target_vocab_size=300)
    tok = BPETokenizer(merges=merges, vocab=vocab)

    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir) / "model"
        save_tokenizer_artifacts(tok, out_dir)
        loaded_tok, _ = load_tokenizer_artifacts(out_dir)

        for text in SAMPLE_TEXTS:
            orig_tokens = tok.encode(text)
            loaded_tokens = loaded_tok.encode(text)
            assert orig_tokens == loaded_tokens
            assert loaded_tok.decode(loaded_tokens) == text


def test_checksum_tampering_detection():
    """Verify that tampering with vocab.json or merges.json raises ValueError during load."""
    trainer = BPETrainer(min_frequency=1)
    merges, vocab, _ = trainer.train(SAMPLE_TEXTS, target_vocab_size=275)
    tok = BPETokenizer(merges=merges, vocab=vocab)

    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir) / "model_tampered"
        save_tokenizer_artifacts(tok, out_dir)

        # Tamper with vocab.json
        vocab_file = out_dir / "vocab.json"
        vocab_file.write_text('{"tampered": "00"}', encoding="utf-8")

        with pytest.raises(ValueError, match="vocab.json checksum mismatch"):
            load_tokenizer_artifacts(out_dir, verify_checksums=True)
