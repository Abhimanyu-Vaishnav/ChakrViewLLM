"""
ChakrView Tokenizer Tests: Byte Vocabulary & Raw Byte Round Trip (Phases A & D).
"""

import pytest

from chakrview.tokenizer.bytes import (
    BYTE_OFFSET,
    NUM_BYTE_TOKENS,
    byte_seq_to_token_ids,
    byte_to_token_id,
    is_byte_token,
    token_id_to_byte,
    token_ids_to_byte_seq,
)
from chakrview.tokenizer.encoder import encode_bytes
from chakrview.tokenizer.decoder import decode_bytes
from chakrview.tokenizer.special_tokens import BOS_ID, EOS_ID, PAD_ID


def test_byte_tokens_count_and_range():
    """Verify that exactly 256 byte tokens exist and are cleanly separated from special tokens."""
    assert NUM_BYTE_TOKENS == 256
    assert BYTE_OFFSET == 3  # IDs 0, 1, 2 reserved for BOS, EOS, PAD

    # Byte 0x00 maps to 3, 0xFF (255) maps to 258
    assert byte_to_token_id(0) == 3
    assert byte_to_token_id(255) == 258


def test_special_tokens_no_collision():
    """Verify that special token IDs are never classified as byte tokens."""
    for special_id in (BOS_ID, EOS_ID, PAD_ID):
        assert not is_byte_token(special_id)
        with pytest.raises(ValueError):
            token_id_to_byte(special_id)


def test_all_256_bytes_bijective_mapping():
    """Verify that every byte from 0x00 to 0xFF has a strictly bijective token ID."""
    seen_ids = set()
    for b in range(256):
        tid = byte_to_token_id(b)
        assert tid not in seen_ids, f"Collision detected for byte {b}"
        seen_ids.add(tid)
        assert is_byte_token(tid)
        assert token_id_to_byte(tid) == b

    assert len(seen_ids) == 256


def test_byte_bounds_validation():
    """Verify that invalid byte integers raise ValueError."""
    with pytest.raises(ValueError):
        byte_to_token_id(-1)
    with pytest.raises(ValueError):
        byte_to_token_id(256)
    with pytest.raises(ValueError):
        byte_to_token_id(1000)

    with pytest.raises(ValueError):
        token_id_to_byte(0)  # BOS
    with pytest.raises(ValueError):
        token_id_to_byte(2)  # PAD
    with pytest.raises(ValueError):
        token_id_to_byte(259)  # Merge token range


def test_raw_byte_round_trip_all_256_bytes():
    """
    Phase D: Test that all 256 possible byte values round-trip perfectly
    through encode_bytes and decode_bytes.
    """
    all_bytes = bytes(range(256))
    tokens = encode_bytes(all_bytes)
    assert len(tokens) == 256

    reconstructed = decode_bytes(tokens)
    assert reconstructed == all_bytes
    assert len(reconstructed) == 256


def test_arbitrary_non_utf8_binary_round_trip():
    """
    Phase D: Test arbitrary binary sequences containing invalid UTF-8 sequences.
    Must not crash, mutate, or drop any bytes.
    """
    # 0x80 is an invalid UTF-8 start byte; 0xFF and 0xFE are invalid UTF-8 bytes
    invalid_utf8_sequences = [
        b"\x80",
        b"\xff",
        b"\xfe\xff",
        b"\xc0\xaf",  # Overlong UTF-8
        b"\x00\x00\x00",  # Null bytes
        b"\xed\xa0\x80",  # UTF-16 surrogate in UTF-8
        b"\xf4\x90\x80\x80",  # Codepoint above U+10FFFF
        bytes([0xFF, 0x80, 0x00, 0x7F, 0xC3, 0x28]),
    ]

    for binary_blob in invalid_utf8_sequences:
        tokens = encode_bytes(binary_blob)
        reconstructed = decode_bytes(tokens)
        assert reconstructed == binary_blob, f"Failed round-trip for {binary_blob!r}"


def test_byte_seq_helpers():
    """Test lower-level byte_seq_to_token_ids and token_ids_to_byte_seq."""
    data = b"ChakrView"
    token_ids = byte_seq_to_token_ids(data)
    assert len(token_ids) == len(data)
    assert token_ids_to_byte_seq(token_ids) == data

    with pytest.raises(TypeError):
        byte_seq_to_token_ids("not bytes")  # type: ignore

    with pytest.raises(ValueError):
        token_ids_to_byte_seq([BOS_ID])  # contains special token
