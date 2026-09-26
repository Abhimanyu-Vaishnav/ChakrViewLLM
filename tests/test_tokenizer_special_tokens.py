"""
ChakrView Tokenizer Tests: Special Token Safety & Isolation (Phases B & G).

Verifies that:
1. Special tokens have IDs <BOS>=0, <EOS>=1, <PAD>=2.
2. Literal text containing "<BOS>", "<EOS>", "<PAD>" never emits token IDs 0, 1, 2.
3. Special tokens are only added when explicitly requested via add_bos / add_eos.
4. decode_tokens ignores special tokens during text reconstruction.
"""

from chakrview.tokenizer.bytes import byte_to_token_id
from chakrview.tokenizer.decoder import decode_tokens
from chakrview.tokenizer.encoder import encode_text
from chakrview.tokenizer.special_tokens import (
    BOS_ID,
    EOS_ID,
    NUM_SPECIAL_TOKENS,
    PAD_ID,
    is_special_token,
)
from chakrview.tokenizer.tokenizer import BPETokenizer


def test_special_token_id_values():
    """Verify exact frozen special token IDs."""
    assert BOS_ID == 0
    assert EOS_ID == 1
    assert PAD_ID == 2
    assert NUM_SPECIAL_TOKENS == 3


def test_literal_special_token_string_isolation():
    """
    Phase G: Literal text '<BOS>', '<EOS>', '<PAD>' must NEVER produce
    token IDs 0, 1, or 2.
    """
    literal_strings = [
        "<BOS>",
        "<EOS>",
        "<PAD>",
        "The model emitted <BOS> unexpectedly.",
        "<BOS>Hello<EOS>",
        "<PAD><PAD><PAD>",
    ]

    for lit in literal_strings:
        tokens = encode_text(lit, add_bos=False, add_eos=False)
        # None of the tokens should be 0, 1, or 2
        for tid in tokens:
            assert not is_special_token(tid), (
                f"Literal text {lit!r} emitted special token ID {tid}"
            )

        # Token count must correspond to UTF-8 byte length (prior to merges)
        assert len(tokens) == len(lit.encode("utf-8"))

        # Reconstructed text must exactly match the literal string including "<BOS>"
        reconstructed = decode_tokens(tokens)
        assert reconstructed == lit


def test_explicit_special_token_generation():
    """
    Verify that special tokens are only generated when explicitly requested via parameters.
    """
    text = "ChakrView Core"

    # Default: no special tokens
    tokens_raw = encode_text(text, add_bos=False, add_eos=False)
    assert BOS_ID not in tokens_raw
    assert EOS_ID not in tokens_raw

    # With add_bos=True
    tokens_bos = encode_text(text, add_bos=True, add_eos=False)
    assert tokens_bos[0] == BOS_ID
    assert tokens_bos[1:] == tokens_raw

    # With add_eos=True
    tokens_eos = encode_text(text, add_bos=False, add_eos=True)
    assert tokens_eos[-1] == EOS_ID
    assert tokens_eos[:-1] == tokens_raw

    # With both
    tokens_both = encode_text(text, add_bos=True, add_eos=True)
    assert tokens_both[0] == BOS_ID
    assert tokens_both[-1] == EOS_ID
    assert tokens_both[1:-1] == tokens_raw

    # Decoding with skip_special_tokens=True restores the exact text
    assert decode_tokens(tokens_both, skip_special_tokens=True) == text


def test_bpe_tokenizer_special_token_behavior():
    """Verify BPETokenizer class handles special tokens cleanly."""
    tokenizer = BPETokenizer()
    text = "Hello <BOS> World <EOS>"

    # Encode with flags
    tokens = tokenizer.encode(text, add_bos=True, add_eos=True)
    assert tokens[0] == BOS_ID
    assert tokens[-1] == EOS_ID

    # Internal occurrences of '<BOS>' and '<EOS>' are NOT special tokens
    internal_tokens = tokens[1:-1]
    assert BOS_ID not in internal_tokens
    assert EOS_ID not in internal_tokens

    # Decoding reconstructs text cleanly
    decoded = tokenizer.decode(tokens, skip_special_tokens=True)
    assert decoded == text
