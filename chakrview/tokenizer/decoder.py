"""
ChakrView Tokenizer: Decoder Module.

Converts token ID sequences back into exact original raw bytes or Unicode text.

Phases Covered:
- Phase C: Lossless text reconstruction: decode_tokens(encode_text(text)) == text.
- Phase D: Raw byte sequence round trip: decode_bytes(encode_bytes(data)) == data.
- Phase F: Strict determinism: repeated calls yield identical decoded outputs.
- Phase G: Special token filtering: skips special tokens during text reconstruction.
"""

from typing import Dict, List, Optional

from chakrview.tokenizer.bpe import create_base_vocab
from chakrview.tokenizer.special_tokens import is_special_token


def decode_bytes(
    tokens: List[int],
    vocab: Optional[Dict[int, bytes]] = None,
    skip_special_tokens: bool = True,
) -> bytes:
    """
    Decode a list of token IDs back into raw bytes.

    Args:
        tokens: Sequence of integer token IDs.
        vocab: Mapping from token ID to its underlying bytes (base bytes + merges).
               If None, defaults to foundational 256 base byte vocabulary.
        skip_special_tokens: If True, ignores special tokens (IDs 0, 1, 2).

    Returns:
        Exact reconstructed bytes.
    """
    if not isinstance(tokens, list):
        raise TypeError(f"Expected list of int, got {type(tokens).__name__}")

    if vocab is None:
        vocab = create_base_vocab()

    buffer = bytearray()

    for tid in tokens:
        if is_special_token(tid):
            if skip_special_tokens:
                continue
            # Special tokens do not have a raw byte representation in vocabulary
            continue

        if tid not in vocab:
            raise KeyError(f"Token ID {tid} not found in tokenizer vocabulary")

        buffer.extend(vocab[tid])

    return bytes(buffer)


def decode_tokens(
    tokens: List[int],
    vocab: Optional[Dict[int, bytes]] = None,
    skip_special_tokens: bool = True,
    errors: str = "strict",
) -> str:
    """
    Decode a list of token IDs into a reconstructed Unicode string.

    Args:
        tokens: Sequence of integer token IDs.
        vocab: Mapping from token ID to its underlying bytes.
        skip_special_tokens: If True, ignores special tokens during text reconstruction.
        errors: Error handling scheme for UTF-8 decoding ('strict', 'replace', etc.).

    Returns:
        Reconstructed Unicode string.

    Invariant:
        For every valid Unicode string:
        decode_tokens(encode_text(text, merges=m), vocab=v) == text
    """
    raw_bytes = decode_bytes(tokens, vocab=vocab, skip_special_tokens=skip_special_tokens)
    return raw_bytes.decode("utf-8", errors=errors)
