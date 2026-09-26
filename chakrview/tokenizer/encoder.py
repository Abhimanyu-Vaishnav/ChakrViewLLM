"""
ChakrView Tokenizer: Encoder Module.

Converts raw text or binary data into deterministic sequences of token IDs.

Phases Covered:
- Phase C: Lossless UTF-8 text encoding.
- Phase D: Raw byte sequence encoding (preserves arbitrary byte sequences).
- Phase F: Strict determinism (repeated calls yield identical sequences).
- Phase G: Special token isolation (literal strings like '<BOS>' are encoded as
           their literal constituent characters/bytes, NEVER special token IDs).
"""

from typing import Dict, List, Optional, Tuple

from chakrview.tokenizer.bpe import encode_with_merges
from chakrview.tokenizer.bytes import byte_seq_to_token_ids
from chakrview.tokenizer.special_tokens import BOS_ID, EOS_ID


def encode_bytes(
    data: bytes,
    merges: Optional[Dict[Tuple[int, int], int]] = None,
) -> List[int]:
    """
    Encode an arbitrary sequence of raw bytes into token IDs.

    Args:
        data: Raw bytes to encode (can be any 8-bit byte sequence, valid or invalid UTF-8).
        merges: Optional dictionary of learned BPE merges (pair -> new_id).

    Returns:
        List of integer token IDs.
    """
    if not isinstance(data, (bytes, bytearray)):
        raise TypeError(f"Expected bytes or bytearray, got {type(data).__name__}")

    tokens = byte_seq_to_token_ids(data)
    if merges:
        tokens = encode_with_merges(tokens, merges)
    return tokens


def encode_text(
    text: str,
    merges: Optional[Dict[Tuple[int, int], int]] = None,
    add_bos: bool = False,
    add_eos: bool = False,
) -> List[int]:
    """
    Encode a Unicode text string into a deterministic sequence of token IDs.

    Args:
        text: Input string.
        merges: Optional dictionary of learned BPE merges.
        add_bos: If True, prepend the special <BOS> token (ID 0).
        add_eos: If True, append the special <EOS> token (ID 1).

    Returns:
        Deterministic list of integer token IDs.

    Security & Safety Invariant (Phase G):
        Literal substrings such as "<BOS>", "<EOS>", or "<PAD>" within `text`
        are encoded strictly into their UTF-8 byte representations. They NEVER
        trigger special-token emission. Special tokens are only injected when
        explicitly instructed via `add_bos` or `add_eos`.
    """
    if not isinstance(text, str):
        raise TypeError(f"Expected str, got {type(text).__name__}")

    raw_bytes = text.encode("utf-8")
    tokens = encode_bytes(raw_bytes, merges=merges)

    if add_bos:
        tokens = [BOS_ID] + tokens
    if add_eos:
        tokens = tokens + [EOS_ID]

    return tokens
