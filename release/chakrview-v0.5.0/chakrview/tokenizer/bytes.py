"""
ChakrView Tokenizer: Byte Vocabulary Module.

Handles mapping between raw 8-bit bytes (0x00 through 0xFF) and foundational
byte-level token IDs.

Linguistic and Computational Disambiguation:
-------------------------------------------
1. Unicode Character:
   An abstract typographic element or grapheme cluster (e.g., 'क', 'A', '🚀').
2. Unicode Code Point:
   A unique integer scalar value assigned by the Unicode Consortium
   (e.g., 'क' -> U+0915, 'A' -> U+0041, '🚀' -> U+1F680).
3. UTF-8 Bytes:
   The standard variable-length binary encoding of code points into sequences
   of 1 to 4 raw bytes (e.g., 'क' -> b'\\xe0\\xa4\\x95', 'A' -> b'A').
4. Tokenizer Token IDs:
   Discrete integer identifiers used inside the neural vocabulary:
     - IDs 0..2: Special tokens (<BOS>=0, <EOS>=1, <PAD>=2)
     - IDs 3..258: Foundational 256 byte primitives (0x00..0xFF)
     - IDs 259+: Learned BPE merged subwords

This module guarantees that all 256 fundamental byte primitives have distinct,
collision-free token IDs separated from special-token IDs.
"""

from typing import Final, List

BYTE_OFFSET: Final[int] = 3
NUM_BYTE_TOKENS: Final[int] = 256
BYTE_TOKEN_RANGE: Final[range] = range(BYTE_OFFSET, BYTE_OFFSET + NUM_BYTE_TOKENS)


def byte_to_token_id(b: int) -> int:
    """Map a single 8-bit byte integer (0..255) to its unique token ID."""
    if not (0 <= b <= 255):
        raise ValueError(f"Byte value must be in range [0, 255], got {b}")
    return b + BYTE_OFFSET


def token_id_to_byte(token_id: int) -> int:
    """Map a byte-level token ID back to its original 8-bit byte integer (0..255)."""
    if token_id not in BYTE_TOKEN_RANGE:
        raise ValueError(
            f"Token ID {token_id} is not in byte token range [{BYTE_OFFSET}, {BYTE_OFFSET + NUM_BYTE_TOKENS - 1}]"
        )
    return token_id - BYTE_OFFSET


def is_byte_token(token_id: int) -> bool:
    """Return True if token_id represents one of the 256 foundational byte primitives."""
    return token_id in BYTE_TOKEN_RANGE


def byte_seq_to_token_ids(data: bytes) -> List[int]:
    """Convert an arbitrary sequence of raw bytes into base byte token IDs."""
    if not isinstance(data, (bytes, bytearray)):
        raise TypeError(f"Expected bytes or bytearray, got {type(data).__name__}")
    return [b + BYTE_OFFSET for b in data]


def token_ids_to_byte_seq(token_ids: List[int]) -> bytes:
    """
    Convert a list of base byte token IDs back into raw bytes.
    Raises ValueError if any token ID is outside the foundational byte token range.
    """
    out = bytearray(len(token_ids))
    for i, tid in enumerate(token_ids):
        if tid not in BYTE_TOKEN_RANGE:
            raise ValueError(
                f"Token ID {tid} at index {i} is not a foundational byte token. "
                "For sequences containing merged tokens or special tokens, use the full decoder."
            )
        out[i] = tid - BYTE_OFFSET
    return bytes(out)
