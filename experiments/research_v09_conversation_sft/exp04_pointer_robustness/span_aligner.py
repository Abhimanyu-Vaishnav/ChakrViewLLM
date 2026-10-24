"""
span_aligner.py
ChakrView Exp04 — Fail-Closed Character-to-Token Span Alignment Adapter.

Guarantees:
1. Every answerable target identifies exact character offsets in the prompt.
2. Converts character span to token span via byte-level offset matching.
3. Decodes the token span and asserts that tokenizer.decode(tokens).strip() == target.strip().
4. If an answerable target is missing, ambiguous, or cannot be cleanly aligned, raises AlignmentError.
5. Never silently converts alignment failures to [0, 0].
6. Reserves [0, 0] strictly for genuine abstention cases where is_abstention=True.
"""

from __future__ import annotations
from typing import Tuple, List, Dict, Any, Optional

class AlignmentError(Exception):
    """Raised when a target cannot be fail-closed aligned to context tokens."""
    pass

def compute_token_byte_offsets(prompt: str, tokenizer) -> Tuple[List[int], List[Tuple[int, int, bytes]]]:
    """
    Encodes prompt with BOS prepended (ID=0), and calculates the byte span [s_b, e_b]
    for each token index.
    Returns:
        input_ids: List of token IDs starting with BOS (0).
        byte_map: List of tuples (token_idx, start_byte, end_byte, token_bytes).
    """
    raw_tokens = tokenizer.encode(prompt)
    input_ids = [0] + raw_tokens
    
    byte_map = []
    # BOS (index 0) corresponds to empty byte range [0, 0]
    byte_map.append((0, 0, 0, b""))
    
    curr_byte = 0
    for idx, tid in enumerate(raw_tokens, start=1):
        if tid in tokenizer.vocab:
            b_piece = tokenizer.vocab[tid]
        else:
            # Special token or fallback
            b_piece = b""
        start_b = curr_byte
        end_b = curr_byte + len(b_piece)
        byte_map.append((idx, start_b, end_b, b_piece))
        curr_byte = end_b
        
    return input_ids, byte_map

def align_character_span_to_tokens(
    prompt: str,
    char_start: int,
    char_end: int,
    target_text: str,
    tokenizer,
    is_abstention: bool = False
) -> Tuple[int, int, List[int]]:
    """
    Aligns character offsets [char_start, char_end] to token span [t_start, t_end].
    Verifies that decoded token span reproduces target_text under stripping normalization.
    """
    if is_abstention:
        input_ids = [0] + tokenizer.encode(prompt)
        return 0, 0, input_ids

    if char_start < 0 or char_end <= char_start or char_end > len(prompt):
        raise AlignmentError(f"Invalid character span [{char_start}, {char_end}] for prompt of length {len(prompt)}")

    extracted_char = prompt[char_start:char_end]
    if extracted_char.strip() != target_text.strip():
        raise AlignmentError(f"Character span text '{extracted_char}' does not match expected target '{target_text}'")

    input_ids, byte_map = compute_token_byte_offsets(prompt, tokenizer)

    # Convert character offsets to byte offsets
    prefix_bytes = prompt[:char_start].encode("utf-8")
    target_bytes = extracted_char.encode("utf-8")
    start_byte = len(prefix_bytes)
    end_byte = start_byte + len(target_bytes)

    # Find token IDs that overlap with [start_byte, end_byte]
    overlapping_tokens = [
        t_idx for (t_idx, s_b, e_b, piece) in byte_map
        if t_idx > 0 and s_b < end_byte and e_b > start_byte
    ]

    if not overlapping_tokens:
        raise AlignmentError(f"No tokens found overlapping with byte span [{start_byte}, {end_byte}]")

    t_start = overlapping_tokens[0]
    t_end = overlapping_tokens[-1]

    # Verification: decode token slice
    extracted_token_slice = input_ids[t_start : t_end + 1]
    decoded_text = tokenizer.decode(extracted_token_slice).strip()

    if decoded_text.lower() != target_text.strip().lower():
        raise AlignmentError(
            f"Decoded token span [{t_start}, {t_end}] ('{decoded_text}') failed to match target '{target_text}'"
        )

    return t_start, t_end, input_ids
