"""
ChakrView Tokenizer: Minimal BPE Merge Engine.

Implements the deterministic Byte-Level Byte Pair Encoding (BBPE) core logic.
Supports:
  1. Foundational byte vocabulary construction.
  2. Pair frequency counting across token sequences.
  3. Deterministic merge selection with explicit tie-breaking rules.
  4. Merge application to token sequences.
  5. Multi-iteration training of a toy merge vocabulary.
  6. Fixed-rule deterministic sequence encoding using merge ranks.
  7. Exact subword byte reconstruction for decoding.

Deterministic Tie-Breaking Rule:
--------------------------------
When multiple candidate token pairs have identical frequencies during training,
ties are broken by evaluating:
  (-frequency, pair[0], pair[1])
Primary key: Frequency descending.
Secondary key: First token ID ascending (lexicographical numerical order).
Tertiary key: Second token ID ascending.
This ensures 100% platform-independent, reproducible merge tables.
"""

from typing import Dict, List, Optional, Set, Tuple

from chakrview.tokenizer.bytes import (
    BYTE_OFFSET,
    NUM_BYTE_TOKENS,
    byte_seq_to_token_ids,
)

BASE_VOCAB_SIZE: int = BYTE_OFFSET + NUM_BYTE_TOKENS  # 3 + 256 = 259


def create_base_vocab() -> Dict[int, bytes]:
    """
    Construct the foundational 256 byte vocabulary.
    Returns:
        Mapping from token_id (3..258) to its exact 1-byte representation.
    """
    return {b + BYTE_OFFSET: bytes([b]) for b in range(NUM_BYTE_TOKENS)}


def count_pairs(tokens: List[int]) -> Dict[Tuple[int, int], int]:
    """
    Count the frequency of each adjacent token pair in the sequence.
    """
    counts: Dict[Tuple[int, int], int] = {}
    for i in range(len(tokens) - 1):
        pair = (tokens[i], tokens[i + 1])
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def select_best_pair(counts: Dict[Tuple[int, int], int]) -> Optional[Tuple[int, int]]:
    """
    Select the highest frequency token pair using strict deterministic tie-breaking.
    Tie-breaking rule:
      Key: (-frequency, pair[0], pair[1])
    Returns None if counts dictionary is empty.
    """
    if not counts:
        return None
    return min(counts.keys(), key=lambda p: (-counts[p], p[0], p[1]))


def apply_merge(
    tokens: List[int], pair: Tuple[int, int], new_token_id: int
) -> List[int]:
    """
    Replace all non-overlapping occurrences of `pair` with `new_token_id`
    in a deterministic left-to-right single pass.
    """
    if len(tokens) < 2:
        return list(tokens)

    p0, p1 = pair
    merged: List[int] = []
    i = 0
    n = len(tokens)

    while i < n:
        if i < n - 1 and tokens[i] == p0 and tokens[i + 1] == p1:
            merged.append(new_token_id)
            i += 2
        else:
            merged.append(tokens[i])
            i += 1

    return merged


def train_toy_bpe(
    data: bytes,
    max_merges: int = 10,
    min_frequency: int = 2,
) -> Tuple[Dict[Tuple[int, int], int], Dict[int, bytes]]:
    """
    Train a minimal toy BPE merge table on raw bytes.

    Args:
        data: Raw training bytes.
        max_merges: Maximum number of merges to perform.
        min_frequency: Minimum frequency required to qualify for a merge.

    Returns:
        merges: Dict mapping (token_a, token_b) -> new_token_id
        vocab: Dict mapping token_id -> bytes
    """
    tokens = byte_seq_to_token_ids(data)
    vocab = create_base_vocab()
    merges: Dict[Tuple[int, int], int] = {}

    current_token_id = BASE_VOCAB_SIZE  # Starts at 259

    for _ in range(max_merges):
        if len(tokens) < 2:
            break

        counts = count_pairs(tokens)
        if not counts:
            break

        best_pair = select_best_pair(counts)
        if best_pair is None or counts[best_pair] < min_frequency:
            break

        p0, p1 = best_pair
        merges[best_pair] = current_token_id
        vocab[current_token_id] = vocab[p0] + vocab[p1]

        tokens = apply_merge(tokens, best_pair, current_token_id)
        current_token_id += 1

    return merges, vocab


def encode_with_merges(
    tokens: List[int],
    merges: Dict[Tuple[int, int], int],
) -> List[int]:
    """
    Deterministically encode a list of base token IDs using a fixed merge dictionary.
    Iteratively merges the lowest-ranked (highest-priority) adjacent pair until
    no further valid merges exist.
    """
    if len(tokens) < 2 or not merges:
        return list(tokens)

    # Precompute rank mapping for O(1) rank check: rank is insertion order
    merge_ranks: Dict[Tuple[int, int], int] = {pair: rank for rank, pair in enumerate(merges.keys())}

    current = list(tokens)

    while len(current) >= 2:
        # Find all adjacent pairs present in merges
        eligible_pairs: Set[Tuple[int, int]] = set()
        for i in range(len(current) - 1):
            pair = (current[i], current[i + 1])
            if pair in merge_ranks:
                eligible_pairs.add(pair)

        if not eligible_pairs:
            break

        # Pick the pair with the lowest rank (highest priority)
        best_pair = min(eligible_pairs, key=lambda p: merge_ranks[p])
        new_token_id = merges[best_pair]

        current = apply_merge(current, best_pair, new_token_id)

    return current
