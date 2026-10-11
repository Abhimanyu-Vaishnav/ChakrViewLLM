"""
ChakrView Tokenizer Corpus Pipeline: Deduplication Module.

Detects and filters exact and normalized duplicate lines to prevent
artificial frequency distortion during BPE merge learning.
"""

from typing import List, Set, Tuple
from chakrview.tokenizer_corpus.loader import CorpusItem
from chakrview.tokenizer_corpus.normalization import apply_training_normalization


def deduplicate_lines(
    items: List[CorpusItem],
    use_training_norm: bool = True,
) -> Tuple[List[CorpusItem], int]:
    """
    Deduplicate items while preserving original appearance order.

    Args:
        items: List of CorpusItem instances.
        use_training_norm: If True, uses normalized text for duplicate key matching.

    Returns:
        unique_items: List of deduplicated CorpusItem instances.
        duplicate_count: Number of duplicate items removed.
    """
    seen_keys: Set[str] = set()
    unique_items: List[CorpusItem] = []
    duplicate_count = 0

    for item in items:
        key = apply_training_normalization(item.text) if use_training_norm else item.text
        if key in seen_keys:
            duplicate_count += 1
        else:
            seen_keys.add(key)
            unique_items.append(item)

    return unique_items, duplicate_count
