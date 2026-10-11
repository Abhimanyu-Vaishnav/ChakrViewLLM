"""
ChakrView Tokenizer Corpus Pipeline: Deterministic Split Module.

Partitions corpus into train and validation sets using deterministic hashing.
Ensures stratified category representation across splits.
"""

import hashlib
from typing import Dict, List, Tuple
from chakrview.tokenizer_corpus.loader import CorpusItem


def item_hash_score(item: CorpusItem) -> float:
    """
    Produce a deterministic floating-point score in [0.0, 1.0) for a CorpusItem
    based on its category and text content using SHA-256.
    """
    payload = f"{item.category}:{item.text}".encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    # Use first 8 hex characters as integer
    int_val = int(digest[:8], 16)
    return int_val / 0xFFFFFFFF


def deterministic_train_val_split(
    corpus: Dict[str, List[CorpusItem]],
    val_ratio: float = 0.20,
) -> Tuple[List[CorpusItem], List[CorpusItem]]:
    """
    Split the corpus into training and validation sets deterministically.

    Args:
        corpus: Mapping from category name to list of CorpusItem instances.
        val_ratio: Fraction of items assigned to the validation set (default: 0.20).

    Returns:
        train_items: Deterministic training set.
        val_items: Deterministic validation set.
    """
    if not (0.0 < val_ratio < 1.0):
        raise ValueError(f"val_ratio must be between 0.0 and 1.0, got {val_ratio}")

    train_items: List[CorpusItem] = []
    val_items: List[CorpusItem] = []

    for category in sorted(corpus.keys()):
        items = corpus[category]
        # Sort items deterministically by text to prevent any ordering instability
        sorted_items = sorted(items, key=lambda x: x.text)
        for item in sorted_items:
            score = item_hash_score(item)
            if score < val_ratio:
                val_items.append(item)
            else:
                train_items.append(item)

    return train_items, val_items
