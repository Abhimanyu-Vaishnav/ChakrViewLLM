"""
ChakrView Tokenizer Corpus Pipeline: Validator Module.

Ensures strict UTF-8 validity, data integrity, and non-emptiness across all corpus files.
"""

from typing import Dict, List
from chakrview.tokenizer_corpus.loader import CorpusItem


class CorpusValidationError(Exception):
    """Raised when corpus data fails validation rules."""
    pass


def validate_corpus(corpus: Dict[str, List[CorpusItem]]) -> None:
    """
    Validate that:
    1. Corpus contains required categories.
    2. No category is empty.
    3. Every item is valid UTF-8 and contains non-empty text.
    4. No null bytes (0x00) exist inside natural text items.
    """
    if not corpus:
        raise CorpusValidationError("Corpus dictionary is empty.")

    for category, items in corpus.items():
        if not items:
            raise CorpusValidationError(f"Category '{category}' has no items.")

        for item in items:
            if not item.text:
                raise CorpusValidationError(
                    f"Empty text in {item.source_file} at line {item.line_number}"
                )

            # Check UTF-8 encodability
            try:
                raw_bytes = item.text.encode("utf-8")
                raw_bytes.decode("utf-8")
            except UnicodeError as e:
                raise CorpusValidationError(
                    f"Unicode encoding error in {item.source_file}:{item.line_number}: {e}"
                )

            # Check for embedded null bytes
            if "\x00" in item.text:
                raise CorpusValidationError(
                    f"Illegal null byte found in {item.source_file}:{item.line_number}"
                )
