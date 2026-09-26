"""
ChakrView Tokenizer Corpus Pipeline Package.

Provides isolated tools for loading, validating, analyzing, deduplicating,
and partitioning the tokenizer research corpus.
"""

from chakrview.tokenizer_corpus.loader import load_corpus, load_category
from chakrview.tokenizer_corpus.validator import validate_corpus, CorpusValidationError
from chakrview.tokenizer_corpus.normalization import (
    TRAINING_NORMALIZATION_POLICY,
    RUNTIME_ENCODING_POLICY,
    apply_training_normalization,
)
from chakrview.tokenizer_corpus.dedup import deduplicate_lines
from chakrview.tokenizer_corpus.split import deterministic_train_val_split
from chakrview.tokenizer_corpus.statistics import compute_corpus_statistics, format_statistics_report

__all__ = [
    "load_corpus",
    "load_category",
    "validate_corpus",
    "CorpusValidationError",
    "TRAINING_NORMALIZATION_POLICY",
    "RUNTIME_ENCODING_POLICY",
    "apply_training_normalization",
    "deduplicate_lines",
    "deterministic_train_val_split",
    "compute_corpus_statistics",
    "format_statistics_report",
]
