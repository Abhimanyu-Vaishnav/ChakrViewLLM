"""
ChakrView Corpus Pipeline Package.

Provides:
- Document loading & memory-conscious streaming (loader.py)
- Quality validation & reporting (validators.py)
- Distinct cleaning stages (cleaner.py)
- Normalization policies (normalizer.py)
- Deterministic category-aware splitting (splitter.py)
- Statistical profiling & script analysis (statistics.py)
- Manifest generation & checksumming (manifest.py)
"""

from chakrview.corpus.cleaner import (
    clean_benchmark_text,
    clean_training_text,
    prepare_text_by_stage,
    preserve_adversarial_text,
)
from chakrview.corpus.loader import (
    CorpusDocument,
    load_corpus_tree,
    load_file_documents,
    stream_corpus_documents,
)
from chakrview.corpus.manifest import (
    generate_corpus_manifest,
    generate_file_manifest,
)
from chakrview.corpus.normalizer import (
    NORMALIZATION_POLICIES,
    normalize_training_text,
    runtime_identity,
)
from chakrview.corpus.splitter import (
    partition_corpus,
    split_category_documents,
)
from chakrview.corpus.statistics import (
    compute_corpus_statistics,
    save_statistics_report,
)
from chakrview.corpus.validators import (
    ValidationIssue,
    validate_corpus_collection,
    validate_single_text,
)

__all__ = [
    "CorpusDocument",
    "load_file_documents",
    "load_corpus_tree",
    "stream_corpus_documents",
    "ValidationIssue",
    "validate_single_text",
    "validate_corpus_collection",
    "clean_training_text",
    "clean_benchmark_text",
    "preserve_adversarial_text",
    "prepare_text_by_stage",
    "NORMALIZATION_POLICIES",
    "normalize_training_text",
    "runtime_identity",
    "split_category_documents",
    "partition_corpus",
    "compute_corpus_statistics",
    "save_statistics_report",
    "generate_file_manifest",
    "generate_corpus_manifest",
]
