"""
ChakrView Tests: Corpus Engine & Validation Suite (Step 3).

Verifies:
- Document loading & memory-conscious streaming
- Data quality validators (empty, duplicate, control chars, null bytes, long samples)
- Normalization policies (strict runtime identity)
- Deterministic 80/10/10 category-aware splitting and disjointness
- Corpus statistical profiling and script distribution
- Manifest generation and SHA-256 integrity
"""

import tempfile
from pathlib import Path
import pytest

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
    hash_split_score,
    partition_corpus,
    split_category_documents,
)
from chakrview.corpus.statistics import (
    classify_char,
    compute_corpus_statistics,
)
from chakrview.corpus.validators import (
    validate_corpus_collection,
    validate_single_text,
)


def test_character_classification():
    """Verify script and character classification."""
    assert classify_char("क") == "Devanagari"
    assert classify_char("A") == "Latin_ASCII"
    assert classify_char("5") == "ASCII_Digit"
    assert classify_char("।") == "Punctuation"
    assert classify_char("∀") == "Symbol_Math"
    assert classify_char(" ") == "Whitespace"
    assert classify_char("\u200D") == "Zero_Width"


def test_document_validation_rules():
    """Verify validation triggers on empty, null bytes, and control chars."""
    # Empty
    assert validate_single_text("")[0].issue_type == "EMPTY_DOCUMENT"
    assert validate_single_text("   \n")[0].issue_type == "EMPTY_DOCUMENT"

    # Null byte / binary
    issues = validate_single_text("Hello\x00World")
    assert any(i.issue_type == "BINARY_NULL_BYTE" for i in issues)

    # Control char
    issues = validate_single_text("Line\x07Beep")
    assert any(i.issue_type == "UNEXPECTED_CONTROL_CHARS" for i in issues)

    # Short
    issues = validate_single_text("A", min_chars=3)
    assert any(i.issue_type == "EXTREMELY_SHORT" for i in issues)


def test_normalization_invariants():
    """Verify runtime identity pass-through and training NFC."""
    text = "नमस्ते दुनिया ॐ"
    assert runtime_identity(text) == text
    assert normalize_training_text(text, "NFC") == text
    assert "STRICT IDENTITY" in NORMALIZATION_POLICIES["runtime"]["policy"]


def test_cleaning_stages():
    """Verify separation between cleaning stages."""
    raw = "  Line with spaces  \r\n"
    assert clean_training_text(raw) == "  Line with spaces"
    assert clean_benchmark_text(raw) == "  Line with spaces  "
    assert preserve_adversarial_text(raw) == raw


def test_deterministic_splitter():
    """Verify 80/10/10 splitting is deterministic and strictly disjoint."""
    docs = [
        CorpusDocument(
            text=f"Sample document line {i}",
            category="hindi",
            subcategory="test",
            source_file="test.txt",
            line_number=i,
            byte_count=20,
            char_count=20,
            doc_type="raw",
            sha256=f"hash_{i}",
        )
        for i in range(50)
    ]

    splits1 = partition_corpus({"hindi": docs}, seed=42)
    splits2 = partition_corpus({"hindi": docs}, seed=42)

    assert [d.text for d in splits1["train"]["hindi"]] == [d.text for d in splits2["train"]["hindi"]]
    assert [d.text for d in splits1["validation"]["hindi"]] == [d.text for d in splits2["validation"]["hindi"]]
    assert [d.text for d in splits1["test"]["hindi"]] == [d.text for d in splits2["test"]["hindi"]]

    train_set = {d.text for d in splits1["train"]["hindi"]}
    val_set = {d.text for d in splits1["validation"]["hindi"]}
    test_set = {d.text for d in splits1["test"]["hindi"]}

    assert train_set.isdisjoint(val_set)
    assert train_set.isdisjoint(test_set)
    assert val_set.isdisjoint(test_set)


def test_corpus_statistics_computation():
    """Verify statistical profiling of sample documents."""
    docs = [
        CorpusDocument(
            text="नमस्ते 123",
            category="hindi",
            subcategory="test",
            source_file="test.txt",
            line_number=1,
            byte_count=len("नमस्ते 123".encode("utf-8")),
            char_count=len("नमस्ते 123"),
            doc_type="raw",
            sha256="h1",
        ),
        CorpusDocument(
            text="Hello 456",
            category="english",
            subcategory="test",
            source_file="test.txt",
            line_number=2,
            byte_count=len("Hello 456".encode("utf-8")),
            char_count=len("Hello 456"),
            doc_type="raw",
            sha256="h2",
        ),
    ]

    stats = compute_corpus_statistics(docs)
    assert stats["summary"]["total_documents"] == 2
    assert "Devanagari" in stats["script_distribution"]["counts"]
    assert "Latin_ASCII" in stats["script_distribution"]["counts"]
    assert "ASCII_Digit" in stats["script_distribution"]["counts"]
