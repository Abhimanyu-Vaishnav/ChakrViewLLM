"""
ChakrView Tests: Corpus Pipeline Modules.

Tests:
- Corpus loading
- Corpus validation
- Deterministic train/validation splitting
- Duplicate detection
- Corpus statistics computation
"""

from pathlib import Path
import pytest

from chakrview.tokenizer_corpus.dedup import deduplicate_lines
from chakrview.tokenizer_corpus.loader import CorpusItem, load_category, load_corpus
from chakrview.tokenizer_corpus.normalization import (
    TRAINING_NORMALIZATION_POLICY,
    RUNTIME_ENCODING_POLICY,
    apply_training_normalization,
)
from chakrview.tokenizer_corpus.split import deterministic_train_val_split, item_hash_score
from chakrview.tokenizer_corpus.statistics import compute_corpus_statistics
from chakrview.tokenizer_corpus.validator import CorpusValidationError, validate_corpus


@pytest.fixture
def corpus_dir() -> Path:
    p = Path(__file__).resolve().parent.parent / "data" / "tokenizer_corpus"
    assert p.is_dir(), f"Corpus directory not found: {p}"
    return p


def test_corpus_loading(corpus_dir: Path):
    """Verify loading of all 8 designated corpus category files."""
    corpus = load_corpus(corpus_dir)
    expected_categories = {"hindi", "english", "hinglish", "mixed", "code", "numbers", "math", "unicode"}
    assert set(corpus.keys()) == expected_categories

    for cat, items in corpus.items():
        assert len(items) > 0, f"Category '{cat}' is empty"
        for item in items:
            assert isinstance(item, CorpusItem)
            assert item.category == cat
            assert len(item.text) > 0


def test_corpus_validation(corpus_dir: Path):
    """Verify that current corpus passes all validation criteria."""
    corpus = load_corpus(corpus_dir)
    validate_corpus(corpus)


def test_corpus_validation_failure_modes():
    """Verify validation triggers on empty corpus, empty items, or null bytes."""
    with pytest.raises(CorpusValidationError):
        validate_corpus({})

    with pytest.raises(CorpusValidationError):
        validate_corpus({"empty_cat": []})

    with pytest.raises(CorpusValidationError):
        bad_item = CorpusItem(text="", category="bad", source_file="test.txt", line_number=1)
        validate_corpus({"bad": [bad_item]})

    with pytest.raises(CorpusValidationError):
        null_item = CorpusItem(text="Hello\x00World", category="bad", source_file="test.txt", line_number=1)
        validate_corpus({"bad": [null_item]})


def test_normalization_policy_distinction():
    """Verify training normalization vs runtime lossless invariant."""
    assert "NFC" in TRAINING_NORMALIZATION_POLICY["unicode_normalization"]
    assert "STRICTLY LOSSLESS" in RUNTIME_ENCODING_POLICY["status"]

    sample = "  Hello World\r\n"
    norm = apply_training_normalization(sample)
    assert norm == "  Hello World"
    assert "\r" not in norm


def test_duplicate_detection():
    """Verify accurate detection and preservation of appearance order in deduplication."""
    items = [
        CorpusItem(text="Line A", category="cat", source_file="f.txt", line_number=1),
        CorpusItem(text="Line B", category="cat", source_file="f.txt", line_number=2),
        CorpusItem(text="Line A", category="cat", source_file="f.txt", line_number=3),
        CorpusItem(text="Line C", category="cat", source_file="f.txt", line_number=4),
        CorpusItem(text="Line B  \r\n", category="cat", source_file="f.txt", line_number=5),
    ]

    unique_items, dup_count = deduplicate_lines(items, use_training_norm=True)
    assert len(unique_items) == 3
    assert dup_count == 2
    assert [x.text for x in unique_items] == ["Line A", "Line B", "Line C"]


def test_deterministic_split(corpus_dir: Path):
    """Verify that train/validation splitting is 100% deterministic across multiple runs."""
    corpus = load_corpus(corpus_dir)

    train1, val1 = deterministic_train_val_split(corpus, val_ratio=0.20)
    train2, val2 = deterministic_train_val_split(corpus, val_ratio=0.20)

    assert len(train1) == len(train2)
    assert len(val1) == len(val2)
    assert [x.text for x in train1] == [x.text for x in train2]
    assert [x.text for x in val1] == [x.text for x in val2]

    # Verify no overlap between train and val
    train_keys = {f"{x.category}:{x.text}" for x in train1}
    val_keys = {f"{x.category}:{x.text}" for x in val1}
    assert train_keys.isdisjoint(val_keys)


def test_corpus_statistics(corpus_dir: Path):
    """Verify statistics computation metrics."""
    corpus = load_corpus(corpus_dir)
    stats = compute_corpus_statistics(corpus)

    assert stats["total_documents"] > 400
    assert stats["total_bytes"] > 30000
    assert "hindi" in stats["category_stats"]
    assert "english" in stats["category_stats"]
    assert "Devanagari" in stats["script_distribution"]
    assert "ASCII Latin" in stats["script_distribution"]
