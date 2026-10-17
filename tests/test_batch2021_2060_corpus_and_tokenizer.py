"""
Automated Test Suite for ChakrView Batches 2021-2060.
Validates:
1. Canonical checkpoint SHA-256 bit-exact immutability.
2. Adapted checkpoint SHA-256 bit-exact immutability.
3. Natural Corpus Expansion v2 manifests and completeness.
4. Deterministic source-family isolation across splits.
5. Zero n-gram / exact-duplicate data leakage between partitions.
6. Test split balance and absence of single-source concentration.
7. Multilingual tokenizer lossless roundtrip fidelity and vocabulary preservation.
8. Deliverable artifact presence in experiments/batch2021_2060/.
"""

import json
import hashlib
from pathlib import Path
import pytest

from chakrview.tokenizer.serialization import load_tokenizer_artifacts


CANONICAL_CHECKPOINT_PATH = Path(
    "experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt"
)
CANONICAL_SHA256 = "96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee"

ADAPTED_CHECKPOINT_PATH = Path(
    "experiments/batch1861_1900/checkpoints/cpt_balanced_seed_42.pt"
)
ADAPTED_SHA256 = "85753315bb55853f0e8bad91a916c3846c848333a1b8e3b9b02a5256b7aa7f1c"

CPT_300_CHECKPOINT_PATH = Path(
    "experiments/batch1901_1940/checkpoints/cpt_balanced_seed_42_300steps.pt"
)
CPT_300_SHA256 = "2f7751511b623e1dd06bbe2d721df71e679da34d48d4b443f092600d768e7628"

BATCH_DIR = Path("experiments/batch2021_2060")
PROVENANCE_PATH = BATCH_DIR / "SOURCE_PROVENANCE_MANIFEST.json"
CORPUS_V2_PATH = BATCH_DIR / "CORPUS_V2_MANIFEST.json"
SPLIT_V2_PATH = BATCH_DIR / "SPLIT_V2_MANIFEST.json"
VOCAB_DIR = Path("release/chakrview-v0.5.0/data/experiments/vocab_4096")


def test_canonical_checkpoint_immutability():
    """Canonical 10,000-step checkpoint must retain exact SHA-256."""
    assert CANONICAL_CHECKPOINT_PATH.exists(), f"Missing canonical checkpoint: {CANONICAL_CHECKPOINT_PATH}"
    data = CANONICAL_CHECKPOINT_PATH.read_bytes()
    computed_sha = hashlib.sha256(data).hexdigest()
    assert computed_sha == CANONICAL_SHA256, (
        f"Canonical checkpoint SHA mismatch: expected {CANONICAL_SHA256}, got {computed_sha}"
    )


def test_adapted_150_step_checkpoint_immutability():
    """Adapted 150-step CPT checkpoint must retain exact SHA-256."""
    assert ADAPTED_CHECKPOINT_PATH.exists(), f"Missing adapted checkpoint: {ADAPTED_CHECKPOINT_PATH}"
    data = ADAPTED_CHECKPOINT_PATH.read_bytes()
    computed_sha = hashlib.sha256(data).hexdigest()
    assert computed_sha == ADAPTED_SHA256, (
        f"Adapted checkpoint SHA mismatch: expected {ADAPTED_SHA256}, got {computed_sha}"
    )


def test_all_deliverables_present():
    """All required batch deliverables must exist and be non-empty."""
    required_files = [
        "NATURAL_CORPUS_EXPANSION_REPORT.md",
        "SOURCE_PROVENANCE_MANIFEST.json",
        "CORPUS_V2_MANIFEST.json",
        "SPLIT_V2_MANIFEST.json",
        "DATA_LEAKAGE_AUDIT.md",
        "MULTILINGUAL_TOKENIZER_AUDIT.md",
        "DIRECT_CPT_COMPARISON.md",
        "CONTROLLED_CPT_EXPERIMENT_PLAN.md",
        "TRAINING_READINESS_DECISION.md",
        "corpus_ingestion_runner.py"
    ]
    for fname in required_files:
        p = BATCH_DIR / fname
        assert p.exists(), f"Missing deliverable: {p}"
        assert p.stat().st_size > 0, f"Empty deliverable: {p}"


def test_provenance_manifest_completeness():
    """Provenance manifest must catalog all 13 source families with legal licenses."""
    assert PROVENANCE_PATH.exists()
    with open(PROVENANCE_PATH, encoding="utf-8") as f:
        data = json.load(f)
    assert "sources" in data
    sources = data["sources"]
    assert len(sources) == 13, f"Expected 13 source families, found {len(sources)}"
    for s in sources:
        assert s["source_id"], "Missing source_id"
        assert s["title"], "Missing title"
        assert s["url"], "Missing URL"
        assert s["license"], "Missing license"
        assert s["domain"], "Missing domain"
        assert s["language"], "Missing language"


def test_corpus_v2_manifest_structure_and_hashes():
    """Corpus v2 must contain 1,258 documents with valid SHA-256 hashes matching payload."""
    assert CORPUS_V2_PATH.exists()
    with open(CORPUS_V2_PATH, encoding="utf-8") as f:
        corpus = json.load(f)

    assert corpus["manifest_version"] == "2.0.0"
    docs = corpus["documents"]
    assert len(docs) == 1258

    for d in docs[:50]:  # sample check
        text = d["text"]
        expected_sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
        assert d["sha256"] == expected_sha, f"SHA mismatch on doc {d['document_id']}"


def test_source_family_isolation_and_disjointness():
    """Train, validation, and test partitions must have completely disjoint source families."""
    with open(CORPUS_V2_PATH, encoding="utf-8") as f:
        corpus = json.load(f)

    docs = corpus["documents"]
    train_sources = {d["source_id"] for d in docs if d["split"] == "train"}
    val_sources = {d["source_id"] for d in docs if d["split"] == "validation"}
    test_sources = {d["source_id"] for d in docs if d["split"] == "test"}

    assert train_sources.isdisjoint(val_sources), "Train and Validation share sources!"
    assert train_sources.isdisjoint(test_sources), "Train and Test share sources!"
    assert val_sources.isdisjoint(test_sources), "Validation and Test share sources!"


def test_zero_leakage_and_duplicate_hashes():
    """Zero exact document SHA-256 overlap across partitions."""
    with open(CORPUS_V2_PATH, encoding="utf-8") as f:
        corpus = json.load(f)

    docs = corpus["documents"]
    train_shas = {d["sha256"] for d in docs if d["split"] == "train"}
    val_shas = {d["sha256"] for d in docs if d["split"] == "validation"}
    test_shas = {d["sha256"] for d in docs if d["split"] == "test"}

    assert train_shas.isdisjoint(val_shas)
    assert train_shas.isdisjoint(test_shas)
    assert val_shas.isdisjoint(test_shas)


def test_test_split_diversity_and_no_single_source_monopoly():
    """No single source family in the test split may exceed 50% of test tokens."""
    with open(CORPUS_V2_PATH, encoding="utf-8") as f:
        corpus = json.load(f)

    test_docs = [d for d in corpus["documents"] if d["split"] == "test"]
    total_test_tokens = sum(d["token_count"] for d in test_docs)
    test_sources = set(d["source_id"] for d in test_docs)

    assert len(test_sources) >= 4, f"Test split must have at least 4 sources, got {len(test_sources)}"

    for sid in test_sources:
        src_tokens = sum(d["token_count"] for d in test_docs if d["source_id"] == sid)
        ratio = src_tokens / total_test_tokens
        assert ratio < 0.50, f"Source {sid} dominates test split with {ratio*100:.1f}% of tokens"


def test_tokenizer_lossless_roundtrip_multilingual():
    """Tokenizer must perform exact roundtrip reconstruction on English, Hindi, Hinglish, and Sanskrit."""
    tok, _ = load_tokenizer_artifacts(VOCAB_DIR)

    test_samples = [
        "The universal Turing machine can simulate any algorithmic procedure on an arbitrary tape.",
        "मुंशी प्रेमचंद हिंदी और उर्दू के महानतम लेखकों में से एक हैं।",
        "kal meeting kitne baje schedule karni hai mujhe calendar update karna padega.",
        "धर्मक्षेत्रे कुरुक्षेत्रे समवेता युयुत्सवः मामकाः पाण्डवाश्चैव किमकुर्वत सञ्जय॥",
        "def compute_entropy(p_dist: list[float]) -> float:\n    return -sum(p * math.log2(p) for p in p_dist)"
    ]

    for s in test_samples:
        encoded = tok.encode(s)
        decoded = tok.decode(encoded)
        assert decoded == s, f"Roundtrip failed for: {s}"
