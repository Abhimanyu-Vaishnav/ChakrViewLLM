"""
Automated unit test suite for UCIP-12 corpus foundation, tokenization,
and cryptographic integrity.
"""

import hashlib
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chakrview.tokenizer.serialization import load_tokenizer_artifacts


def test_protected_session_hash():
    """Verify runtime session.py remains untouched."""
    session_path = ROOT / "chakrview" / "runtime" / "session.py"
    assert session_path.is_file(), "session.py missing!"
    h = hashlib.sha256(session_path.read_bytes()).hexdigest()
    expected = "9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67"
    assert h == expected, f"session.py hash mismatch: {h} != {expected}"


def test_frozen_tokenizer_roundtrip():
    """Verify frozen BPE tokenizer loads and exhibits lossless round-trip on multilingual samples."""
    tok_dir = ROOT / "data" / "experiments" / "vocab_4096"
    tokenizer, cfg = load_tokenizer_artifacts(tok_dir)
    assert tokenizer.vocab_size == 4096
    assert tokenizer.num_merges == 3837

    samples = [
        "ChakrView neural language model architecture",
        "कृत्रिम बुद्धिमत्ता और भाषा अधिग्रहण का अध्ययन",
        "def compute_loss(logits, targets):\n    return F.cross_entropy(logits, targets)",
        "सत्यमेव जयते नानृतम्",
        "Bhai yeh system ka scale 7.7M tokens hai."
    ]
    for s in samples:
        tokens = tokenizer.encode(s)
        decoded = tokenizer.decode(tokens)
        assert decoded == s, f"Lossless roundtrip failed for: {s}"


def test_stage_c_manifest_and_scale():
    """Verify Stage C manifest documents >7.7M tokens and valid domain breakdown."""
    manifest_path = ROOT / "data" / "manifests" / "stage_c_manifest.json"
    assert manifest_path.is_file()
    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    tokens = data.get("total_content_tokens", 0)
    assert tokens >= 7_700_000, f"Token count {tokens} < 7,700,000"
    assert data.get("total_documents", 0) >= 5500

    domains = data.get("domain_distribution", {})
    assert "english" in domains and domains["english"] > 2_000_000
    assert "hindi" in domains and domains["hindi"] > 2_000_000
    assert "code" in domains and domains["code"] > 1_000_000


def test_stage_c_binary_shards_exist():
    """Verify that Stage C binary shards exist and match expected counts."""
    tokenized_dir = ROOT / "data" / "tokenized" / "stage_c"
    train_shards = list((tokenized_dir / "train").glob("*.bin"))
    val_shards = list((tokenized_dir / "validation").glob("*.bin"))
    test_shards = list((tokenized_dir / "test").glob("*.bin"))

    assert len(train_shards) >= 27
    assert len(val_shards) >= 2
    assert len(test_shards) >= 3

    # Check that shard files are non-empty and multiple of 2 bytes (uint16)
    for shard in train_shards[:3]:
        sz = shard.stat().st_size
        assert sz > 0
        assert sz % 2 == 0


def test_sealed_benchmark_integrity():
    """Verify sealed benchmark hash matches canonical hash."""
    sealed_path = ROOT / "experiments" / "ucip12" / "benchmarks" / "sealed" / "benchmark_r2_sealed.json"
    assert sealed_path.is_file()
    h = hashlib.sha256(sealed_path.read_bytes()).hexdigest()
    expected = "ef94076849dd5c61cdea291e3a2ede35fdda96f47e36c39e53d8cc54fb0f36ef"
    assert h == expected, f"Sealed benchmark hash mismatch: {h} != {expected}"


def test_source_registry_cleanliness():
    """Verify that no rejected sources are approved in source registry."""
    reg_path = ROOT / "experiments" / "ucip12" / "provenance" / "source_registry.json"
    assert reg_path.is_file()
    with open(reg_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for s in data["sources"]:
        if s["source_id"] in ["iitb_parallel_hinglish", "common_crawl_unfiltered", "books3_pile"]:
            assert s["acquisition_status"] == "REJECTED"
        elif s["source_id"] == "l3cube_mahahinglish":
            assert s["acquisition_status"] == "CONDITIONAL"
