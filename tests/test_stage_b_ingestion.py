"""
Step 6.2 Tests: Stage B Multi-Domain Corpus Ingestion, Validation, and Sharding.

Verifies:
1. Stage B Manifest completeness and schema adherence (Phase 6 contract).
2. Binary shard integrity and SHA-256 consistency across train, val, and test splits.
3. Token sequence dtype (uint16), bounds [0, 4095], and EOS delimiter contract.
4. Streaming dataset reader compatibility on production binary shards.
5. Strict deterministic split disjointness (zero document overlap across splits).
"""

import json
from pathlib import Path
import numpy as np
import pytest
import torch

from chakrview.training.sharding import read_shard_tokens, verify_shard_integrity
from chakrview.training.dataset import StreamingTokenDataset

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
STAGE_B_TOKENIZED = DATA_DIR / "tokenized" / "stage_b"
STAGE_B_MANIFEST = DATA_DIR / "manifests" / "stage_b_manifest.json"
STAGE_B_PROCESSED = DATA_DIR / "processed" / "stage_b"


def test_stage_b_manifest_contract():
    """Verify Stage B manifest exists, has valid structure and required metadata fields."""
    assert STAGE_B_MANIFEST.is_file(), f"Stage B manifest missing at {STAGE_B_MANIFEST}"

    with open(STAGE_B_MANIFEST, "r", encoding="utf-8") as f:
        meta = json.load(f)

    assert meta["manifest_version"] == "0.2.0"
    assert meta["tokenizer_vocab_size"] == 4096
    assert meta["total_documents"] >= 8000
    assert meta["total_content_tokens"] >= 450_000

    required_fields = {
        "document_id",
        "source_id",
        "source_type",
        "path",
        "language",
        "script",
        "category",
        "license",
        "provenance",
        "sha256",
        "byte_count",
        "character_count",
        "token_count",
        "quality_status",
        "duplicate_group",
        "ingestion_timestamp",
        "preprocessing_version",
        "tokenizer_version",
    }

    sample_doc = meta["documents"][0]
    assert required_fields.issubset(sample_doc.keys()), (
        f"Missing required fields: {required_fields - set(sample_doc.keys())}"
    )
    assert sample_doc["quality_status"] == "VALID"
    assert 0 <= sample_doc["token_count"]


def test_stage_b_shard_integrity():
    """Verify SHA-256 integrity and metadata consistency across train, val, and test shards."""
    for split in ("train", "validation", "test"):
        split_dir = STAGE_B_TOKENIZED / split
        assert split_dir.is_dir(), f"Missing split directory: {split_dir}"
        assert (split_dir / "metadata.json").is_file()

        # Check cryptographic integrity of binary shards
        assert verify_shard_integrity(split_dir) is True, f"Integrity check failed for {split}"

        # Check metadata attributes
        with open(split_dir / "metadata.json", "r", encoding="utf-8") as f:
            meta = json.load(f)

        assert meta["split"] == split
        assert meta["vocab_size"] == 4096
        assert meta["token_dtype"] == "uint16"
        assert meta["total_tokens"] > 0
        assert meta["total_documents"] > 0
        assert len(meta["shards"]) == meta["shard_count"]


def test_stage_b_shard_binary_contract():
    """Verify uint16 little-endian format and token bounds [0, 4095] on production shards."""
    train_shard = STAGE_B_TOKENIZED / "train" / "shard_00000.bin"
    assert train_shard.is_file(), "Train shard 00000 missing"

    tokens = read_shard_tokens(train_shard)
    assert tokens.dtype == np.uint16
    assert len(tokens) == 250_000  # Exact max_tokens_per_shard

    # Check bounds
    assert np.all(tokens < 4096), "Token ID out of vocabulary range [0, 4095]"
    # Ensure special tokens exist (EOS_ID = 1 delimiter)
    assert 1 in tokens, "EOS token missing from document boundaries in shard"


def test_stage_b_streaming_dataset_compatibility():
    """Verify StreamingTokenDataset produces aligned input/target pairs from Stage B shards."""
    train_dir = STAGE_B_TOKENIZED / "train"
    seq_len = 32
    dataset = StreamingTokenDataset(
        shard_dir=train_dir,
        sequence_length=seq_len,
        loop=False,
        drop_remainder=True,
    )

    batch_count = 0
    for item in dataset:
        inp = item["input_ids"]
        tgt = item["target_ids"]
        mask = item["attention_mask"]

        assert inp.shape == (seq_len,)
        assert tgt.shape == (seq_len,)
        assert mask.shape == (seq_len,)
        assert (inp < 4096).all()
        assert (tgt < 4096).all()

        # Next-token alignment check: inp[1:] == tgt[:-1]
        assert torch.equal(inp[1:], tgt[:-1])

        batch_count += 1
        if batch_count >= 10:
            break

    assert batch_count == 10


def test_stage_b_processed_split_disjointness():
    """Verify zero overlap across train, validation, and test processed splits."""
    def load_split_texts(split_dir: Path) -> set:
        texts = set()
        for f in split_dir.glob("*.txt"):
            content = f.read_text(encoding="utf-8")
            for line in content.splitlines():
                if line.strip():
                    texts.add(line.strip())
        return texts

    train_texts = load_split_texts(STAGE_B_PROCESSED / "train")
    val_texts = load_split_texts(STAGE_B_PROCESSED / "validation")
    test_texts = load_split_texts(STAGE_B_PROCESSED / "test")

    assert len(train_texts) > 0
    assert len(val_texts) > 0
    assert len(test_texts) > 0

    assert train_texts.isdisjoint(val_texts), "Train and Validation splits share documents!"
    assert train_texts.isdisjoint(test_texts), "Train and Test splits share documents!"
    assert val_texts.isdisjoint(test_texts), "Validation and Test splits share documents!"
