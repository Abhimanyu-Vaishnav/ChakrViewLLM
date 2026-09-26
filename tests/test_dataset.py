"""
Tests for Sharding and Streaming Dataset Reader (Phase 4, 5 & 18).
"""

from pathlib import Path
import pytest
import torch

from chakrview.training.sharding import (
    ShardWriter,
    read_shard_tokens,
    verify_shard_integrity,
)
from chakrview.training.dataset import StreamingTokenDataset


def test_shard_writer_and_integrity_check(tmp_path: Path):
    """Verify ShardWriter writes valid binary shards and metadata.json."""
    writer = ShardWriter(
        output_dir=tmp_path,
        split_name="train",
        vocab_size=4096,
        max_tokens_per_shard=100,
    )

    # Write 3 documents totaling 250 tokens
    doc1 = list(range(100, 180))  # 80 tokens
    doc2 = list(range(200, 290))  # 90 tokens -> triggers flush for shard 0
    doc3 = list(range(300, 380))  # 80 tokens
    writer.add_document(doc1)
    writer.add_document(doc2)
    writer.add_document(doc3)

    meta = writer.close()
    assert meta["total_tokens"] == 250
    assert meta["total_documents"] == 3
    assert meta["shard_count"] == 3

    # Check file integrity
    train_dir = tmp_path / "train"
    assert (train_dir / "metadata.json").is_file()
    assert verify_shard_integrity(train_dir) is True


def test_streaming_token_dataset_next_token_pairs(tmp_path: Path):
    """Verify StreamingTokenDataset produces exact [x_0..x_{T-1}] and [x_1..x_T] pairs."""
    writer = ShardWriter(
        output_dir=tmp_path,
        split_name="train",
        vocab_size=4096,
        max_tokens_per_shard=1000,
    )
    # Add sequential integers: 0, 1, 2, ..., 129
    tokens = list(range(130))
    writer.add_document(tokens)
    writer.close()

    train_dir = tmp_path / "train"
    seq_len = 16
    dataset = StreamingTokenDataset(
        shard_dir=train_dir,
        sequence_length=seq_len,
        loop=False,
        drop_remainder=True,
    )

    items = list(dataset)
    # 130 tokens // 17 tokens per chunk = 7 full chunks
    assert len(items) == 7

    first_item = items[0]
    inp = first_item["input_ids"]
    tgt = first_item["target_ids"]
    mask = first_item["attention_mask"]

    assert len(inp) == seq_len
    assert len(tgt) == seq_len
    assert len(mask) == seq_len

    # Next-token shift verification
    # inp: [0, 1, 2, ..., 15]
    # tgt: [1, 2, 3, ..., 16]
    assert inp.tolist() == list(range(0, 16))
    assert tgt.tolist() == list(range(1, 17))
    assert (tgt == inp + 1).all()
