"""
Tokenized Shard Builder and Format Specification for ChakrView (Phase 4).

Format Specification:
- Binary Shards: Contiguous uint16 little-endian arrays (2 bytes per token for V=4096).
- Metadata: metadata.json storing dataset-level and shard-level metadata:
  - total_tokens, total_documents, vocab_size, tokenizer_checksum,
    creation_timestamp, source_manifest_hash, and per-shard checksums.
"""

import os
import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Generator
import numpy as np


def compute_file_sha256(path: Path | str) -> str:
    """Compute SHA-256 checksum of a file."""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


class ShardWriter:
    """
    Writes sequences of token IDs into compact, sequential binary shards.
    """
    def __init__(
        self,
        output_dir: Path | str,
        split_name: str,
        vocab_size: int = 4096,
        max_tokens_per_shard: int = 250_000,
        tokenizer_checksum: str = "",
        source_manifest_hash: str = "",
    ) -> None:
        self.output_dir = Path(output_dir) / split_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.split_name = split_name
        self.vocab_size = vocab_size
        self.max_tokens_per_shard = max_tokens_per_shard
        self.tokenizer_checksum = tokenizer_checksum
        self.source_manifest_hash = source_manifest_hash

        self.current_shard_idx = 0
        self.current_buffer: List[int] = []
        self.shards_metadata: List[Dict[str, Any]] = []
        self.total_tokens = 0
        self.total_documents = 0

    def add_document(self, token_ids: List[int]) -> None:
        """Add a tokenized document to the shard writer."""
        if not token_ids:
            return
        
        # Verify vocabulary bounds
        for tid in token_ids:
            if tid < 0 or tid >= self.vocab_size:
                raise ValueError(
                    f"Token ID {tid} out of vocabulary range [0, {self.vocab_size - 1}]"
                )

        self.current_buffer.extend(token_ids)
        self.total_tokens += len(token_ids)
        self.total_documents += 1

        while len(self.current_buffer) >= self.max_tokens_per_shard:
            shard_tokens = self.current_buffer[:self.max_tokens_per_shard]
            self._flush_tokens(shard_tokens)
            self.current_buffer = self.current_buffer[self.max_tokens_per_shard:]

    def _flush_tokens(self, tokens: List[int]) -> None:
        """Write token list to binary shard on disk."""
        if not tokens:
            return

        shard_name = f"shard_{self.current_shard_idx:05d}.bin"
        shard_path = self.output_dir / shard_name

        token_array = np.array(tokens, dtype=np.uint16)
        token_array.tofile(shard_path)

        file_hash = compute_file_sha256(shard_path)
        self.shards_metadata.append({
            "filename": shard_name,
            "token_count": len(tokens),
            "byte_size": shard_path.stat().st_size,
            "sha256": file_hash,
        })

        self.current_shard_idx += 1

    def close(self) -> Dict[str, Any]:
        """Flush remaining buffer and write metadata.json."""
        if self.current_buffer:
            self._flush_tokens(self.current_buffer)
            self.current_buffer = []

        metadata = {
            "split": self.split_name,
            "total_tokens": self.total_tokens,
            "total_documents": self.total_documents,
            "shard_count": len(self.shards_metadata),
            "vocab_size": self.vocab_size,
            "token_dtype": "uint16",
            "tokenizer_checksum": self.tokenizer_checksum,
            "source_manifest_hash": self.source_manifest_hash,
            "creation_timestamp": datetime.now(timezone.utc).isoformat(),
            "shards": self.shards_metadata,
        }

        meta_path = self.output_dir / "metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        return metadata


def read_shard_tokens(shard_path: Path | str) -> np.ndarray:
    """Read token IDs from a binary shard as uint16 array."""
    path = Path(shard_path)
    if not path.is_file():
        raise FileNotFoundError(f"Shard file not found: {path}")
    return np.fromfile(path, dtype=np.uint16)


def verify_shard_integrity(shard_dir: Path | str) -> bool:
    """Verify that all shards match checksums listed in metadata.json."""
    dir_path = Path(shard_dir)
    meta_path = dir_path / "metadata.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"metadata.json missing in {dir_path}")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    for s_info in meta.get("shards", []):
        s_path = dir_path / s_info["filename"]
        if not s_path.is_file():
            return False
        if compute_file_sha256(s_path) != s_info["sha256"]:
            return False
    return True
