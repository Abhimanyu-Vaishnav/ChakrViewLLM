"""
Streaming Dataset Reader for Tokenized Shards (Phase 5).

Reads tokenized binary shards sequentially without loading the complete corpus into RAM.
Generates next-token prediction pairs:
    sequence [x0, x1, ..., xT] ->
    input_ids  [x0, x1, ..., x_{T-1}]
    target_ids [x1, x2, ..., xT]
"""

import json
import random
from pathlib import Path
from typing import Iterator, Dict, Any, List, Optional
import numpy as np
import torch
from torch.utils.data import IterableDataset

from chakrview.training.sharding import read_shard_tokens


class StreamingTokenDataset(IterableDataset):
    """
    Streaming dataset reading binary token shards on demand.
    
    Attributes:
        shard_dir: Directory containing shard_*.bin and metadata.json.
        sequence_length: Target sequence length T for input_ids and target_ids.
        shuffle: Whether to shuffle shard order per iteration.
        seed: Random seed for deterministic shard shuffling.
        loop: Whether to loop infinitely across shards (for continuous training steps).
        drop_remainder: If True, discard partial final blocks smaller than sequence_length + 1.
        pad_token_id: Token ID used for padding if drop_remainder=False.
    """
    def __init__(
        self,
        shard_dir: Path | str,
        sequence_length: int = 512,
        shuffle: bool = False,
        seed: int = 42,
        loop: bool = True,
        drop_remainder: bool = True,
        pad_token_id: int = 2,
    ) -> None:
        super().__init__()
        self.shard_dir = Path(shard_dir)
        self.sequence_length = sequence_length
        self.shuffle = shuffle
        self.seed = seed
        self.loop = loop
        self.drop_remainder = drop_remainder
        self.pad_token_id = pad_token_id

        # Discover shard files
        self.metadata = self._load_metadata()
        self.shard_files = self._get_shard_files()
        if not self.shard_files:
            raise FileNotFoundError(f"No shard binary files found in {self.shard_dir}")

    def _load_metadata(self) -> Dict[str, Any]:
        meta_path = self.shard_dir / "metadata.json"
        if meta_path.is_file():
            with open(meta_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _get_shard_files(self) -> List[Path]:
        if "shards" in self.metadata and self.metadata["shards"]:
            files = [self.shard_dir / s["filename"] for s in self.metadata["shards"]]
        else:
            files = sorted(list(self.shard_dir.glob("shard_*.bin")))
        return [f for f in files if f.is_file()]

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        rng = random.Random(self.seed)
        chunk_size = self.sequence_length + 1  # 1 extra token for next-token target

        while True:
            shards = list(self.shard_files)
            if self.shuffle:
                rng.shuffle(shards)

            for shard_path in shards:
                tokens = read_shard_tokens(shard_path)
                n_tokens = len(tokens)
                if n_tokens < 2:
                    continue

                # Stream full chunks
                num_full_chunks = n_tokens // chunk_size
                for i in range(num_full_chunks):
                    start_idx = i * chunk_size
                    end_idx = start_idx + chunk_size
                    chunk = tokens[start_idx:end_idx]

                    input_ids = torch.from_numpy(chunk[:-1].astype(np.int64))
                    target_ids = torch.from_numpy(chunk[1:].astype(np.int64))
                    mask = torch.ones(self.sequence_length, dtype=torch.int64)

                    yield {
                        "input_ids": input_ids,
                        "target_ids": target_ids,
                        "attention_mask": mask,
                    }

                # Handle remainder
                remainder_tokens = tokens[num_full_chunks * chunk_size:]
                if len(remainder_tokens) >= 2 and not self.drop_remainder:
                    # Pad remainder up to chunk_size
                    pad_len = chunk_size - len(remainder_tokens)
                    padded = np.pad(
                        remainder_tokens,
                        (0, pad_len),
                        mode="constant",
                        constant_values=self.pad_token_id,
                    )
                    input_ids = torch.from_numpy(padded[:-1].astype(np.int64))
                    target_ids = torch.from_numpy(padded[1:].astype(np.int64))
                    mask = torch.zeros(self.sequence_length, dtype=torch.int64)
                    mask[:len(remainder_tokens) - 1] = 1

                    yield {
                        "input_ids": input_ids,
                        "target_ids": target_ids,
                        "attention_mask": mask,
                    }

            if not self.loop:
                break
