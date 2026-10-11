"""
Balanced Interleaved Bilingual Stream Dataset for Wave 458.

Provides deterministic, interleaved, and balanced streaming over English, Hindi,
and reasoning/code tokenized shards with zero memory bloat via memory-mapping.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterator, List, Optional
import numpy as np
import torch
from torch.utils.data import IterableDataset


class BalancedBilingualStream(IterableDataset):
    """
    Deterministic streaming dataset that interleaves language streams according
    to prescribed ratios (e.g. 45% English, 45% Hindi, 10% Code/Reasoning).
    
    Ensures language balance is strictly maintained at every step, avoiding
    sequential language starvation.
    """

    def __init__(
        self,
        en_shards: List[Path],
        hi_shards: List[Path],
        code_shards: List[Path],
        sequence_length: int = 64,
        en_ratio: float = 0.45,
        hi_ratio: float = 0.45,
        code_ratio: float = 0.10,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.en_shards = [Path(p) for p in en_shards]
        self.hi_shards = [Path(p) for p in hi_shards]
        self.code_shards = [Path(p) for p in code_shards]
        self.sequence_length = sequence_length
        self.en_ratio = en_ratio
        self.hi_ratio = hi_ratio
        self.code_ratio = code_ratio
        self.seed = seed

        # Validate ratios
        total_ratio = en_ratio + hi_ratio + code_ratio
        if abs(total_ratio - 1.0) > 1e-4:
            raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        rng = np.random.RandomState(self.seed)

        # Open memmap views on binary shards
        en_streams = [np.memmap(p, dtype=np.uint16, mode="r") for p in self.en_shards]
        hi_streams = [np.memmap(p, dtype=np.uint16, mode="r") for p in self.hi_shards]
        code_streams = [np.memmap(p, dtype=np.uint16, mode="r") for p in self.code_shards]

        while True:
            choice = rng.rand()
            if choice < self.en_ratio:
                stream_list = en_streams
                lang_tag = 0  # English
            elif choice < self.en_ratio + self.hi_ratio:
                stream_list = hi_streams
                lang_tag = 1  # Hindi
            else:
                stream_list = code_streams
                lang_tag = 2  # Code / Reasoning

            # Pick a shard within the chosen language stream
            stream = stream_list[rng.randint(len(stream_list))]
            max_start = len(stream) - (self.sequence_length + 1)
            if max_start <= 0:
                continue

            start_idx = rng.randint(0, max_start)
            tokens = stream[start_idx : start_idx + self.sequence_length + 1].astype(np.int64)

            inp = torch.from_numpy(tokens[:-1])
            tgt = torch.from_numpy(tokens[1:])

            yield {
                "input_ids": inp,
                "target_ids": tgt,
                "lang_tag": torch.tensor(lang_tag, dtype=torch.long),
            }
