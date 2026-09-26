"""
ChakrView Tokenizer: BPE Trainer Module.

Provides deterministic training of candidate BPE merge tables for empirical
vocabulary experiments (V = 2048, 4096, 8192, 16384).

Mathematical Determinism:
-------------------------
Tie-breaking rule:
    key = (-frequency, pair[0], pair[1])
Primary: Frequency descending.
Secondary: First token ID ascending.
Tertiary: Second token ID ascending.
"""

import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

from chakrview.tokenizer.bpe import create_base_vocab
from chakrview.tokenizer.bytes import (
    BYTE_OFFSET,
    NUM_BYTE_TOKENS,
    byte_seq_to_token_ids,
)
from chakrview.tokenizer.special_tokens import NUM_SPECIAL_TOKENS
from chakrview.tokenizer.tokenizer import BPETokenizer

import re

BASE_VOCAB_SIZE: int = BYTE_OFFSET + NUM_BYTE_TOKENS  # 259

GRAPHEME_REGEX = re.compile(r"[\u0900-\u097F]+|[a-zA-Z]+|[0-9]+|\s+|[\s\S]")


class BPETrainer:
    """
    Deterministic BPE Trainer for research vocabulary experimentation.

    Supports:
    - Raw byte-level BPE (Variant A)
    - Grapheme-aware pre-tokenization + byte-level BPE (Variant B)
    - Memory-conscious streaming / chunked training without full-file RAM loading
    """

    def __init__(
        self,
        min_frequency: int = 1,
        tie_breaking_rule: str = "(-frequency, pair[0], pair[1])",
        pretokenization: str = "byte",
    ) -> None:
        self.min_frequency = min_frequency
        self.tie_breaking_rule = tie_breaking_rule
        self.pretokenization = pretokenization

    def split_segments(self, text: str) -> List[str]:
        """Split text into segments according to pre-tokenization strategy."""
        if self.pretokenization == "grapheme":
            segments = GRAPHEME_REGEX.findall(text)
            return segments if segments else [text]
        return [text]

    def train_stream(
        self,
        chunk_iterator: Iterator[List[Union[str, bytes]]],
        target_vocab_size: int,
        verbose: bool = False,
    ) -> Tuple[Dict[Tuple[int, int], int], Dict[int, bytes], Dict[str, Any]]:
        """
        Train BPE merges from a chunked stream of lines/documents.
        """
        if target_vocab_size <= BASE_VOCAB_SIZE:
            raise ValueError(
                f"target_vocab_size must be > {BASE_VOCAB_SIZE}, got {target_vocab_size}"
            )

        max_merges = target_vocab_size - BASE_VOCAB_SIZE

        # Accumulate token frequency across chunks
        line_counts: Dict[Tuple[int, ...], int] = Counter()
        total_initial_bytes = 0

        for chunk in chunk_iterator:
            for item in chunk:
                if isinstance(item, bytes):
                    if not item:
                        continue
                    total_initial_bytes += len(item)
                    token_tuple = tuple(byte_seq_to_token_ids(item))
                    line_counts[token_tuple] += 1
                else:
                    if not item:
                        continue
                    segments = self.split_segments(item)
                    for seg in segments:
                        raw_bytes = seg.encode("utf-8")
                        if not raw_bytes:
                            continue
                        total_initial_bytes += len(raw_bytes)
                        token_tuple = tuple(byte_seq_to_token_ids(raw_bytes))
                        line_counts[token_tuple] += 1

        vocab = create_base_vocab()
        merges: Dict[Tuple[int, int], int] = {}

        t0 = time.perf_counter()
        current_token_id = BASE_VOCAB_SIZE
        last_pair_freq = 0

        for step in range(max_merges):
            # 1. Count adjacent pairs across unique sequences
            pair_counts: Dict[Tuple[int, int], int] = defaultdict(int)
            for seq, count in line_counts.items():
                seq_len = len(seq)
                if seq_len < 2:
                    continue
                for i in range(seq_len - 1):
                    pair_counts[(seq[i], seq[i + 1])] += count

            if not pair_counts:
                # No more adjacent pairs exist in corpus
                break

            # 2. Select best pair using deterministic tie-breaking
            best_pair = min(
                pair_counts.keys(),
                key=lambda p: (-pair_counts[p], p[0], p[1]),
            )
            freq = pair_counts[best_pair]

            if freq < self.min_frequency:
                # Exhausted pairs meeting minimum frequency threshold
                break

            last_pair_freq = freq
            p0, p1 = best_pair

            # 3. Record merge and update vocabulary
            merges[best_pair] = current_token_id
            vocab[current_token_id] = vocab[p0] + vocab[p1]

            # 4. Apply merge to sequences
            new_line_counts: Dict[Tuple[int, ...], int] = Counter()
            for seq, count in line_counts.items():
                if p0 in seq and p1 in seq:
                    new_seq: List[int] = []
                    i = 0
                    n = len(seq)
                    while i < n:
                        if i < n - 1 and seq[i] == p0 and seq[i + 1] == p1:
                            new_seq.append(current_token_id)
                            i += 2
                        else:
                            new_seq.append(seq[i])
                            i += 1
                    new_line_counts[tuple(new_seq)] += count
                else:
                    new_line_counts[seq] += count

            line_counts = new_line_counts
            current_token_id += 1

            if verbose and (step + 1) % 500 == 0:
                print(f"Merge step {step + 1}/{max_merges}: pair={best_pair}, freq={freq}")

        training_time = time.perf_counter() - t0
        actual_vocab_size = BASE_VOCAB_SIZE + len(merges)

        final_tokens = sum(len(seq) * count for seq, count in line_counts.items())

        train_stats: Dict[str, Any] = {
            "requested_vocab_size": target_vocab_size,
            "actual_vocab_size": actual_vocab_size,
            "merges_performed": len(merges),
            "max_merges_target": max_merges,
            "training_time_seconds": round(training_time, 4),
            "total_initial_bytes": total_initial_bytes,
            "final_training_tokens": final_tokens,
            "compression_ratio": round(total_initial_bytes / final_tokens, 4) if final_tokens > 0 else 0.0,
            "last_pair_frequency": last_pair_freq,
            "tie_breaking_rule": self.tie_breaking_rule,
            "pretokenization": self.pretokenization,
        }

        return merges, vocab, train_stats

    def train(
        self,
        corpus_data: List[Union[str, bytes]],
        target_vocab_size: int,
        verbose: bool = False,
    ) -> Tuple[Dict[Tuple[int, int], int], Dict[int, bytes], Dict[str, Any]]:
        """
        Train BPE merges on in-memory list of items.
        """
        return self.train_stream(iter([corpus_data]), target_vocab_size, verbose=verbose)


def save_experiment_artifacts(
    output_dir: Union[str, Path],
    target_vocab_size: int,
    merges: Dict[Tuple[int, int], int],
    vocab: Dict[int, bytes],
    train_stats: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Save trained BPE experiment artifacts to a dedicated directory.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Save merges.json (convert tuple keys to string format "p0,p1")
    merges_serializable = {f"{p[0]},{p[1]}": tid for p, tid in merges.items()}
    with (out_path / "merges.json").open("w", encoding="utf-8") as f:
        json.dump(merges_serializable, f, indent=2)

    # 2. Save vocab.json (convert bytes to hex string for unambiguous serialization)
    vocab_serializable = {str(tid): b.hex() for tid, b in vocab.items()}
    with (out_path / "vocab.json").open("w", encoding="utf-8") as f:
        json.dump(vocab_serializable, f, indent=2)

    # 3. Save train_stats.json
    with (out_path / "train_stats.json").open("w", encoding="utf-8") as f:
        json.dump(train_stats, f, indent=2)

    # 4. Save metadata.json
    meta = metadata or {}
    meta_full: Dict[str, Any] = {
        "experiment_name": f"v{target_vocab_size}",
        "target_vocab_size": target_vocab_size,
        "actual_vocab_size": train_stats.get("actual_vocab_size", BASE_VOCAB_SIZE + len(merges)),
        "merges_count": len(merges),
        "base_bytes_count": NUM_BYTE_TOKENS,
        "special_tokens_count": NUM_SPECIAL_TOKENS,
        "tie_breaking_rule": train_stats.get("tie_breaking_rule", "(-frequency, pair[0], pair[1])"),
        "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    meta_full.update(meta)
    with (out_path / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(meta_full, f, indent=2)

    return out_path


def load_experiment_artifacts(experiment_dir: Union[str, Path]) -> BPETokenizer:
    """
    Load a trained experiment directory into a BPETokenizer instance.
    """
    exp_path = Path(experiment_dir)
    if not exp_path.is_dir():
        raise FileNotFoundError(f"Experiment directory not found: {exp_path}")

    with (exp_path / "merges.json").open("r", encoding="utf-8") as f:
        raw_merges = json.load(f)
    merges: Dict[Tuple[int, int], int] = {}
    for pair_str, tid in raw_merges.items():
        p0_str, p1_str = pair_str.split(",")
        merges[(int(p0_str), int(p1_str))] = tid

    with (exp_path / "vocab.json").open("r", encoding="utf-8") as f:
        raw_vocab = json.load(f)
    vocab: Dict[int, bytes] = {int(tid): bytes.fromhex(hex_str) for tid, hex_str in raw_vocab.items()}

    return BPETokenizer(merges=merges, vocab=vocab)
