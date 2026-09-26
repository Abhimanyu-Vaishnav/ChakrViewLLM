"""
ChakrView Tokenizer: Corpus Engineering & Pipeline Module.

Implements the multi-stage corpus processing pipeline:
    Raw Corpus
        ↓
    Validation (UTF-8, Byte Integrity, no null bytes, no destructive normalization)
        ↓
    Deduplication (category-aware, order-preserving)
        ↓
    Deterministic Splitting (80% Train, 10% Validation, 10% Test)
        ↓
    Disk Artifacts (data/processed/train/, data/validation/, data/processed/test/)

Memory-Conscious Training Support:
Provides chunked and streaming iterators to ensure large corpora never force
full-file RAM loading.
"""

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple, Union


@dataclass(frozen=True)
class CorpusItem:
    """Represents a validated document line within the corpus pipeline."""
    text: str
    category: str
    source_file: str
    line_number: int


class CorpusValidationError(Exception):
    """Raised when a corpus file fails UTF-8 or structural validation."""
    pass


def validate_corpus_file(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Validate that a corpus file contains clean UTF-8 text and no forbidden control bytes.

    Invariants checked:
    - Valid UTF-8 encoding
    - No null bytes (\\x00)
    - Original byte sequences preserved without destructive normalization
    """
    p = Path(file_path)
    if not p.is_file():
        raise FileNotFoundError(f"File not found: {p}")

    raw_bytes = p.read_bytes()
    if b"\x00" in raw_bytes:
        raise CorpusValidationError(f"Null byte detected in corpus file: {p.name}")

    try:
        text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError as e:
        raise CorpusValidationError(f"Invalid UTF-8 in {p.name}: {e}") from e

    lines = [line.rstrip("\r\n") for line in text.splitlines()]
    non_empty = [line for line in lines if line.strip()]

    if not non_empty:
        raise CorpusValidationError(f"Corpus file contains no non-empty text: {p.name}")

    sha256 = hashlib.sha256(raw_bytes).hexdigest()

    return {
        "filename": p.name,
        "byte_count": len(raw_bytes),
        "total_lines": len(lines),
        "non_empty_lines": len(non_empty),
        "sha256": sha256,
        "valid": True,
    }


def deduplicate_lines(lines: List[str]) -> Tuple[List[str], int]:
    """
    Deduplicate a list of text lines, preserving initial appearance order.

    Returns:
        (unique_lines, duplicate_count)
    """
    seen: Set[str] = set()
    unique: List[str] = []
    duplicates = 0

    for line in lines:
        cleaned = line.strip()
        if not cleaned:
            continue
        if cleaned in seen:
            duplicates += 1
        else:
            seen.add(cleaned)
            unique.append(cleaned)

    return unique, duplicates


def hash_split_bucket(text: str, category: str, seed: int = 42) -> float:
    """
    Deterministically hash an item into a float in [0.0, 1.0) using SHA-256.
    Ensures zero random seed ambiguity and strict category-aware stability.
    """
    key = f"{seed}:{category}:{text}".encode("utf-8")
    digest = hashlib.sha256(key).hexdigest()
    # Use first 8 bytes of digest as integer
    int_val = int(digest[:16], 16)
    return int_val / (16**16)


def partition_lines(
    lines: List[str],
    category: str,
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
    seed: int = 42,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Deterministically partition lines into train, validation, and test sets.
    """
    train_lines: List[str] = []
    val_lines: List[str] = []
    test_lines: List[str] = []

    for line in lines:
        bucket = hash_split_bucket(line, category, seed=seed)
        if bucket < train_ratio:
            train_lines.append(line)
        elif bucket < (train_ratio + val_ratio):
            val_lines.append(line)
        else:
            test_lines.append(line)

    # In small sample cases, ensure train split has at least 1 item if lines exist
    if not train_lines and lines:
        train_lines.append(lines[0])
        if lines[0] in val_lines:
            val_lines.remove(lines[0])
        elif lines[0] in test_lines:
            test_lines.remove(lines[0])

    return train_lines, val_lines, test_lines


def execute_corpus_pipeline(
    raw_dir: Union[str, Path],
    processed_dir: Union[str, Path],
    validation_dir: Union[str, Path],
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Execute full pipeline from data/raw/ to data/processed/train, data/validation/, data/processed/test.

    Invariants:
    1. Validation check on every file.
    2. Deduplication.
    3. Category-aware deterministic splitting.
    4. Train, Val, and Test splits are strictly disjoint.
    5. Detailed split manifest generated.
    """
    raw_p = Path(raw_dir)
    proc_p = Path(processed_dir)
    val_p = Path(validation_dir)

    train_out = proc_p / "train"
    val_out = val_p
    test_out = proc_p / "test"

    train_out.mkdir(parents=True, exist_ok=True)
    val_out.mkdir(parents=True, exist_ok=True)
    test_out.mkdir(parents=True, exist_ok=True)

    manifest: Dict[str, Any] = {
        "pipeline_name": "ChakrView Corpus Pipeline (Step 3)",
        "seed": seed,
        "split_ratios": {"train": train_ratio, "val": val_ratio, "test": test_ratio},
        "categories": {},
        "totals": {
            "raw_files_validated": 0,
            "raw_bytes": 0,
            "raw_lines": 0,
            "dedup_lines": 0,
            "train_lines": 0,
            "train_bytes": 0,
            "val_lines": 0,
            "val_bytes": 0,
            "test_lines": 0,
            "test_bytes": 0,
        },
    }

    # Iterate over category directories or files in raw_dir
    categories: Set[str] = set()
    for entry in raw_p.iterdir():
        if entry.is_dir() and not entry.name.startswith("."):
            categories.add(entry.name)

    for cat in sorted(categories):
        cat_dir = raw_p / cat
        cat_raw_lines: List[str] = []
        cat_files_meta: List[Dict[str, Any]] = []
        cat_raw_bytes = 0

        for txt_file in sorted(cat_dir.glob("*.txt")):
            meta = validate_corpus_file(txt_file)
            cat_files_meta.append(meta)
            cat_raw_bytes += meta["byte_count"]
            manifest["totals"]["raw_files_validated"] += 1
            manifest["totals"]["raw_bytes"] += meta["byte_count"]

            text = txt_file.read_text(encoding="utf-8")
            for line in text.splitlines():
                if line.strip():
                    cat_raw_lines.append(line.rstrip("\r\n"))

        manifest["totals"]["raw_lines"] += len(cat_raw_lines)

        # Deduplicate within category
        unique_lines, dup_count = deduplicate_lines(cat_raw_lines)
        manifest["totals"]["dedup_lines"] += len(unique_lines)

        # Split
        train_l, val_l, test_l = partition_lines(
            unique_lines,
            category=cat,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
        )

        # Write to outputs
        def write_lines(target_file: Path, lines: List[str]) -> int:
            content = "\n".join(lines) + ("\n" if lines else "")
            target_file.write_text(content, encoding="utf-8")
            return len(content.encode("utf-8"))

        tr_bytes = write_lines(train_out / f"{cat}.txt", train_l)
        v_bytes = write_lines(val_out / f"{cat}.txt", val_l)
        te_bytes = write_lines(test_out / f"{cat}.txt", test_l)

        manifest["totals"]["train_lines"] += len(train_l)
        manifest["totals"]["train_bytes"] += tr_bytes
        manifest["totals"]["val_lines"] += len(val_l)
        manifest["totals"]["val_bytes"] += v_bytes
        manifest["totals"]["test_lines"] += len(test_l)
        manifest["totals"]["test_bytes"] += te_bytes

        manifest["categories"][cat] = {
            "raw_files": cat_files_meta,
            "raw_bytes": cat_raw_bytes,
            "raw_lines": len(cat_raw_lines),
            "unique_lines": len(unique_lines),
            "duplicates_removed": dup_count,
            "train_lines": len(train_l),
            "train_bytes": tr_bytes,
            "val_lines": len(val_l),
            "val_bytes": v_bytes,
            "test_lines": len(test_l),
            "test_bytes": te_bytes,
        }

    manifest_path = proc_p / "split_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    return manifest


def stream_corpus_lines(
    file_paths: List[Union[str, Path]],
    chunk_size: int = 1000,
) -> Iterator[List[str]]:
    """
    Memory-conscious line streaming iterator.
    Yields batches of lines across multiple files without loading all into RAM.
    """
    batch: List[str] = []
    for fp in file_paths:
        path = Path(fp)
        if not path.is_file():
            continue
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                cleaned = line.rstrip("\r\n")
                if cleaned.strip():
                    batch.append(cleaned)
                    if len(batch) >= chunk_size:
                        yield batch
                        batch = []
    if batch:
        yield batch


def load_corpus_split(
    split_dir: Union[str, Path],
) -> Dict[str, List[CorpusItem]]:
    """
    Load items from a processed split directory (e.g. data/processed/train or data/validation).
    """
    p = Path(split_dir)
    if not p.is_dir():
        raise NotADirectoryError(f"Directory not found: {p}")

    split_items: Dict[str, List[CorpusItem]] = {}
    for txt in sorted(p.glob("*.txt")):
        cat = txt.stem
        lines = txt.read_text(encoding="utf-8").splitlines()
        items: List[CorpusItem] = []
        for idx, line in enumerate(lines, start=1):
            if line.strip():
                items.append(
                    CorpusItem(
                        text=line,
                        category=cat,
                        source_file=txt.name,
                        line_number=idx,
                    )
                )
        split_items[cat] = items

    return split_items
