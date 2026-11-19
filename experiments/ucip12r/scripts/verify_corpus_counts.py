"""
UCIP-12R Phase D: Corpus Count Reconciliation Tool.
Derives exact token counts across all binary shards and text datasets in the repository.
"""

import glob
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

datasets = [
    ("Stage C (Train/Val/Test shards)", list((ROOT / "data/tokenized/stage_c").glob("**/*.bin"))),
    ("Stage B (Binary shards)", list((ROOT / "data/tokenized/stage_b").glob("**/*.bin"))),
    ("Curriculum Step 53 (Shards)", list((ROOT / "data/tokenized/curriculum_step53").glob("**/*.bin"))),
    ("Reasoning Step 54 (Shards)", list((ROOT / "data/tokenized/reasoning_step54").glob("**/*.bin"))),
    ("Coding Arena (Shards)", list((ROOT / "data/tokenized/coding_arena").glob("**/*.bin"))),
]

print("=" * 80)
print(f"{'Dataset':<35} | {'Files':<6} | {'Tokens':<12} | {'Bytes':<14} | {'Format'}")
print("=" * 80)

total_tokens = 0
total_bytes = 0

for name, files in datasets:
    sz = sum(f.stat().st_size for f in files)
    toks = sz // 2
    total_tokens += toks
    total_bytes += sz
    print(f"{name:<35} | {len(files):<6} | {toks:<12,d} | {sz:<14,d} | uint16 bin")

print("-" * 80)
print(f"{'TOTAL BINARY TOKENS IN REPOSITORY':<35} | {sum(len(f) for _, f in datasets):<6} | {total_tokens:<12,d} | {total_bytes:<14,d} |")
print("=" * 80)
