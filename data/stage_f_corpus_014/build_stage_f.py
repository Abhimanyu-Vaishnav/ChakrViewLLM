"""
ChakrView Stage-F Balanced Multi-Domain Corpus Builder.
Mission CV-GENESIS-NR-014 Phase 4.

Merges:
1. Stage-D Multilingual Conversational Foundation (4,686 train, 551 val)
2. Expanded Verified Python Coding Dataset (234 unique tasks, weighted 3x to create 702 samples, representing ~13.0% balanced share)

Prevents catastrophic conversational regression while conditioning the network on rich, diverse algorithmic syntax.
"""

from __future__ import annotations
import json
import random
from pathlib import Path

def build_stage_f():
    random.seed(42)

    # 1. Load Stage-D Multilingual Base
    stage_d_train_p = Path("data/stage_d/splits/stage_d_train.jsonl")
    stage_d_val_p = Path("data/stage_d/splits/stage_d_val.jsonl")

    stage_d_train = []
    with open(stage_d_train_p, "r", encoding="utf-8") as f:
        for l in f:
            if l.strip(): stage_d_train.append(json.loads(l))

    stage_d_val = []
    with open(stage_d_val_p, "r", encoding="utf-8") as f:
        for l in f:
            if l.strip(): stage_d_val.append(json.loads(l))

    # 2. Load Expanded Coding Dataset
    coding_train_p = Path("data/coding_sft_014/coding_train_expanded.jsonl")
    coding_val_p = Path("data/coding_sft_014/coding_val_014.jsonl")

    coding_train = []
    with open(coding_train_p, "r", encoding="utf-8") as f:
        for l in f:
            if l.strip():
                item = json.loads(l)
                coding_train.append({
                    "prompt": item["prompt"],
                    "response": item["response"],
                    "category": f"code_{item['topic']}",
                    "source": "verified_python_014"
                })

    coding_val = []
    with open(coding_val_p, "r", encoding="utf-8") as f:
        for l in f:
            if l.strip():
                item = json.loads(l)
                coding_val.append({
                    "prompt": item["prompt"],
                    "response": item["response"],
                    "category": f"code_{item['topic']}",
                    "source": "verified_python_014"
                })

    # Weighted coding: 3x repetitions (702 samples)
    weighted_coding = []
    for _ in range(3):
        weighted_coding.extend(coding_train)

    composite_train = stage_d_train + weighted_coding
    random.shuffle(composite_train)

    composite_val = stage_d_val + coding_val
    random.shuffle(composite_val)

    out_dir = Path("data/stage_f_corpus_014")
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "stage_f_train.jsonl", "w", encoding="utf-8") as f:
        for s in composite_train:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    with open(out_dir / "stage_f_val.jsonl", "w", encoding="utf-8") as f:
        for s in composite_val:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    manifest = {
        "dataset_name": "ChakrView Stage-F Balanced Multi-Domain Corpus (NR-014)",
        "total_train_samples": len(composite_train),
        "stage_d_train_samples": len(stage_d_train),
        "coding_train_unique": len(coding_train),
        "coding_weighted_samples": len(weighted_coding),
        "coding_share_pct": (len(weighted_coding) / len(composite_train)) * 100.0,
        "total_val_samples": len(composite_val),
        "random_seed": 42
    }
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Stage-F Composite Train: {len(composite_train)} samples ({manifest['coding_share_pct']:.1f}% coding).")
    print(f"Stage-F Composite Val: {len(composite_val)} samples.")

if __name__ == "__main__":
    build_stage_f()
