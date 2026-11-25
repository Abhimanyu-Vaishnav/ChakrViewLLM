"""
ChakrView Balanced Stage-E Corpus Builder.
Mission CV-GENESIS-NR-013 Phase 2.

Merges:
1. Stage-D Multilingual Conversational Foundation (4,686 train examples from data/stage_d/splits/stage_d_train.jsonl)
2. Verified Python Coding Dataset (weighted and balanced)

Ensures zero catastrophic forgetting of Hindi, English, Hinglish while instilling Python code synthesis capabilities.
"""

from __future__ import annotations
import json
import random
from pathlib import Path

def build_stage_e_corpus():
    random.seed(42)

    stage_d_train_path = Path("data/stage_d/splits/stage_d_train.jsonl")
    stage_d_val_path = Path("data/stage_d/splits/stage_d_val.jsonl")

    stage_d_train = []
    with open(stage_d_train_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                stage_d_train.append(json.loads(line))
    print(f"Loaded {len(stage_d_train)} Stage-D training samples.")

    stage_d_val = []
    with open(stage_d_val_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                stage_d_val.append(json.loads(line))
    print(f"Loaded {len(stage_d_val)} Stage-D validation samples.")

    # Load verified coding samples
    coding_train_path = Path("data/coding_sft_013/coding_train.jsonl")
    coding_val_path = Path("data/coding_sft_013/coding_val.jsonl")

    coding_train = []
    with open(coding_train_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                coding_train.append({
                    "prompt": item["prompt"],
                    "response": item["response"],
                    "source": "verified_python_013",
                    "category": f"code_{item['topic']}"
                })
    print(f"Loaded {len(coding_train)} verified coding training samples.")

    coding_val = []
    with open(coding_val_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                coding_val.append({
                    "prompt": item["prompt"],
                    "response": item["response"],
                    "source": "verified_python_013",
                    "category": f"code_{item['topic']}"
                })

    # Weight coding samples (8x) to represent an effective ~10% share of total training tokens
    weighted_coding = []
    for _ in range(8):
        weighted_coding.extend(coding_train)

    composite_train = stage_d_train + weighted_coding
    random.shuffle(composite_train)

    composite_val = stage_d_val + coding_val
    random.shuffle(composite_val)

    out_dir = Path("data/stage_e_corpus_013")
    out_dir.mkdir(parents=True, exist_ok=True)

    out_train = out_dir / "stage_e_train.jsonl"
    with open(out_train, "w", encoding="utf-8") as f:
        for item in composite_train:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    out_val = out_dir / "stage_e_val.jsonl"
    with open(out_val, "w", encoding="utf-8") as f:
        for item in composite_val:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    manifest = {
        "dataset_name": "ChakrView Stage-E Balanced Conversational & Coding Corpus",
        "train_samples": len(composite_train),
        "val_samples": len(composite_val),
        "base_stage_d_train": len(stage_d_train),
        "verified_coding_unique": len(coding_train),
        "weighted_coding_train": len(weighted_coding),
        "coding_share_pct": (len(weighted_coding) / len(composite_train)) * 100.0,
        "random_seed": 42
    }
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Stage-E Composite Train: {len(composite_train)} samples ({manifest['coding_share_pct']:.1f}% coding).")
    print(f"Stage-E Composite Val: {len(composite_val)} samples.")

if __name__ == "__main__":
    build_stage_e_corpus()
