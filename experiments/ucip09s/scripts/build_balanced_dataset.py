"""
Builds Condition C Balanced Replay Dataset for UCIP-09S.
Mixture:
- 70% Anchor Replay from UCIP-09:
  - English conversation & identity (240)
  - Devanagari Hindi (240)
  - Hinglish & Explanations (200)
  - Instruction following (160)
  Total Anchor = 840
- 30% Targeted Capability Data:
  - Diverse canonical arithmetic pairs (240)
  - Exact copying pairs (120)
  Total Targeted = 360
Total Dataset = 1,200 samples.

Validation split: 120 balanced samples (70% anchor / 30% targeted).
"""

import json
import random
from pathlib import Path

REPO_ROOT = Path(r"D:\Project\ChakrView")

def main():
    random.seed(42)

    # 1. Load UCIP-09 source data
    u09_train_path = REPO_ROOT / "experiments" / "cognition" / "ucip_09" / "data" / "ucip09_train.jsonl"
    u09_val_path = REPO_ROOT / "experiments" / "cognition" / "ucip_09" / "data" / "ucip09_validation.jsonl"
    
    u09_records = []
    for p in [u09_train_path, u09_val_path]:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    u09_records.append(json.loads(line))

    # Bin UCIP-09 by category
    by_cat = {}
    for r in u09_records:
        cat = r.get("category", "")
        by_cat.setdefault(cat, []).append(r)

    print("UCIP-09 available categories:", {k: len(v) for k, v in by_cat.items()})

    for k in by_cat:
        random.shuffle(by_cat[k])

    eng_conv = by_cat.get("English conversation", [])[:240]
    dev_hin = by_cat.get("Devanagari Hindi", [])[:240]
    hing_exp = (by_cat.get("Hinglish", [])[:100] + by_cat.get("English explanations", [])[:100])
    inst_fol = by_cat.get("Instruction following", [])[:160]

    anchor_samples = eng_conv + dev_hin + hing_exp + inst_fol
    print(f"Selected {len(anchor_samples)} anchor samples (Target: 840).")

    # 2. Curate Targeted Arithmetic (Diverse operations, diverse numbers, prevent 21/22 skew)
    u09r_train_path = REPO_ROOT / "experiments" / "cognition" / "ucip_09r" / "data" / "ucip09r_train.jsonl"
    u09r_records = []
    with open(u09r_train_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                u09r_records.append(json.loads(line))

    u09r_arith = [r for r in u09r_records if r.get("category") == "Arithmetic"]
    u09r_copy = [r for r in u09r_records if r.get("category") == "Exact copying"]

    # Filter arithmetic to ensure diversity and avoid repetitive single numbers
    random.shuffle(u09r_arith)
    random.shuffle(u09r_copy)

    # Balance arithmetic across addition, subtraction, word problems
    selected_arith = []
    num_counts = {}
    for r in u09r_arith:
        tgt = r.get("target", "")
        words = tgt.split()
        # Find numeric tokens in target
        nums = [w for w in words if w.isdigit()]
        if any(num_counts.get(n, 0) >= 10 for n in nums):
            continue
        for n in nums:
            num_counts[n] = num_counts.get(n, 0) + 1
        selected_arith.append(r)
        if len(selected_arith) >= 240:
            break

    # If still need more, take remaining
    if len(selected_arith) < 240:
        for r in u09r_arith:
            if r not in selected_arith:
                selected_arith.append(r)
                if len(selected_arith) >= 240:
                    break

    selected_copy = u09r_copy[:120]
    # If copy is less than 120, take from ucip09 copying
    if len(selected_copy) < 120:
        u09_copy = by_cat.get("Exact copying", [])
        for r in u09_copy:
            if r not in selected_copy:
                selected_copy.append(r)
                if len(selected_copy) >= 120:
                    break

    targeted_samples = selected_arith + selected_copy
    print(f"Selected {len(targeted_samples)} targeted samples ({len(selected_arith)} arith, {len(selected_copy)} copy).")

    full_dataset = anchor_samples + targeted_samples
    random.shuffle(full_dataset)
    print(f"Total Combined Dataset: {len(full_dataset)} samples.")

    # Split 1,200 train and 120 val
    val_set = full_dataset[:120]
    train_set = full_dataset[120:1200+120]

    out_dir = REPO_ROOT / "experiments" / "ucip09s" / "datasets"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_out = out_dir / "ucip09s_balanced_train.jsonl"
    val_out = out_dir / "ucip09s_balanced_val.jsonl"

    with open(train_out, "w", encoding="utf-8") as f:
        for r in train_set:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(val_out, "w", encoding="utf-8") as f:
        for r in val_set:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Saved {len(train_set)} train samples to {train_out}")
    print(f"Saved {len(val_set)} val samples to {val_out}")

if __name__ == "__main__":
    main()
