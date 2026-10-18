import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

DATA_DIR = Path("experiments/batch107_110_operand_copy_benchmark/data")

def build_record(a: int, b: int, split_name: str, idx: int) -> Dict:
    prompt = f"<|im_start|>user\nCopy:\nA={a}\nB={b}<|im_end|>\n<|im_start|>assistant\n"
    target = f"A={a}\nB={b}<|im_end|>\n"
    full_conv = prompt + target
    return {
        "id": f"COPY-{split_name.upper()}-{idx:02d}",
        "operands": [a, b],
        "prompt": prompt,
        "target": target,
        "full_conversation": full_conv,
        "split": split_name,
    }

def generate_splits() -> Dict[str, List[Dict]]:
    all_pairs = [(a, b) for a in range(10) for b in range(10)]
    
    # 20 held-out test pairs: repeated, reverse pairs, and distinct coverage
    test_pairs = [
        (0, 0), (9, 9), (4, 4),
        (1, 8), (8, 1),
        (2, 7), (7, 2),
        (3, 6), (6, 3),
        (0, 5), (5, 0),
        (1, 4), (4, 1),
        (2, 9), (9, 2),
        (3, 5), (5, 3),
        (6, 8), (8, 6),
        (7, 9)
    ]
    test_set = set(test_pairs)
    remaining = [p for p in all_pairs if p not in test_set]
    
    rng = random.Random(42)
    rng.shuffle(remaining)
    val_pairs = remaining[:16]
    train_pairs = remaining[16:]
    
    # Assert zero overlap
    train_set = set(train_pairs)
    val_set = set(val_pairs)
    assert len(train_set & test_set) == 0, "Train-test leakage detected!"
    assert len(train_set & val_set) == 0, "Train-val leakage detected!"
    assert len(val_set & test_set) == 0, "Val-test leakage detected!"
    assert len(train_pairs) + len(val_pairs) + len(test_pairs) == 100, "Incomplete pair coverage!"
    
    splits = {
        "train": [build_record(a, b, "train", i) for i, (a, b) in enumerate(train_pairs)],
        "val": [build_record(a, b, "val", i) for i, (a, b) in enumerate(val_pairs)],
        "test": [build_record(a, b, "test", i) for i, (a, b) in enumerate(test_pairs)],
    }
    return splits

def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    splits = generate_splits()
    manifest = {
        "dataset": "ChakrView Operand Copy Benchmark",
        "format": "A={a}\nB={b}",
        "seed": 42,
        "counts": {k: len(v) for k, v in splits.items()},
        "total_pairs": 100,
        "zero_leakage_verified": True,
    }
    for name, recs in splits.items():
        p = DATA_DIR / f"{name}.jsonl"
        with open(p, "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        print(f"Saved {len(recs)} records to {p}")
        
    m_path = DATA_DIR / "manifest.json"
    with open(m_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved manifest to {m_path}")

if __name__ == "__main__":
    main()
