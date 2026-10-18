import json
import random
from pathlib import Path

DATA_DIR = Path("experiments/batch111_114_curriculum_transfer/data")

def build_tool_record(a: int, b: int, split_name: str, idx: int) -> dict:
    prompt = f"<|im_start|>user\nCalculate sum of:\nA={a}\nB={b}<|im_end|>\n<|im_start|>assistant\n"
    target = f'TOOL_CALL: {{"tool": "calculator", "op": "add", "a": {a}, "b": {b}}}<|im_end|>\n'
    full_conv = prompt + target
    return {
        "id": f"TOOL-{split_name.upper()}-{idx:02d}",
        "operands": [a, b],
        "prompt": prompt,
        "target": target,
        "full_conversation": full_conv,
        "type": "tool_call",
        "split": split_name,
    }

def build_clarification_controls() -> list:
    queries = [
        ("What is the capital of France?", "I am a calculator assistant and only support arithmetic calculations."),
        ("Translate hello to Spanish.", "I am a calculator assistant and only support arithmetic calculations."),
        ("Write a Python script to list files.", "Unsupported action. I only support calculator addition."),
        ("Explain quantum mechanics.", "I am a calculator assistant and only support arithmetic calculations."),
        ("Who wrote Hamlet?", "I am a calculator assistant and only support arithmetic calculations."),
    ]
    controls = []
    for i, (q, ans) in enumerate(queries):
        prompt = f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"
        target = f"{ans}<|im_end|>\n"
        controls.append({
            "id": f"CTRL-{i:02d}",
            "prompt": prompt,
            "target": target,
            "full_conversation": prompt + target,
            "type": "clarification",
            "requires_tool": False,
        })
    return controls

def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    all_pairs = [(a, b) for a in range(10) for b in range(10)]
    
    # Identical held-out test set as Batch 108
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
    
    splits = {
        "train": [build_tool_record(a, b, "train", i) for i, (a, b) in enumerate(train_pairs)],
        "val": [build_tool_record(a, b, "val", i) for i, (a, b) in enumerate(val_pairs)],
        "test": [build_tool_record(a, b, "test", i) for i, (a, b) in enumerate(test_pairs)],
    }
    
    controls = build_clarification_controls()
    
    # Save splits
    for name, recs in splits.items():
        p = DATA_DIR / f"{name}.jsonl"
        with open(p, "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        print(f"Saved {len(recs)} records to {p}")
        
    ctrl_p = DATA_DIR / "clarification_controls.jsonl"
    with open(ctrl_p, "w", encoding="utf-8") as f:
        for r in controls:
            f.write(json.dumps(r) + "\n")
    print(f"Saved {len(controls)} clarification controls to {ctrl_p}")
    
    manifest = {
        "dataset": "ChakrView Tool Argument Transfer Benchmark",
        "counts": {k: len(v) for k, v in splits.items()},
        "clarification_controls": len(controls),
        "held_out_test_pairs": test_pairs,
    }
    with open(DATA_DIR / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

if __name__ == "__main__":
    main()
