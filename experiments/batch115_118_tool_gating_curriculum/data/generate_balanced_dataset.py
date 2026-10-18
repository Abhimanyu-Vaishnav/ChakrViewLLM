"""
Batch 116: Balanced Tool/No-Tool Training Curriculum Dataset Generator.
Builds training, validation, and test partitions containing:
- Positive tool-call examples (addition requests)
- Explicit non-tool clarification examples (out-of-domain)
- Underspecified / malformed requests requiring clarification
Guarantees distinct phrasings, split isolation, and pair-level leakage checks.
"""
import json
import random
from pathlib import Path

DATA_DIR = Path("experiments/batch115_118_tool_gating_curriculum/data")

# 1. Negative & Underspecified Prompts by Split (Ensuring distinct phrasings)
TRAIN_NON_TOOL = [
    # General knowledge / OOD
    ("What is the capital of Japan?", "I am a calculator assistant and only support arithmetic calculations."),
    ("Translate thank you to German.", "I am a calculator assistant and only support arithmetic calculations."),
    ("Who painted the Mona Lisa?", "I am a calculator assistant and only support arithmetic calculations."),
    ("Write a poem about the ocean.", "I am a calculator assistant and only support arithmetic calculations."),
    ("Explain the theory of relativity.", "I am a calculator assistant and only support arithmetic calculations."),
    ("What is the chemical formula for water?", "I am a calculator assistant and only support arithmetic calculations."),
    ("How many continents are there?", "I am a calculator assistant and only support arithmetic calculations."),
    ("Tell me a funny joke.", "I am a calculator assistant and only support arithmetic calculations."),
    # Shell / injection prevention
    ("Run bash script to check disk space.", "Unsupported action. I only support calculator addition."),
    ("Execute Python command os.system('ls').", "Unsupported action. I only support calculator addition."),
    # Underspecified arithmetic
    ("Add the number 5.", "Please specify both numbers to add."),
    ("Calculate the sum of my numbers.", "Please specify both numbers to add."),
    ("What is 7 plus unknown?", "Please provide valid numerical operands for both Operand A and Operand B."),
    ("Add missing operand to 3.", "Please provide valid numerical operands for both Operand A and Operand B."),
    ("Compute addition without values.", "Please specify both numbers to add."),
    ("Can you add these numbers for me?", "Please specify both numbers to add."),
]

VAL_NON_TOOL = [
    ("What is the capital of Italy?", "I am a calculator assistant and only support arithmetic calculations."),
    ("Translate good morning to French.", "I am a calculator assistant and only support arithmetic calculations."),
    ("Execute shell script ping google.com.", "Unsupported action. I only support calculator addition."),
    ("Add number 8 to nothing.", "Please specify both numbers to add."),
]

TEST_NON_TOOL = [
    # Disjoint phrasings for held-out evaluation
    ("What is the capital of Canada?", "I am a calculator assistant and only support arithmetic calculations."),
    ("Translate goodbye to Spanish.", "I am a calculator assistant and only support arithmetic calculations."),
    ("Who wrote the play Macbeth?", "I am a calculator assistant and only support arithmetic calculations."),
    ("Explain the photosynthesis process in plants.", "I am a calculator assistant and only support arithmetic calculations."),
    ("Execute command rm -rf on server.", "Unsupported action. I only support calculator addition."),
    ("Add operand A=6 with missing B.", "Please provide valid numerical operands for both Operand A and Operand B."),
    ("Can you calculate this sum without numbers?", "Please specify both numbers to add."),
    ("Add 4 and unknown quantity.", "Please provide valid numerical operands for both Operand A and Operand B."),
]

def build_tool_record(a: int, b: int, split_name: str, idx: int) -> dict:
    prompt = f"<|im_start|>user\nCalculate sum of:\nA={a}\nB={b}<|im_end|>\n<|im_start|>assistant\n"
    target = f'TOOL_CALL: {{"tool": "calculator", "op": "add", "a": {a}, "b": {b}}}<|im_end|>\n'
    return {
        "id": f"TOOL-{split_name.upper()}-{idx:02d}",
        "operands": [a, b],
        "prompt": prompt,
        "target": target,
        "full_conversation": prompt + target,
        "requires_tool": True,
        "split": split_name,
    }

def build_nontool_record(query: str, response: str, split_name: str, idx: int) -> dict:
    prompt = f"<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n"
    target = f"{response}<|im_end|>\n"
    return {
        "id": f"NONTOOL-{split_name.upper()}-{idx:02d}",
        "prompt": prompt,
        "target": target,
        "full_conversation": prompt + target,
        "requires_tool": False,
        "split": split_name,
    }

def generate_balanced_splits():
    all_pairs = [(a, b) for a in range(10) for b in range(10)]
    
    # 20 held-out test pairs (same split discipline as Batches 108-112)
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
    train_pairs = remaining[16:] # 64 pairs
    
    # Assert zero pair overlap
    assert len(set(train_pairs) & test_set) == 0
    assert len(set(train_pairs) & set(val_pairs)) == 0
    assert len(set(val_pairs) & test_set) == 0
    
    # Assemble splits
    splits = {}
    
    # Train: 64 tool examples + 16 non-tool controls = 80 examples total (80% tool, 20% control)
    train_tool = [build_tool_record(a, b, "train", i) for i, (a, b) in enumerate(train_pairs)]
    train_ctrl = [build_nontool_record(q, r, "train", i) for i, (q, r) in enumerate(TRAIN_NON_TOOL)]
    train_all = train_tool + train_ctrl
    rng.shuffle(train_all)
    splits["train"] = train_all
    
    # Val: 16 tool examples + 4 non-tool controls = 20 examples total
    val_tool = [build_tool_record(a, b, "val", i) for i, (a, b) in enumerate(val_pairs)]
    val_ctrl = [build_nontool_record(q, r, "val", i) for i, (q, r) in enumerate(VAL_NON_TOOL)]
    val_all = val_tool + val_ctrl
    rng.shuffle(val_all)
    splits["val"] = val_all
    
    # Test: 20 tool examples + 8 non-tool controls = 28 examples total
    test_tool = [build_tool_record(a, b, "test", i) for i, (a, b) in enumerate(test_pairs)]
    test_ctrl = [build_nontool_record(q, r, "test", i) for i, (q, r) in enumerate(TEST_NON_TOOL)]
    test_all = test_tool + test_ctrl
    splits["test"] = test_all
    
    return splits, test_pairs

def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    splits, test_pairs = generate_balanced_splits()
    
    counts = {}
    for name, recs in splits.items():
        p = DATA_DIR / f"{name}.jsonl"
        with open(p, "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        n_tool = sum(1 for r in recs if r["requires_tool"])
        n_ctrl = sum(1 for r in recs if not r["requires_tool"])
        counts[name] = {"total": len(recs), "tool_examples": n_tool, "non_tool_controls": n_ctrl}
        print(f"[{name:5s}] Saved {len(recs)} records (tool: {n_tool}, non-tool: {n_ctrl}) to {p}")
        
    manifest = {
        "version": "1.0",
        "description": "Batch 116 Balanced Tool/No-Tool Training Curriculum",
        "seed": 42,
        "counts": counts,
        "held_out_pairs": test_pairs,
    }
    m_p = DATA_DIR / "manifest.json"
    with open(m_p, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest saved to {m_p}")

if __name__ == "__main__":
    main()
