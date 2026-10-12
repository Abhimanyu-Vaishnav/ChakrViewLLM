"""
Deterministic multi-digit prompt-diverse dataset generator for Batches 123–126.
Generates:
- 8 training templates
- 2 held-out evaluation templates
- Controlled splits:
  - train (70% tool, 30% negative controls)
  - val
  - test partitioned into:
    * test_familiar_pairs_unseen_templates
    * test_unseen_pairs_familiar_templates
    * test_unseen_pairs_unseen_templates
    * test_negative_controls
"""
import json
import random
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Tuple

OUTPUT_DIR = Path("experiments/batch123_126_prompt_diverse_multidigit/data")

# 8 distinct training templates
TRAIN_TEMPLATES = [
    "Calculate sum of:\nA={a}\nB={b}",
    "Please add A={a} and B={b}.",
    "Compute addition for A={a} and B={b}.",
    "Find the total sum of A={a} and B={b}.",
    "Sum operands: A={a}, B={b}.",
    "Perform add operation on A={a} and B={b}.",
    "Calculate total: A={a} + B={b}.",
    "What is the sum of A={a} and B={b}?",
]

# 2 strictly held-out evaluation templates (never in train or val)
HELD_OUT_TEMPLATES = [
    "Execute calculator addition: first={a}, second={b}.",
    "Determine arithmetic total for inputs: A is {a} and B is {b}.",
]

NEGATIVE_PROMPTS_TRAIN = [
    "What is the capital of Japan?",
    "Explain the concept of entropy in physics.",
    "Translate the word 'ocean' into German.",
    "Summarize the plot of the Odyssey.",
    "Write a short python function to reverse a string.",
    "Calculate sum of numbers without giving any operands.",
    "Add operand A=25 with missing second value.",
    "Execute addition with unknown operand B.",
    "Compute calculator total for null arguments.",
    "Find the sum of undefined variables.",
]

NEGATIVE_PROMPTS_TEST = [
    "What is the speed of light in vacuum?",
    "Describe the process of cellular mitosis.",
    "Translate 'good morning' into Japanese.",
    "Who painted the ceiling of the Sistine Chapel?",
    "What is the function of an operating system kernel?",
    "Add two numbers when neither operand is specified.",
    "Perform addition on A=77 with no operand B.",
    "Calculate the total when operand A is missing.",
    "Invoke the calculator on blank input.",
    "What is the arithmetic result of unknown numbers?",
]

def format_record(ex_id: str, prompt_text: str, is_tool: bool, a: int = None, b: int = None, category: str = "tool") -> dict:
    prompt = f"<|im_start|>user\n{prompt_text}<|im_end|>\n<|im_start|>assistant\n"
    if is_tool:
        expected_call = f'TOOL_CALL: {{"tool": "calculator", "op": "add", "a": {a}, "b": {b}}}'
        full = f"{prompt}{expected_call}<|im_end|>"
        return {
            "id": ex_id,
            "category": category,
            "requires_tool": True,
            "operands": [a, b],
            "prompt": prompt,
            "expected_call": expected_call,
            "full_conversation": full,
        }
    else:
        refusal = "I can only invoke the calculator tool when both valid operands A and B are provided."
        full = f"{prompt}{refusal}<|im_end|>"
        return {
            "id": ex_id,
            "category": category,
            "requires_tool": False,
            "operands": None,
            "prompt": prompt,
            "expected_call": None,
            "full_conversation": full,
        }

def build_dataset(seed: int = 42):
    rng = random.Random(seed)
    
    # Generate pool of 2-digit pairs: 10..99
    # To keep experiments bounded, we select structured subsets
    all_pairs = [(a, b) for a in range(10, 100) for b in range(10, 100)]
    rng.shuffle(all_pairs)
    
    # 1. Training pairs pool (60 distinct ordered pairs)
    train_pairs_pool = all_pairs[:60]
    # Reserve reversed pairs check
    train_pairs_set = set(train_pairs_pool)
    
    # 2. Unseen test pairs pool (strictly disjoint from train pairs and their reverse)
    remaining_pairs = [
        p for p in all_pairs[60:]
        if p not in train_pairs_set and (p[1], p[0]) not in train_pairs_set
    ]
    unseen_test_pairs = remaining_pairs[:30]
    
    # Build Train Set:
    # 70 tool examples (using the 8 train templates) + 30 negative controls
    train_records = []
    # Sample 70 tool instances from the 60 train pairs across 8 templates
    for i in range(70):
        a, b = train_pairs_pool[i % len(train_pairs_pool)]
        tmpl = TRAIN_TEMPLATES[i % len(TRAIN_TEMPLATES)]
        prompt_text = tmpl.format(a=a, b=b)
        train_records.append(format_record(f"TRAIN-TOOL-{i:03d}", prompt_text, True, a, b, "train_tool"))
        
    for j, neg in enumerate(NEGATIVE_PROMPTS_TRAIN * 3): # 30 negative controls
        train_records.append(format_record(f"TRAIN-NEG-{j:03d}", neg, False, category="train_negative_control"))
        
    rng.shuffle(train_records)
    
    # Build Validation Set:
    # 15 tool examples (train pairs, train templates) + 5 negative controls
    val_records = []
    for i in range(15):
        a, b = train_pairs_pool[(i + 20) % len(train_pairs_pool)]
        tmpl = TRAIN_TEMPLATES[(i + 3) % len(TRAIN_TEMPLATES)]
        prompt_text = tmpl.format(a=a, b=b)
        val_records.append(format_record(f"VAL-TOOL-{i:03d}", prompt_text, True, a, b, "val_tool"))
    for j, neg in enumerate(NEGATIVE_PROMPTS_TRAIN[:5]):
        val_records.append(format_record(f"VAL-NEG-{j:03d}", neg, False, category="val_negative_control"))
        
    # Build Test Set Partitioned into 4 rigorously isolated sub-benchmarks:
    test_records = []
    
    # Partition 1: Familiar Pairs with Held-Out (Unseen) Templates (10 examples)
    # Tests if known number pairs can be bound when syntax is novel
    for i in range(10):
        a, b = train_pairs_pool[i]
        tmpl = HELD_OUT_TEMPLATES[i % len(HELD_OUT_TEMPLATES)]
        prompt_text = tmpl.format(a=a, b=b)
        test_records.append(format_record(f"TEST-FAMPAIR-UNSEENTMPL-{i:02d}", prompt_text, True, a, b, "familiar_pairs_unseen_templates"))
        
    # Partition 2: Unseen Pairs with Familiar Templates (15 examples)
    # Tests if unseen number pairs can be bound within known syntax
    for i in range(15):
        a, b = unseen_test_pairs[i]
        tmpl = TRAIN_TEMPLATES[i % len(TRAIN_TEMPLATES)]
        prompt_text = tmpl.format(a=a, b=b)
        test_records.append(format_record(f"TEST-UNSEENPAIR-FAMTMPL-{i:02d}", prompt_text, True, a, b, "unseen_pairs_familiar_templates"))

    # Partition 3: Unseen Pairs with Held-Out Templates (10 examples)
    # The ultimate double-generalization test: both numbers and syntax are unseen
    for i in range(10):
        a, b = unseen_test_pairs[15 + i]
        tmpl = HELD_OUT_TEMPLATES[i % len(HELD_OUT_TEMPLATES)]
        prompt_text = tmpl.format(a=a, b=b)
        test_records.append(format_record(f"TEST-UNSEENPAIR-UNSEENTMPL-{i:02d}", prompt_text, True, a, b, "unseen_pairs_unseen_templates"))

    # Partition 4: Negative Controls (10 examples)
    for i, neg in enumerate(NEGATIVE_PROMPTS_TEST):
        test_records.append(format_record(f"TEST-NEG-{i:02d}", neg, False, category="test_negative_control"))

    return train_records, val_records, test_records, {
        "seed": seed,
        "train_pairs_count": len(train_pairs_pool),
        "unseen_test_pairs_count": len(unseen_test_pairs),
        "train_records_count": len(train_records),
        "val_records_count": len(val_records),
        "test_records_count": len(test_records),
        "train_templates": TRAIN_TEMPLATES,
        "held_out_templates": HELD_OUT_TEMPLATES,
    }

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_recs, val_recs, test_recs, meta = build_dataset(seed=42)
    
    # Save files
    for name, recs in [("train.jsonl", train_recs), ("val.jsonl", val_recs), ("test.jsonl", test_recs)]:
        p = OUTPUT_DIR / name
        with open(p, "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        meta[f"{name}_sha256"] = hashlib.sha256(p.read_bytes()).hexdigest()
        
    manifest_p = OUTPUT_DIR / "manifest.json"
    with open(manifest_p, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
        
    print("Dataset generation complete.")
    print(f"Train: {len(train_recs)} records | Val: {len(val_recs)} records | Test: {len(test_recs)} records")
    print(f"Manifest saved to {manifest_p}")

if __name__ == "__main__":
    main()
