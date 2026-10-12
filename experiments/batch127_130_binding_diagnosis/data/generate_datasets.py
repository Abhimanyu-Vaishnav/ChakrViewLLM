"""
Batch 127–130: Deterministic Dataset Generator and Leakage Verification Module.
Generates:
1. Standard Benchmark (Baseline arm): 70 tool examples + 30 controls.
2. Expanded Pair-Density Benchmark (Intervention arm): 200 tool examples + 60 controls.
3. Test Set (45 strictly isolated records across 4 partitions):
   - Partition 1: Familiar pairs with held-out templates (10 examples)
   - Partition 2: Unseen pairs with familiar templates (15 examples)
   - Partition 3: Unseen pairs with held-out templates (10 examples)
   - Partition 4: Negative controls (10 examples)

Strict Leakage Guarantees:
- Test unseen pairs (Partitions 2 and 3) have ZERO overlap with train pairs and reversed train pairs across both arms.
- Held-out templates have ZERO overlap with training or validation templates.
"""
import json
import random
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Tuple, Set

DATA_DIR = Path("experiments/batch127_130_binding_diagnosis/data")

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

def generate_datasets(seed: int = 42):
    rng = random.Random(seed)
    
    # All 2-digit pairs: 10..99 (8,100 pairs)
    all_pairs = [(a, b) for a in range(10, 100) for b in range(10, 100)]
    rng.shuffle(all_pairs)
    
    # 1. Baseline Training pool (60 pairs)
    baseline_train_pool = all_pairs[:60]
    
    # 2. Expanded Training pool (160 pairs, superset of baseline pool)
    expanded_train_pool = all_pairs[:160]
    
    # 3. Compute forbidden pairs for unseen test set:
    # Any pair or reversed pair in expanded_train_pool
    forbidden_pairs: Set[Tuple[int, int]] = set(expanded_train_pool) | {(b, a) for (a, b) in expanded_train_pool}
    
    # 4. Unseen test pairs pool (strictly disjoint from forbidden pairs)
    available_unseen = [p for p in all_pairs[160:] if p not in forbidden_pairs]
    unseen_test_pairs = available_unseen[:30]
    
    # Assert zero leakage
    unseen_set = set(unseen_test_pairs)
    assert len(unseen_set & forbidden_pairs) == 0, "Test-train pair leakage detected!"
    reversed_unseen = {(b, a) for (a, b) in unseen_set}
    assert len(reversed_unseen & forbidden_pairs) == 0, "Test-train reversed pair leakage detected!"
    
    # Build Baseline Train (100 records: 70 tool, 30 controls)
    train_baseline_records = []
    for i in range(70):
        a, b = baseline_train_pool[i % len(baseline_train_pool)]
        tmpl = TRAIN_TEMPLATES[i % len(TRAIN_TEMPLATES)]
        prompt_text = tmpl.format(a=a, b=b)
        train_baseline_records.append(format_record(f"TRAIN-BASE-TOOL-{i:03d}", prompt_text, True, a, b, "train_tool"))
    for j, neg in enumerate(NEGATIVE_PROMPTS_TRAIN * 3):
        train_baseline_records.append(format_record(f"TRAIN-BASE-NEG-{j:03d}", neg, False, category="train_negative_control"))
    rng.shuffle(train_baseline_records)
    
    # Build Expanded Train (260 records: 200 tool, 60 controls)
    train_expanded_records = []
    for i in range(200):
        a, b = expanded_train_pool[i % len(expanded_train_pool)]
        tmpl = TRAIN_TEMPLATES[i % len(TRAIN_TEMPLATES)]
        prompt_text = tmpl.format(a=a, b=b)
        train_expanded_records.append(format_record(f"TRAIN-EXP-TOOL-{i:03d}", prompt_text, True, a, b, "train_tool"))
    for j, neg in enumerate(NEGATIVE_PROMPTS_TRAIN * 6):
        train_expanded_records.append(format_record(f"TRAIN-EXP-NEG-{j:03d}", neg, False, category="train_negative_control"))
    rng.shuffle(train_expanded_records)
    
    # Build Test Set (45 records, identical across both arms)
    test_records = []
    
    # Partition 1: Familiar Pairs with Held-Out Templates (10 examples)
    # Uses pairs from baseline_train_pool (familiar to both arms)
    for i in range(10):
        a, b = baseline_train_pool[i]
        tmpl = HELD_OUT_TEMPLATES[i % len(HELD_OUT_TEMPLATES)]
        prompt_text = tmpl.format(a=a, b=b)
        test_records.append(format_record(f"TEST-FAMPAIR-UNSEENTMPL-{i:02d}", prompt_text, True, a, b, "familiar_pairs_unseen_templates"))
        
    # Partition 2: Unseen Pairs with Familiar Templates (15 examples)
    for i in range(15):
        a, b = unseen_test_pairs[i]
        tmpl = TRAIN_TEMPLATES[i % len(TRAIN_TEMPLATES)]
        prompt_text = tmpl.format(a=a, b=b)
        test_records.append(format_record(f"TEST-UNSEENPAIR-FAMTMPL-{i:02d}", prompt_text, True, a, b, "unseen_pairs_familiar_templates"))
        
    # Partition 3: Unseen Pairs with Held-Out Templates (10 examples)
    for i in range(10):
        a, b = unseen_test_pairs[15 + i]
        tmpl = HELD_OUT_TEMPLATES[i % len(HELD_OUT_TEMPLATES)]
        prompt_text = tmpl.format(a=a, b=b)
        test_records.append(format_record(f"TEST-UNSEENPAIR-UNSEENTMPL-{i:02d}", prompt_text, True, a, b, "unseen_pairs_unseen_templates"))
        
    # Partition 4: Negative Controls (10 examples)
    for i, neg in enumerate(NEGATIVE_PROMPTS_TEST):
        test_records.append(format_record(f"TEST-NEG-{i:02d}", neg, False, category="test_negative_control"))
        
    # Validation Set (20 records)
    val_records = []
    for i in range(15):
        a, b = baseline_train_pool[(i + 20) % len(baseline_train_pool)]
        tmpl = TRAIN_TEMPLATES[(i + 3) % len(TRAIN_TEMPLATES)]
        prompt_text = tmpl.format(a=a, b=b)
        val_records.append(format_record(f"VAL-TOOL-{i:03d}", prompt_text, True, a, b, "val_tool"))
    for j, neg in enumerate(NEGATIVE_PROMPTS_TRAIN[:5]):
        val_records.append(format_record(f"VAL-NEG-{j:03d}", neg, False, category="val_negative_control"))
        
    return train_baseline_records, train_expanded_records, val_records, test_records

def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_base, train_exp, val, test = generate_datasets(seed=42)
    
    files = {
        "train_baseline.jsonl": train_base,
        "train_expanded.jsonl": train_exp,
        "val.jsonl": val,
        "test.jsonl": test,
    }
    
    manifest = {
        "benchmark": "Batch 127-130 Multi-Digit Binding Diagnosis",
        "seed": 42,
        "counts": {k: len(v) for k, v in files.items()},
        "train_templates": TRAIN_TEMPLATES,
        "held_out_templates": HELD_OUT_TEMPLATES,
        "file_hashes": {},
    }
    
    for name, recs in files.items():
        p = DATA_DIR / name
        with open(p, "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        manifest["file_hashes"][name] = hashlib.sha256(p.read_bytes()).hexdigest()
        print(f"Saved {name}: {len(recs)} records (SHA-256: {manifest['file_hashes'][name]})")
        
    m_p = DATA_DIR / "manifest.json"
    with open(m_p, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest written to {m_p}")

if __name__ == "__main__":
    main()
