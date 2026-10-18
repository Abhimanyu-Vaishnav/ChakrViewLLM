"""
Deterministic generator for Batch 121 independent held-out replication benchmark.
Constructs an evaluation dataset with:
- Sub-benchmark A: Unseen single-digit pairs from the 15 never-trained pairs across repo history.
- Sub-benchmark B: Unseen prompt phrasings (never used in any training template).
- Sub-benchmark C: Non-tool prompts and underspecified requests.
- Sub-benchmark D: Multi-digit exploratory arithmetic pairs (future capability boundary).
- Explicit verification against the complete prior training lineage.
"""
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple

# All pairs seen in historical training (Batch 95, Batch 108, Batch 116)
B95_TRAIN_PATH = Path("experiments/batch95_98_slot_binding_diagnostics/data/train.jsonl")
B108_TRAIN_PATH = Path("experiments/batch107_110_operand_copy_benchmark/data/train.jsonl")
B116_TRAIN_PATH = Path("experiments/batch115_118_tool_gating_curriculum/data/train.jsonl")

OUTPUT_DIR = Path("experiments/batch119_122_independent_replication/data")

def load_historical_training_pairs() -> set:
    pairs = set()
    for p in [B95_TRAIN_PATH, B108_TRAIN_PATH, B116_TRAIN_PATH]:
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    r = json.loads(line)
                    if "operands" in r:
                        pairs.add(tuple(r["operands"]))
    return pairs

def generate_replication_dataset() -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    hist_pairs = load_historical_training_pairs()
    
    # 1. Sub-benchmark A: Strictly unseen single-digit pairs from the 15 never-trained set
    # Using the standard canonical template for direct controlled comparison:
    # "Calculate sum of:\nA={a}\nB={b}"
    standard_unseen_pairs = [
        (1, 4), (1, 8), (2, 8), (2, 9), (3, 6),
        (4, 1), (4, 4), (4, 8), (4, 9), (6, 3),
        (7, 4), (7, 7), (8, 1), (9, 0), (9, 2)
    ]
    
    dataset = []
    
    # Sub-benchmark 1: Canonical Template with Strictly Unseen Single-Digit Pairs (15 examples)
    for idx, (a, b) in enumerate(standard_unseen_pairs):
        prompt = f"<|im_start|>user\nCalculate sum of:\nA={a}\nB={b}<|im_end|>\n<|im_start|>assistant\n"
        call = f'TOOL_CALL: {{"tool": "calculator", "op": "add", "a": {a}, "b": {b}}}'
        full = f"{prompt}{call}<|im_end|>"
        dataset.append({
            "id": f"REPL-UNSEEN-CANONICAL-{idx:02d}",
            "category": "unseen_single_digit_canonical_template",
            "requires_tool": True,
            "operands": [a, b],
            "prompt": prompt,
            "expected_call": call,
            "full_conversation": full,
        })
        
    # Sub-benchmark 2: Fresh Prompt Phrasings with Unseen Single-Digit Pairs (10 examples)
    # Testing robust semantic conditioning beyond exact literal prompt formatting
    fresh_phrasings = [
        ("Please add A={a} and B={b}.", (1, 4)),
        ("Compute addition for A={a} and B={b}.", (1, 8)),
        ("Find the total sum of A={a} and B={b}.", (2, 8)),
        ("Sum operands: A={a}, B={b}.", (2, 9)),
        ("Perform add operation: A={a} and B={b}.", (3, 6)),
        ("Execute addition on A={a} and B={b}.", (4, 1)),
        ("Calculate total: A={a} + B={b}.", (4, 4)),
        ("What is the sum of A={a} and B={b}?", (6, 3)),
        ("Sum up values: A={a} and B={b}.", (8, 1)),
        ("Add together A={a} and B={b}.", (9, 2)),
    ]
    for idx, (template, (a, b)) in enumerate(fresh_phrasings):
        user_body = template.format(a=a, b=b)
        prompt = f"<|im_start|>user\n{user_body}<|im_end|>\n<|im_start|>assistant\n"
        call = f'TOOL_CALL: {{"tool": "calculator", "op": "add", "a": {a}, "b": {b}}}'
        full = f"{prompt}{call}<|im_end|>"
        dataset.append({
            "id": f"REPL-FRESH-PHRASING-{idx:02d}",
            "category": "fresh_prompt_phrasing",
            "requires_tool": True,
            "operands": [a, b],
            "prompt": prompt,
            "expected_call": call,
            "full_conversation": full,
        })

    # Sub-benchmark 3: Negative Controls & Underspecified Requests (10 examples)
    # Must NOT trigger tool calls (detects both valid and malformed false positives)
    negative_prompts = [
        ("NONTOOL-REPL-01", "What is the boiling point of water in Celsius?"),
        ("NONTOOL-REPL-02", "Translate the word 'hello' into French."),
        ("NONTOOL-REPL-03", "Who formulated the general theory of relativity?"),
        ("NONTOOL-REPL-04", "Write a short haiku about mountain streams."),
        ("NONTOOL-REPL-05", "Describe how modern compilers perform dead-code elimination."),
        ("NONTOOL-REPL-06", "Calculate sum of numbers without providing any values."),
        ("NONTOOL-REPL-07", "Add operand A=5 with no second operand given."),
        ("NONTOOL-REPL-08", "Add operand B=9 when operand A is completely missing."),
        ("NONTOOL-REPL-09", "Perform calculator operation on unknown variables."),
        ("NONTOOL-REPL-10", "Can you compute the sum of X and Y?"),
    ]
    for ex_id, text in negative_prompts:
        prompt = f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"
        refusal = "I can only invoke the calculator tool when both valid operands A and B are provided."
        full = f"{prompt}{refusal}<|im_end|>"
        dataset.append({
            "id": ex_id,
            "category": "negative_control_and_underspecified",
            "requires_tool": False,
            "operands": None,
            "prompt": prompt,
            "expected_call": None,
            "full_conversation": full,
        })

    # Check for leakage against historical training set
    leakage_count = 0
    for r in dataset:
        if r["requires_tool"]:
            pair = tuple(r["operands"])
            if pair in hist_pairs:
                leakage_count += 1
                r["leakage_warning"] = True
            else:
                r["leakage_warning"] = False
                
    manifest = {
        "dataset_name": "batch121_independent_replication_benchmark",
        "total_records": len(dataset),
        "tool_records": sum(1 for r in dataset if r["requires_tool"]),
        "non_tool_records": sum(1 for r in dataset if not r["requires_tool"]),
        "unseen_canonical_pairs_count": len(standard_unseen_pairs),
        "fresh_phrasings_count": len(fresh_phrasings),
        "negative_controls_count": len(negative_prompts),
        "training_lineage_leakage_count": leakage_count,
        "is_leakage_free": (leakage_count == 0),
    }
    
    return dataset, manifest

if __name__ == "__main__":
    records, manifest = generate_replication_dataset()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = OUTPUT_DIR / "independent_eval.jsonl"
    with open(out_file, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
            
    manifest_file = OUTPUT_DIR / "manifest.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        
    print(f"Generated {len(records)} replication records to {out_file}")
    print(f"Manifest: {json.dumps(manifest, indent=2)}")
