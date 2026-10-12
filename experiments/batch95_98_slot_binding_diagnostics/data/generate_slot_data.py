"""
Batch 95: Operand Slot-Binding Dataset Generator.

Hypothesis: Separating operand identification from tool-call syntax formulation
by providing explicit structured slots:
  Request: add numbers
  Operand A: {a}
  Operand B: {b}
  Required action: calculator addition
allows testing whether the model can extract and bind dynamic arguments to the
JSON tool schema rather than collapsing to static mode {"a": 1, "b": 1}.

Crucial Constraints:
1. Reuses exactly the identical 100 operand pairs partitioned in Batches 79-94
   (64 train, 18 val, 18 test) to prevent split/template leakage.
2. Zero answer leakage: The prompt NEVER contains the result or the tool call.
3. Generates structured slot representations for:
   - Condition C (Structured Operand Slots)
   - Baseline Condition B records for reference.
"""

from __future__ import annotations

import json
from pathlib import Path
import random
import sys
from typing import Any, Dict, List, Tuple

sys.path.insert(0, ".")
import numpy as np

from experiments.batch42_reproducible_evaluation.evaluation.evaluator import (
    load_tokenizer_artifacts,
    compute_file_sha256,
)

BASE_BENCHMARK_DIR = Path("experiments/batch79_82_balanced_arithmetic/data")
OUTPUT_DIR = Path("experiments/batch95_98_slot_binding_diagnostics/data")
TOKENIZER_DIR = "release/chakrview-v0.5.0/data/experiments/vocab_4096"
SEED = 9501


def build_slot_prompt(a: int, b: int) -> str:
    user_content = (
        f"Request: add numbers\n"
        f"Operand A: {a}\n"
        f"Operand B: {b}\n"
        f"Required action: calculator addition"
    )
    return f"<|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n"


def generate_slot_datasets() -> Tuple[Dict[str, Any], Dict[str, List[Dict[str, Any]]]]:
    train_base = [json.loads(line) for line in open(BASE_BENCHMARK_DIR / "train.jsonl") if line.strip()]
    val_base = [json.loads(line) for line in open(BASE_BENCHMARK_DIR / "validation.jsonl") if line.strip()]
    test_base = [json.loads(line) for line in open(BASE_BENCHMARK_DIR / "test_held_out.jsonl") if line.strip()]

    splits: Dict[str, List[Dict[str, Any]]] = {
        "train": [],
        "val": [],
        "test": [],
        "clarification_controls": [],
    }

    def _convert(recs: List[Dict[str, Any]], split_name: str) -> None:
        for r in recs:
            a, b = r["operands"]
            c = a + b
            prompt = build_slot_prompt(a, b)
            tool_payload = {"tool": "calculator", "op": "add", "a": a, "b": b}
            tool_call_str = f'TOOL_CALL: {json.dumps(tool_payload)}'
            tool_res_str = json.dumps({"result": c, "status": "SUCCESS"})
            final_ans_str = f"The answer is {c}."

            full_conv = (
                f"{prompt}{tool_call_str}<|im_end|>\n"
                f"<|im_start|>tool\n{tool_res_str}<|im_end|>\n"
                f"<|im_start|>assistant\n{final_ans_str}<|im_end|>\n"
            )

            splits[split_name].append({
                "id": f"SLOT-{split_name[:3].upper()}-{a:02d}-{b:02d}",
                "operands": [a, b],
                "expected_sum": c,
                "user_query": f"Operand A: {a}, Operand B: {b}",
                "prompt": prompt,
                "expected_tool_call": tool_payload,
                "expected_tool_call_str": tool_call_str,
                "full_conversation": full_conv,
                "data_role": split_name,
            })

    _convert(train_base, "train")
    _convert(val_base, "val")
    _convert(test_base, "test")

    # Negative / clarification controls
    controls = [
        ("Request: general knowledge\nQuery: What is the capital of India?\nRequired action: answer", "I am a calculator assistant and only support arithmetic calculations."),
        ("Request: translation\nText: hello\nRequired action: translate", "I am a calculator assistant and only support arithmetic calculations."),
        ("Request: add numbers\nOperand A: unknown\nOperand B: 5\nRequired action: calculator addition", "Please provide valid numerical operands for both Operand A and Operand B."),
        ("Request: add numbers\nOperand A: 3\nOperand B: missing\nRequired action: calculator addition", "Please provide valid numerical operands for both Operand A and Operand B."),
        ("Request: execute code\nScript: import os; os.system('ls')\nRequired action: execute", "Unsupported action. I only support calculator addition."),
    ]
    for i, (q, resp) in enumerate(controls):
        splits["clarification_controls"].append({
            "id": f"SLOT-CTRL-{i:02d}",
            "user_query": q,
            "prompt": f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n",
            "expected_response": resp,
            "requires_tool": False,
        })

    manifest = {
        "generator": "generate_slot_data.py",
        "description": "Batch 95 structured operand slot binding benchmark",
        "seed": SEED,
        "counts": {k: len(v) for k, v in splits.items()},
    }
    return manifest, splits


def create_tokenized_stream(
    records: List[Dict[str, Any]],
    tokenizer: Any,
    target_tokens: int = 71139,
) -> np.ndarray:
    """Creates a flattened uint16 token stream repeating conversations up to target_tokens."""
    all_toks: List[int] = []
    rng = random.Random(SEED)
    recs = list(records)

    while len(all_toks) < target_tokens:
        rng.shuffle(recs)
        for r in recs:
            conv = r["full_conversation"]
            encoded = [0] + tokenizer.encode(conv) + [1]
            all_toks.extend(encoded)
            if len(all_toks) >= target_tokens:
                break

    arr = np.array(all_toks[:target_tokens], dtype=np.uint16)
    return arr


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest, splits = generate_slot_datasets()

    for name, recs in splits.items():
        out_file = OUTPUT_DIR / f"{name}.jsonl"
        with open(out_file, "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        print(f"Saved {len(recs)} records to {out_file}")

    manifest_file = OUTPUT_DIR / "slot_benchmark_manifest.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # Tokenize training stream for Condition C (Slot-Binding SFT)
    tokenizer, _ = load_tokenizer_artifacts(TOKENIZER_DIR)
    slot_stream = create_tokenized_stream(splits["train"], tokenizer, target_tokens=71139)
    bin_file = OUTPUT_DIR / "train_slot_tokens.bin"
    slot_stream.tofile(bin_file)
    print(f"Saved tokenized stream ({len(slot_stream)} tokens) to {bin_file}")


if __name__ == "__main__":
    main()
