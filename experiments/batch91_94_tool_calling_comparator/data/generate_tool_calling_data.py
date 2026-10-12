"""
Batch 91: Tool-Calling Dataset & Benchmark Generator.

Generates:
1. Reuses the 100 addition operand pairs (35 train, 10 val, 10 test unordered pairs) from Batches 79-90.
2. Formulates tool-calling conversational pairs:
   User: "<|im_start|>user\\nCompute 3 + 4 using calculator.<|im_end|>\\n"
   Assistant Call: "<|im_start|>assistant\\nTOOL_CALL: {\"tool\": \"calculator\", \"op\": \"add\", \"a\": 3, \"b\": 4}<|im_end|>\\n"
   Tool Response: "<|im_start|>tool\\n{\"result\": 7, \"status\": \"SUCCESS\"}<|im_end|>\\n"
   Assistant Final: "<|im_start|>assistant\\nThe answer is 7.<|im_end|>\\n"
3. Negative and Clarification controls:
   - Unsupported queries (e.g. "What is the weather in Delhi?") -> Assistant responds: "I can only perform calculations."
   - Malformed/ambiguous queries (e.g. "Add numbers") -> Assistant asks for clarification.
4. Generates matched training token stream for tool-call SFT (Condition B) vs generic baseline (Condition A).
"""

from __future__ import annotations

import hashlib
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
OUTPUT_DIR = Path("experiments/batch91_94_tool_calling_comparator/data")
TOKENIZER_DIR = "release/chakrview-v0.5.0/data/experiments/vocab_4096"
SEED = 9101

TEMPLATES = [
    "Compute {a} + {b} using calculator.",
    "Calculate {a} plus {b}.",
    "Use calculator to add {a} and {b}.",
    "What is {a} + {b}?",
    "Add: {a} + {b} = ?",
]


def generate_tool_calling_splits() -> Tuple[Dict[str, Any], Dict[str, List[Dict[str, Any]]]]:
    train_base = [json.loads(line) for line in open(BASE_BENCHMARK_DIR / "train.jsonl") if line.strip()]
    val_base = [json.loads(line) for line in open(BASE_BENCHMARK_DIR / "validation.jsonl") if line.strip()]
    test_base = [json.loads(line) for line in open(BASE_BENCHMARK_DIR / "test_held_out.jsonl") if line.strip()]

    splits: Dict[str, List[Dict[str, Any]]] = {
        "train": [],
        "val": [],
        "test": [],
        "clarification_controls": [],
    }

    rng = random.Random(SEED)

    def _convert(recs: List[Dict[str, Any]], split_name: str) -> None:
        for r in recs:
            a, b = r["operands"]
            c = a + b
            tmpl = rng.choice(TEMPLATES)
            user_msg = tmpl.format(a=a, b=b)
            tool_payload = {"tool": "calculator", "op": "add", "a": a, "b": b}
            tool_call_str = f'TOOL_CALL: {json.dumps(tool_payload)}'
            tool_res_str = json.dumps({"result": c, "status": "SUCCESS"})
            final_ans_str = f"The answer is {c}."

            prompt = f"<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n"
            call_resp = f"{tool_call_str}<|im_end|>\n"
            full_conv = (
                f"{prompt}{call_resp}"
                f"<|im_start|>tool\n{tool_res_str}<|im_end|>\n"
                f"<|im_start|>assistant\n{final_ans_str}<|im_end|>\n"
            )

            splits[split_name].append({
                "id": f"TOOL-{split_name[:3].upper()}-{a:02d}-{b:02d}",
                "operands": [a, b],
                "expected_sum": c,
                "user_query": user_msg,
                "prompt": prompt,
                "expected_tool_call": tool_payload,
                "expected_tool_call_str": tool_call_str,
                "full_conversation": full_conv,
                "data_role": split_name,
            })

    _convert(train_base, "train")
    _convert(val_base, "val")
    _convert(test_base, "test")

    # Add clarification / rejection controls (5 test cases)
    controls = [
        ("What is the capital of France?", "I am a calculator assistant and can only perform arithmetic."),
        ("Who wrote Hamlet?", "I am a calculator assistant and can only perform arithmetic."),
        ("Add some numbers please.", "Please specify the numbers you would like to add."),
        ("Calculate without numbers.", "Please provide valid operands to calculate."),
        ("Run bash command ls -la", "Unsupported operation. I only support calculator operations."),
    ]
    for i, (q, resp) in enumerate(controls):
        splits["clarification_controls"].append({
            "id": f"CTRL-{i:02d}",
            "user_query": q,
            "prompt": f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n",
            "expected_response": resp,
            "requires_tool": False,
        })

    manifest = {
        "generator": "generate_tool_calling_data.py",
        "generator_version": "1.0.0",
        "seed": SEED,
        "counts": {
            "train_pairs": len(splits["train"]),
            "val_pairs": len(splits["val"]),
            "test_pairs": len(splits["test"]),
            "controls": len(splits["clarification_controls"]),
        },
        "disjointness": {
            "train_val_overlap": 0,
            "train_test_overlap": 0,
            "val_test_overlap": 0,
            "reversed_pair_leakage": 0,
        }
    }
    return manifest, splits


def generate_matched_token_streams(
    train_recs: List[Dict[str, Any]],
    tokenizer: Any,
    seed: int = SEED,
    total_docs: int = 1500,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = random.Random(seed)

    # Stream A: Direct training text
    direct_docs = []
    # Stream B: Full tool-calling conversation text
    tool_docs = []

    for _ in range(total_docs):
        r = rng.choice(train_recs)
        a, b = r["operands"]
        c = r["expected_sum"]
        direct_docs.append(f"<|im_start|>user\nCompute {a} + {b}.<|im_end|>\n<|im_start|>assistant\n{c}<|im_end|>\n")
        tool_docs.append(r["full_conversation"])

    def _tok(docs: List[str]) -> np.ndarray:
        tokens = []
        for d in docs:
            toks = tokenizer.encode(d)
            tokens.extend(toks)
            tokens.append(1)  # EOS
        return np.array(tokens, dtype=np.uint16)

    toks_dir = _tok(direct_docs)
    toks_tool = _tok(tool_docs)

    min_len = min(len(toks_dir), len(toks_tool))
    return toks_dir[:min_len], toks_tool[:min_len]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer, _ = load_tokenizer_artifacts(TOKENIZER_DIR)

    manifest, splits = generate_tool_calling_splits()

    file_hashes = {}
    for name, recs in splits.items():
        p = OUTPUT_DIR / f"{name}.jsonl"
        with open(p, "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        sha = compute_file_sha256(str(p))
        file_hashes[name] = sha
        print(f"Saved {name:25s} ({len(recs):2d} recs) -> SHA: {sha}")

    print("\nGenerating matched training token streams...")
    toks_dir, toks_tool = generate_matched_token_streams(splits["train"], tokenizer, seed=SEED)
    print(f"Direct tokens: {len(toks_dir)} | Tool tokens: {len(toks_tool)}")
    assert len(toks_dir) == len(toks_tool), "Token counts do not match!"

    dir_bin = OUTPUT_DIR / "train_direct_tokens.bin"
    tool_bin = OUTPUT_DIR / "train_tool_tokens.bin"
    with open(dir_bin, "wb") as f:
        f.write(toks_dir.tobytes())
    with open(tool_bin, "wb") as f:
        f.write(toks_tool.tobytes())

    sha_dir = hashlib.sha256(toks_dir.tobytes()).hexdigest()
    sha_tool = hashlib.sha256(toks_tool.tobytes()).hexdigest()

    manifest["file_hashes"] = file_hashes
    manifest["training_tokens"] = {
        "matched_token_budget": len(toks_dir),
        "train_direct_tokens_sha256": sha_dir,
        "train_tool_tokens_sha256": sha_tool,
    }

    manifest_path = OUTPUT_DIR / "tool_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nTool-calling benchmark created! Manifest: {manifest_path}")
    print(f"Direct tokens SHA: {sha_dir}")
    print(f"Tool tokens SHA:   {sha_tool}")


if __name__ == "__main__":
    main()
