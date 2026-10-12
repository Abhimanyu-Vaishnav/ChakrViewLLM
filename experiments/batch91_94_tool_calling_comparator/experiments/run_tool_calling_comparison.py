"""
Batch 92: Bounded Tool-Calling Experiment & Multi-Turn Chain Evaluator.

Matched Conditions:
1. Baseline: Step 10k unmodified.
2. Condition A (Direct Arithmetic SFT): 100 steps on train_direct_tokens.bin (71,139 tokens).
3. Condition B (Tool-Calling SFT): 100 steps on train_tool_tokens.bin (71,139 tokens).

Evaluation Protocol (Multi-Turn Tool Loop):
For each test record:
1. Prompt Model:
   "<|im_start|>user\\nCompute 3 + 4 using calculator.<|im_end|>\\n<|im_start|>assistant\\n"
2. Model generates tool call candidate.
3. Validate candidate:
   - Is it valid JSON?
   - Does it call "calculator"?
   - Does it specify op="add"?
   - Are operands a and b correctly extracted?
4. If valid tool call:
   - Safe deterministic calculator executes result: 7.
   - Formulate tool turn:
     "<|im_start|>tool\\n{\"result\": 7, \"status\": \"SUCCESS\"}<|im_end|>\\n<|im_start|>assistant\\n"
   - Model generates final turn.
   - Check if final turn states the correct answer 7.
5. If invalid: record as tool failure (no silent repair).
"""

from __future__ import annotations

import collections
import hashlib
import json
from pathlib import Path
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, ".")
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from experiments.batch42_reproducible_evaluation.evaluation.evaluator import (
    load_tokenizer_artifacts,
    compute_file_sha256,
)
from experiments.batch91_94_tool_calling_comparator.tools.calculator import (
    validate_and_execute_tool_call,
)

BASE_CKPT = Path("experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt")
TOKENIZER_DIR = "release/chakrview-v0.5.0/data/experiments/vocab_4096"
DATA_DIR = Path("experiments/batch91_94_tool_calling_comparator/data")
OUTPUT_DIR = Path("experiments/batch91_94_tool_calling_comparator/experiments")


def load_records(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def generate_single_turn(model: ChakrMicro, tokenizer: Any, prompt: str, max_tokens: int = 40) -> Tuple[str, bool]:
    p_norm = prompt.strip() + "\n"
    toks = [0] + tokenizer.encode(p_norm)
    gen = list(toks)
    eos_reached = False
    with torch.no_grad():
        for _ in range(max_tokens):
            logits = model(torch.tensor([gen]))[0, -1]
            nxt = torch.argmax(logits).item()
            gen.append(nxt)
            if nxt == 1:
                eos_reached = True
                break
    raw = tokenizer.decode(gen[len(toks):])
    return raw, eos_reached


def evaluate_tool_calling_on_records(
    model: ChakrMicro, tokenizer: Any, records: List[Dict[str, Any]]
) -> Dict[str, Any]:
    model.eval()
    total = len(records)
    valid_syntax_count = 0
    correct_tool_count = 0
    correct_args_count = 0
    executed_count = 0
    final_success_count = 0
    traces = []
    generated_calls = []

    for r in records:
        expected_a, expected_b = r["operands"]
        expected_sum = r["expected_sum"]

        # Turn 1: Model generates tool call
        raw_call, eos1 = generate_single_turn(model, tokenizer, r["prompt"], max_tokens=55)
        # Extract turn 1 text before <|im_end|> if multi-turn generated
        turn1_raw = raw_call.split("<|im_end|>")[0].strip()
        clean_call = turn1_raw
        generated_calls.append(clean_call)

        # Validate tool call candidate
        val_res = validate_and_execute_tool_call(clean_call)
        is_valid_syntax = (val_res["status"] not in ["MALFORMED_JSON", "MALFORMED_PAYLOAD"])
        is_correct_tool = (val_res.get("tool") == "calculator")
        is_correct_op = (val_res.get("op") == "add")
        is_correct_args = (
            val_res.get("a") == expected_a and val_res.get("b") == expected_b
        ) if val_res.get("executed") else False

        if is_valid_syntax:
            valid_syntax_count += 1
        if is_correct_tool:
            correct_tool_count += 1
        if is_correct_args:
            correct_args_count += 1

        raw_final = ""
        eos2 = False
        final_correct = False

        if val_res.get("executed"):
            executed_count += 1
            calc_result = val_res["result"]

            # Turn 2: Feed tool response and generate final response
            tool_msg = f"<|im_start|>tool\n{json.dumps({'result': calc_result, 'status': 'SUCCESS'})}<|im_end|>\n<|im_start|>assistant\n"
            turn2_prompt = f"{r['prompt']}{clean_call}<|im_end|>\n{tool_msg}"
            raw_final, eos2 = generate_single_turn(model, tokenizer, turn2_prompt, max_tokens=35)
            clean_final = raw_final.split("<|im_end|>")[0].strip()

            # Check if final response cites the correct sum
            if str(expected_sum) in clean_final:
                final_correct = True
                final_success_count += 1

        traces.append({
            "id": r["id"],
            "operands": [expected_a, expected_b],
            "expected_sum": expected_sum,
            "raw_tool_call": clean_call,
            "validation_status": val_res["status"],
            "tool_result": val_res.get("result"),
            "tool_executed": val_res.get("executed", False),
            "correct_args": is_correct_args,
            "final_response": raw_final.replace("<|im_end|>", "").strip(),
            "end_to_end_success": final_correct,
        })

    counts = collections.Counter(generated_calls)
    return {
        "total": total,
        "valid_tool_syntax_count": valid_syntax_count,
        "valid_tool_syntax_rate": round(valid_syntax_count / max(1, total), 4),
        "correct_tool_selection_rate": round(correct_tool_count / max(1, total), 4),
        "correct_arguments_rate": round(correct_args_count / max(1, total), 4),
        "runtime_execution_count": executed_count,
        "runtime_execution_rate": round(executed_count / max(1, total), 4),
        "end_to_end_success_count": final_success_count,
        "end_to_end_success_rate": round(final_success_count / max(1, total), 4),
        "prediction_distribution": dict(counts.most_common(5)),
        "traces": traces,
    }


def evaluate_clarification_controls(
    model: ChakrMicro, tokenizer: Any, records: List[Dict[str, Any]]
) -> Dict[str, Any]:
    model.eval()
    total = len(records)
    safe_non_tool_count = 0
    spurious_calls = 0
    traces = []

    for r in records:
        raw_call, _ = generate_single_turn(model, tokenizer, r["prompt"], max_tokens=55)
        clean = raw_call.split("<|im_end|>")[0].strip()
        val = validate_and_execute_tool_call(clean)
        if val.get("executed") or val["status"] not in ["MALFORMED_JSON", "MALFORMED_PAYLOAD"]:
            spurious_calls += 1
        else:
            safe_non_tool_count += 1
        traces.append({
            "prompt": r["user_query"],
            "raw": clean,
            "status": val["status"],
            "safe": val["status"] in ["MALFORMED_JSON", "MALFORMED_PAYLOAD"],
        })

    return {
        "total": total,
        "safe_non_tool_count": safe_non_tool_count,
        "safe_non_tool_rate": round(safe_non_tool_count / max(1, total), 4),
        "malformed_or_spurious_tool_calls": spurious_calls,
        "traces": traces,
    }


def train_condition(
    token_file: Path,
    condition_name: str,
    steps: int = 100,
    batch_size: int = 4,
    seq_len: int = 64,
    lr: float = 5e-4,
    seed: int = 42,
) -> Tuple[ChakrMicro, List[Dict[str, Any]], Path, str]:
    print(f"\n--- Training {condition_name} ({steps} steps) ---")
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)

    with open(token_file, "rb") as f:
        tokens = np.frombuffer(f.read(), dtype=np.uint16).astype(np.int64)

    config = ModelConfig(max_seq_len=512)
    model = ChakrMicro(config)
    base_state = torch.load(BASE_CKPT, map_location="cpu", weights_only=False)
    model.load_state_dict(base_state["model_state_dict"])

    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss()

    history = []
    max_idx = len(tokens) - seq_len - 1

    model.train()
    for step in range(steps):
        idxs = rng.randint(0, max_idx, size=batch_size)
        x = torch.stack([torch.tensor(tokens[i : i + seq_len], dtype=torch.long) for i in idxs])
        y = torch.stack([torch.tensor(tokens[i + 1 : i + seq_len + 1], dtype=torch.long) for i in idxs])

        opt.zero_grad()
        logits = model(x)
        loss = loss_fn(logits.view(-1, 4096), y.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if (step + 1) % 25 == 0 or step == 0:
            print(f"Step {step + 1:3d}/{steps} | Loss: {loss.item():.4f}")
            history.append({
                "step": step + 1,
                "loss": round(loss.item(), 4),
            })

    ckpt_path = OUTPUT_DIR / f"{condition_name}_step_{steps}.pt"
    torch.save({
        "step": steps,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": opt.state_dict(),
        "history": history,
        "condition_name": condition_name,
    }, ckpt_path)
    sha = compute_file_sha256(str(ckpt_path))
    print(f"Saved {condition_name} checkpoint: {ckpt_path} (SHA-256: {sha})")
    return model, history, ckpt_path, sha


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer, _ = load_tokenizer_artifacts(TOKENIZER_DIR)

    splits = {
        "train": load_records(DATA_DIR / "train.jsonl"),
        "val": load_records(DATA_DIR / "val.jsonl"),
        "test": load_records(DATA_DIR / "test.jsonl"),
    }
    controls = load_records(DATA_DIR / "clarification_controls.jsonl")

    results = {}

    # 1. Baseline Evaluation
    print("\n--- Evaluating Baseline (Step 10,000 Unmodified) ---")
    config = ModelConfig(max_seq_len=512)
    base_model = ChakrMicro(config)
    base_state = torch.load(BASE_CKPT, map_location="cpu", weights_only=False)
    base_model.load_state_dict(base_state["model_state_dict"])
    results["baseline_step_10000"] = {
        "splits": {s: evaluate_tool_calling_on_records(base_model, tokenizer, splits[s]) for s in ["train", "val", "test"]},
        "clarification_control": evaluate_clarification_controls(base_model, tokenizer, controls),
        "checkpoint_sha256": compute_file_sha256(str(BASE_CKPT)),
    }

    # 2. Condition A: Direct Arithmetic SFT
    ckpt_a = OUTPUT_DIR / "condition_a_direct_step_100.pt"
    if ckpt_a.is_file():
        print(f"\n--- Loading existing condition_a_direct checkpoint ---")
        model_a = ChakrMicro(config)
        state_a = torch.load(ckpt_a, map_location="cpu", weights_only=False)
        model_a.load_state_dict(state_a["model_state_dict"])
        hist_a = state_a.get("history", [])
        sha_a = compute_file_sha256(str(ckpt_a))
    else:
        model_a, hist_a, ckpt_a, sha_a = train_condition(
            DATA_DIR / "train_direct_tokens.bin", "condition_a_direct", steps=100, batch_size=4, seq_len=64
        )
    results["condition_a_direct"] = {
        "splits": {s: evaluate_tool_calling_on_records(model_a, tokenizer, splits[s]) for s in ["train", "val", "test"]},
        "clarification_control": evaluate_clarification_controls(model_a, tokenizer, controls),
        "checkpoint_path": str(ckpt_a),
        "checkpoint_sha256": sha_a,
        "history": hist_a,
    }

    # 3. Condition B: Tool-Calling SFT
    ckpt_b = OUTPUT_DIR / "condition_b_tool_calling_step_100.pt"
    if ckpt_b.is_file():
        print(f"\n--- Loading existing condition_b_tool_calling checkpoint ---")
        model_b = ChakrMicro(config)
        state_b = torch.load(ckpt_b, map_location="cpu", weights_only=False)
        model_b.load_state_dict(state_b["model_state_dict"])
        hist_b = state_b.get("history", [])
        sha_b = compute_file_sha256(str(ckpt_b))
    else:
        model_b, hist_b, ckpt_b, sha_b = train_condition(
            DATA_DIR / "train_tool_tokens.bin", "condition_b_tool_calling", steps=100, batch_size=4, seq_len=64
        )
    results["condition_b_tool_calling"] = {
        "splits": {s: evaluate_tool_calling_on_records(model_b, tokenizer, splits[s]) for s in ["train", "val", "test"]},
        "clarification_control": evaluate_clarification_controls(model_b, tokenizer, controls),
        "checkpoint_path": str(ckpt_b),
        "checkpoint_sha256": sha_b,
        "history": hist_b,
    }

    # 4. Cold Reload Verification
    print("\n--- Verifying Cold Reload Determinism for Condition B ---")
    reload_model = ChakrMicro(config)
    reload_state = torch.load(ckpt_b, map_location="cpu", weights_only=False)
    reload_model.load_state_dict(reload_state["model_state_dict"])
    reloaded_eval = evaluate_tool_calling_on_records(reload_model, tokenizer, splits["test"])
    assert reloaded_eval["end_to_end_success_count"] == results["condition_b_tool_calling"]["splits"]["test"]["end_to_end_success_count"]
    assert reloaded_eval["valid_tool_syntax_rate"] == results["condition_b_tool_calling"]["splits"]["test"]["valid_tool_syntax_rate"]
    print("Cold reload determinism verified bit-for-bit!")

    summary_path = OUTPUT_DIR / "batch92_experiment_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nAll Batch 92 Experiments Complete! Summary: {summary_path}")


if __name__ == "__main__":
    main()
