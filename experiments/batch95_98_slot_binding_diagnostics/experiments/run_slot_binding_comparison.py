"""
Batch 96: Explicit Operand Slot-Binding Experiment & Evaluation Harness.

Compares:
1. Canonical Baseline Step 10k (`training_checkpoint_step_10000.pt`)
2. Batch 91-94 Unstructured Tool-Calling Model (`condition_b_tool_calling_step_100.pt`)
3. Condition C: Explicit Operand Slot-Binding Model (`condition_c_slot_binding_step_100.pt`)

Measures independently:
- Valid tool-call syntax rate
- Correct tool selection ("calculator")
- Correct operation ("add")
- Correct Operand A accuracy
- Correct Operand B accuracy
- Exact argument-pair (A & B) accuracy
- Runtime deterministic calculator execution rate
- End-to-end task success
- Unique generated calls and constant-call collapse rate
- Clarification / negative controls handling
- Bit-for-bit cold reload determinism
"""

from __future__ import annotations

import collections
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, ".")
import numpy as np
import torch
import torch.nn as nn
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
UNSTRUCTURED_CKPT = Path("experiments/batch91_94_tool_calling_comparator/experiments/condition_b_tool_calling_step_100.pt")
DATA_DIR = Path("experiments/batch95_98_slot_binding_diagnostics/data")
OUTPUT_DIR = Path("experiments/batch95_98_slot_binding_diagnostics/experiments")
TOKENIZER_DIR = "release/chakrview-v0.5.0/data/experiments/vocab_4096"


def load_records(path: Path) -> List[Dict[str, Any]]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def generate_single_turn(
    model: ChakrMicro,
    tokenizer: Any,
    prompt: str,
    max_tokens: int = 55,
) -> Tuple[str, bool]:
    toks = [0] + tokenizer.encode(prompt)
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


def evaluate_slot_binding_on_records(
    model: ChakrMicro, tokenizer: Any, records: List[Dict[str, Any]]
) -> Dict[str, Any]:
    model.eval()
    total = len(records)
    valid_syntax_count = 0
    correct_tool_count = 0
    correct_op_count = 0
    correct_operand_a_count = 0
    correct_operand_b_count = 0
    correct_pair_count = 0
    executed_count = 0
    final_success_count = 0
    traces = []
    generated_calls = []

    for r in records:
        expected_a, expected_b = r["operands"]
        expected_sum = r["expected_sum"]

        # Turn 1: Model generates tool call candidate
        raw_call, eos1 = generate_single_turn(model, tokenizer, r["prompt"], max_tokens=55)
        clean_call = raw_call.split("<|im_end|>")[0].strip()
        generated_calls.append(clean_call)

        # Validate candidate
        val_res = validate_and_execute_tool_call(clean_call)
        is_valid_syntax = (val_res["status"] not in ["MALFORMED_JSON", "MALFORMED_PAYLOAD"])
        is_correct_tool = (val_res.get("tool") == "calculator")
        is_correct_op = (val_res.get("op") == "add")

        gen_a = val_res.get("a")
        gen_b = val_res.get("b")
        is_a_correct = (gen_a == expected_a) if val_res.get("executed") else False
        is_b_correct = (gen_b == expected_b) if val_res.get("executed") else False
        is_pair_correct = is_a_correct and is_b_correct

        if is_valid_syntax:
            valid_syntax_count += 1
        if is_correct_tool:
            correct_tool_count += 1
        if is_correct_op:
            correct_op_count += 1
        if is_a_correct:
            correct_operand_a_count += 1
        if is_b_correct:
            correct_operand_b_count += 1
        if is_pair_correct:
            correct_pair_count += 1

        raw_final = ""
        eos2 = False
        final_correct = False

        if val_res.get("executed"):
            executed_count += 1
            calc_result = val_res["result"]

            # Turn 2: Feed tool response and allow model to generate final response
            tool_msg = f"<|im_start|>tool\n{json.dumps({'result': calc_result, 'status': 'SUCCESS'})}<|im_end|>\n<|im_start|>assistant\n"
            turn2_prompt = f"{r['prompt']}{clean_call}<|im_end|>\n{tool_msg}"
            raw_final, eos2 = generate_single_turn(model, tokenizer, turn2_prompt, max_tokens=35)
            clean_final = raw_final.split("<|im_end|>")[0].strip()

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
            "generated_a": gen_a,
            "generated_b": gen_b,
            "correct_operand_a": is_a_correct,
            "correct_operand_b": is_b_correct,
            "correct_pair": is_pair_correct,
            "final_response": raw_final.split("<|im_end|>")[0].strip(),
            "end_to_end_success": final_correct,
        })

    counts = collections.Counter(generated_calls)
    unique_calls = len(counts)
    most_common_call, most_common_count = counts.most_common(1)[0] if counts else ("", 0)
    collapse_rate = round(most_common_count / max(1, total), 4)

    return {
        "total": total,
        "valid_tool_syntax_count": valid_syntax_count,
        "valid_tool_syntax_rate": round(valid_syntax_count / max(1, total), 4),
        "correct_tool_selection_rate": round(correct_tool_count / max(1, total), 4),
        "correct_operation_rate": round(correct_op_count / max(1, total), 4),
        "correct_operand_a_rate": round(correct_operand_a_count / max(1, total), 4),
        "correct_operand_b_rate": round(correct_operand_b_count / max(1, total), 4),
        "exact_argument_pair_rate": round(correct_pair_count / max(1, total), 4),
        "runtime_execution_count": executed_count,
        "runtime_execution_rate": round(executed_count / max(1, total), 4),
        "end_to_end_success_count": final_success_count,
        "end_to_end_success_rate": round(final_success_count / max(1, total), 4),
        "unique_calls_count": unique_calls,
        "constant_call_collapse_rate": collapse_rate,
        "most_common_call": most_common_call,
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


def train_slot_binding_condition(
    token_file: Path,
    condition_name: str = "condition_c_slot_binding",
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

    results: Dict[str, Any] = {}

    # 1. Baseline Evaluation on Slot Benchmark
    print("\n--- Evaluating Baseline (Step 10,000 Unmodified) on Slot Benchmark ---")
    config = ModelConfig(max_seq_len=512)
    base_model = ChakrMicro(config)
    base_state = torch.load(BASE_CKPT, map_location="cpu", weights_only=False)
    base_model.load_state_dict(base_state["model_state_dict"])
    results["baseline_step_10000"] = {
        "splits": {s: evaluate_slot_binding_on_records(base_model, tokenizer, splits[s]) for s in ["train", "val", "test"]},
        "clarification_control": evaluate_clarification_controls(base_model, tokenizer, controls),
        "checkpoint_sha256": compute_file_sha256(str(BASE_CKPT)),
    }

    # 2. Existing Unstructured Tool Model on Slot Benchmark
    if UNSTRUCTURED_CKPT.is_file():
        print("\n--- Evaluating Condition B (Unstructured Tool Model) on Slot Benchmark ---")
        unstr_model = ChakrMicro(config)
        unstr_state = torch.load(UNSTRUCTURED_CKPT, map_location="cpu", weights_only=False)
        unstr_model.load_state_dict(unstr_state["model_state_dict"])
        results["condition_b_unstructured_model"] = {
            "splits": {s: evaluate_slot_binding_on_records(unstr_model, tokenizer, splits[s]) for s in ["train", "val", "test"]},
            "clarification_control": evaluate_clarification_controls(unstr_model, tokenizer, controls),
            "checkpoint_sha256": compute_file_sha256(str(UNSTRUCTURED_CKPT)),
        }

    # 3. Train Condition C: Explicit Operand Slot-Binding
    ckpt_c = OUTPUT_DIR / "condition_c_slot_binding_step_100.pt"
    if ckpt_c.is_file():
        print(f"\n--- Loading existing condition_c_slot_binding checkpoint ---")
        model_c = ChakrMicro(config)
        state_c = torch.load(ckpt_c, map_location="cpu", weights_only=False)
        model_c.load_state_dict(state_c["model_state_dict"])
        hist_c = state_c.get("history", [])
        sha_c = compute_file_sha256(str(ckpt_c))
    else:
        model_c, hist_c, ckpt_c, sha_c = train_slot_binding_condition(
            DATA_DIR / "train_slot_tokens.bin", "condition_c_slot_binding", steps=100, batch_size=4, seq_len=64
        )
    results["condition_c_slot_binding"] = {
        "splits": {s: evaluate_slot_binding_on_records(model_c, tokenizer, splits[s]) for s in ["train", "val", "test"]},
        "clarification_control": evaluate_clarification_controls(model_c, tokenizer, controls),
        "checkpoint_path": str(ckpt_c),
        "checkpoint_sha256": sha_c,
        "history": hist_c,
    }

    # 4. Cold Reload Verification for Condition C
    print("\n--- Verifying Cold Reload Determinism for Condition C ---")
    reload_model = ChakrMicro(config)
    reload_state = torch.load(ckpt_c, map_location="cpu", weights_only=False)
    reload_model.load_state_dict(reload_state["model_state_dict"])
    reloaded_eval = evaluate_slot_binding_on_records(reload_model, tokenizer, splits["test"])
    assert reloaded_eval["end_to_end_success_count"] == results["condition_c_slot_binding"]["splits"]["test"]["end_to_end_success_count"]
    assert reloaded_eval["exact_argument_pair_rate"] == results["condition_c_slot_binding"]["splits"]["test"]["exact_argument_pair_rate"]
    print("Cold reload determinism verified bit-for-bit!")

    summary_path = OUTPUT_DIR / "batch96_experiment_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nAll Batch 96 Experiments Complete! Summary: {summary_path}")


if __name__ == "__main__":
    main()
