"""
Batch 100: Controlled Diagnostic Experiment on Argument Token Probability Distributions.

Evaluates and compares:
1. Canonical Baseline Step 10k (`training_checkpoint_step_10000.pt`)
2. Condition B: Tool Calling Unstructured (`condition_b_tool_calling_step_100.pt`)
3. Condition C: Explicit Slot-Binding (`condition_c_slot_binding_step_100.pt`)

Metrics:
- Valid tool-call syntax rate
- Tool selection accuracy ('calculator')
- Operation selection accuracy ('add')
- Operand A accuracy
- Operand B accuracy
- Exact ordered argument-pair accuracy
- Constant-call collapse rate
- Clarification control handling
- Diagnostic Teacher-Forced Probability Gap:
  Compares P(target_arg_token | prefix) vs P(argmax_generated_token | prefix)
  at identical teacher-forced sequence positions for both Operand A and Operand B.
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
import torch.nn.functional as F

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
CONDB_CKPT = Path("experiments/batch91_94_tool_calling_comparator/experiments/condition_b_tool_calling_step_100.pt")
CONDC_CKPT = Path("experiments/batch95_98_slot_binding_diagnostics/experiments/condition_c_slot_binding_step_100.pt")
DATA_DIR = Path("experiments/batch95_98_slot_binding_diagnostics/data")
TOKENIZER_DIR = "release/chakrview-v0.5.0/data/experiments/vocab_4096"
OUTPUT_DIR = Path("experiments/batch99_102_representation_audit")

# Map single-digit operand integers to their exact subword token IDs in context
# For Operand A: following '"a": '
OPERAND_A_TOKEN_MAP = {
    0: 1290, 1: 628, 2: 892, 3: 660, 4: 1637,
    5: 1298, 6: 1641, 7: 753, 8: 2225, 9: 60,
}
# For Operand B: following '"b": '
OPERAND_B_TOKEN_MAP = {
    0: 1631, 1: 585, 2: 2202, 3: 54, 4: 55,
    5: 56, 6: 57, 7: 58, 8: 59, 9: 60,
}


def load_model(ckpt_path: Path) -> Tuple[ChakrMicro, str]:
    config = ModelConfig(max_seq_len=512)
    model = ChakrMicro(config)
    data = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(data["model_state_dict"])
    model.eval()
    sha = compute_file_sha256(str(ckpt_path))
    return model, sha


def generate_greedy(
    model: ChakrMicro,
    tokenizer: Any,
    prompt: str,
    max_tokens: int = 50,
) -> str:
    toks = [0] + tokenizer.encode(prompt)
    gen = list(toks)
    with torch.no_grad():
        for _ in range(max_tokens):
            logits = model(torch.tensor([gen]))[0, -1]
            nxt = torch.argmax(logits).item()
            gen.append(nxt)
            if nxt == 1:
                break
    return tokenizer.decode(gen[len(toks):])


def evaluate_checkpoint_controlled(
    model: ChakrMicro,
    tokenizer: Any,
    records: List[Dict[str, Any]],
    clarification_records: List[Dict[str, Any]],
) -> Dict[str, Any]:
    valid_syntax_count = 0
    correct_tool_count = 0
    correct_op_count = 0
    correct_a_count = 0
    correct_b_count = 0
    correct_pair_count = 0
    executed_count = 0
    generated_calls = []

    # Token Probability Gap Accumulators
    arg_a_target_probs = []
    arg_a_gen_probs = []
    arg_a_target_ranks = []
    arg_b_target_probs = []
    arg_b_gen_probs = []
    arg_b_target_ranks = []

    for r in records:
        a_true, b_true = r["operands"]
        raw = generate_greedy(model, tokenizer, r["prompt"], max_tokens=50)
        clean = raw.split("<|im_end|>")[0].strip()
        generated_calls.append(clean)

        v_res = validate_and_execute_tool_call(clean)
        is_syntax = (v_res["status"] not in ["MALFORMED_JSON", "MALFORMED_PAYLOAD"])
        is_tool = (v_res.get("tool") == "calculator")
        is_op = (v_res.get("op") == "add")
        is_a = (v_res.get("a") == a_true) if v_res.get("executed") else False
        is_b = (v_res.get("b") == b_true) if v_res.get("executed") else False
        is_pair = is_a and is_b

        if is_syntax:
            valid_syntax_count += 1
        if is_tool:
            correct_tool_count += 1
        if is_op:
            correct_op_count += 1
        if is_a:
            correct_a_count += 1
        if is_b:
            correct_b_count += 1
        if is_pair:
            correct_pair_count += 1
        if v_res.get("executed"):
            executed_count += 1

        # Teacher-Forced Probability Gap for Operand A
        prefix_a = r["prompt"] + 'TOOL_CALL: {"tool": "calculator", "op": "add", "a": '
        toks_a = [0] + tokenizer.encode(prefix_a)
        with torch.no_grad():
            logits_a = model(torch.tensor([toks_a]))[0, -1]
            probs_a = F.softmax(logits_a, dim=-1)
            target_tok_a = OPERAND_A_TOKEN_MAP[a_true]
            gen_tok_a = torch.argmax(probs_a).item()
            p_tgt_a = probs_a[target_tok_a].item()
            p_gen_a = probs_a[gen_tok_a].item()
            rank_a = (torch.argsort(probs_a, descending=True) == target_tok_a).nonzero().item() + 1
            arg_a_target_probs.append(p_tgt_a)
            arg_a_gen_probs.append(p_gen_a)
            arg_a_target_ranks.append(rank_a)

        # Teacher-Forced Probability Gap for Operand B
        prefix_b = r["prompt"] + f'TOOL_CALL: {{"tool": "calculator", "op": "add", "a": {a_true}, "b": '
        toks_b = [0] + tokenizer.encode(prefix_b)
        with torch.no_grad():
            logits_b = model(torch.tensor([toks_b]))[0, -1]
            probs_b = F.softmax(logits_b, dim=-1)
            target_tok_b = OPERAND_B_TOKEN_MAP[b_true]
            gen_tok_b = torch.argmax(probs_b).item()
            p_tgt_b = probs_b[target_tok_b].item()
            p_gen_b = probs_b[gen_tok_b].item()
            rank_b = (torch.argsort(probs_b, descending=True) == target_tok_b).nonzero().item() + 1
            arg_b_target_probs.append(p_tgt_b)
            arg_b_gen_probs.append(p_gen_b)
            arg_b_target_ranks.append(rank_b)

    # Clarification negative controls
    spurious_calls = 0
    for ctrl in clarification_records:
        raw_ctrl = generate_greedy(model, tokenizer, ctrl["prompt"], max_tokens=35)
        clean_ctrl = raw_ctrl.split("<|im_end|>")[0].strip()
        if "TOOL_CALL:" in clean_ctrl or "{" in clean_ctrl:
            spurious_calls += 1

    total = len(records)
    call_counts = collections.Counter(generated_calls)
    most_common_call, most_common_count = call_counts.most_common(1)[0]
    collapse_rate = most_common_count / total

    return {
        "total_evaluated": total,
        "valid_syntax_rate": round(valid_syntax_count / total, 4),
        "tool_selection_accuracy": round(correct_tool_count / total, 4),
        "operation_selection_accuracy": round(correct_op_count / total, 4),
        "operand_a_accuracy": round(correct_a_count / total, 4),
        "operand_b_accuracy": round(correct_b_count / total, 4),
        "exact_pair_accuracy": round(correct_pair_count / total, 4),
        "runtime_execution_rate": round(executed_count / total, 4),
        "constant_call_collapse_rate": round(collapse_rate, 4),
        "most_common_call": most_common_call,
        "clarification_spurious_call_rate": round(spurious_calls / max(1, len(clarification_records)), 4),
        "teacher_forced_diagnostics": {
            "operand_a": {
                "mean_target_prob": round(float(np.mean(arg_a_target_probs)), 4),
                "mean_generated_prob": round(float(np.mean(arg_a_gen_probs)), 4),
                "prob_ratio_target_to_gen": round(float(np.mean(arg_a_target_probs) / max(1e-6, np.mean(arg_a_gen_probs))), 4),
                "mean_target_rank": round(float(np.mean(arg_a_target_ranks)), 2),
            },
            "operand_b": {
                "mean_target_prob": round(float(np.mean(arg_b_target_probs)), 4),
                "mean_generated_prob": round(float(np.mean(arg_b_gen_probs)), 4),
                "prob_ratio_target_to_gen": round(float(np.mean(arg_b_target_probs) / max(1e-6, np.mean(arg_b_gen_probs))), 4),
                "mean_target_rank": round(float(np.mean(arg_b_target_ranks)), 2),
            },
        },
    }


def run_batch100_controlled_experiment() -> Dict[str, Any]:
    tokenizer, _ = load_tokenizer_artifacts(TOKENIZER_DIR)
    splits = {
        "train": [json.loads(l) for l in open(DATA_DIR / "train.jsonl") if l.strip()],
        "val": [json.loads(l) for l in open(DATA_DIR / "val.jsonl") if l.strip()],
        "test": [json.loads(l) for l in open(DATA_DIR / "test.jsonl") if l.strip()],
        "clarification": [json.loads(l) for l in open(DATA_DIR / "clarification_controls.jsonl") if l.strip()],
    }

    checkpoints = [
        ("Baseline Step 10,000", BASE_CKPT),
        ("Condition B (Unstructured Tool)", CONDB_CKPT),
        ("Condition C (Explicit Slot-Binding)", CONDC_CKPT),
    ]

    all_results = {}
    for name, path in checkpoints:
        model, sha = load_model(path)
        print(f"Evaluating {name} ({sha[:12]}...)...")
        cond_eval = {"sha256": sha, "splits": {}}
        for split_name in ["train", "val", "test"]:
            cond_eval["splits"][split_name] = evaluate_checkpoint_controlled(
                model, tokenizer, splits[split_name], splits["clarification"]
            )
        all_results[name] = cond_eval

    return {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "checkpoints": all_results,
    }


if __name__ == "__main__":
    report = run_batch100_controlled_experiment()
    out_path = OUTPUT_DIR / "controlled_diagnosis_summary.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print("Controlled diagnosis complete. Saved to", out_path)
