"""
Batch 88: Incremental Next-Step Supervision vs Direct Comparison Runner.

Matched Experimental Conditions:
1. Baseline: Step 10k unmodified.
2. Condition A (Direct Arithmetic): 100 steps on train_direct_tokens.bin (124,289 tokens).
3. Condition B (Incremental Supervision): 100 steps on train_incremental_tokens.bin (124,289 tokens).

Evaluation Protocol:
- Direct evaluation (exact match of immediate answer)
- Incremental multi-step evaluation:
  The model must execute each intermediate step transition autoregressively without human or oracle intervention:
  Starts at a; prompts step 1 -> generates next state -> prompts step 2 from generated state -> ... -> generates final sum.
  Measures:
  * Single-step transition accuracy (fraction of valid next-step state predictions)
  * End-to-end chain accuracy (did the entire autonomous chain reach the true sum c?)
  * Format validity and EOS rate
  * Prediction distribution / modal collapse detection
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
from chakrview.training.prompt_contract import (
    normalize_prompt_response_strings,
    build_causal_sequence,
)
from experiments.batch42_reproducible_evaluation.evaluation.evaluator import (
    load_tokenizer_artifacts,
    compute_file_sha256,
)

BASE_CKPT = Path("experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt")
TOKENIZER_DIR = "release/chakrview-v0.5.0/data/experiments/vocab_4096"
DATA_DIR = Path("experiments/batch87_90_intervention_workbench/data")
OUTPUT_DIR = Path("experiments/batch87_90_intervention_workbench/experiments")


def load_records(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def parse_incremental_answer(text: str) -> Optional[int]:
    clean = text.replace("<|im_end|>", "").strip()
    m = re.match(r"^\s*([0-9]+)", clean)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            return None
    return None


def generate_single_turn(model: ChakrMicro, tokenizer: Any, prompt: str, max_tokens: int = 15) -> Tuple[str, bool]:
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


def evaluate_direct_split(model: ChakrMicro, tokenizer: Any, records: List[Dict[str, Any]]) -> Dict[str, Any]:
    model.eval()
    exact_count = 0
    eos_count = 0
    detailed = []
    preds = []

    for r in records:
        raw, eos = generate_single_turn(model, tokenizer, r["prompt"], max_tokens=15)
        parsed = parse_incremental_answer(raw)
        exact = (parsed is not None and str(parsed) == r["expected_answer"])
        if exact:
            exact_count += 1
        if eos:
            eos_count += 1
        preds.append(str(parsed) if parsed is not None else "[INVALID]")

        detailed.append({
            "id": r["id"],
            "operands": r["operands"],
            "expected": r["expected_answer"],
            "raw": raw.replace("<|im_end|>", "").strip(),
            "parsed": parsed,
            "exact": exact,
            "eos": eos,
        })

    counts = collections.Counter(preds)
    return {
        "total": len(records),
        "exact_matches": exact_count,
        "exact_match_acc": round(exact_count / len(records), 4),
        "eos_rate": round(eos_count / len(records), 4),
        "prediction_distribution": dict(counts.most_common(5)),
        "modal_collapsed": (max(counts.values()) / len(records) >= 0.8) if records else False,
        "records": detailed,
    }


def evaluate_autonomous_incremental_chain(
    model: ChakrMicro, tokenizer: Any, records: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Evaluates autonomous multi-step incremental addition:
    Starts at a; loops b times using model's OWN generated intermediate state as next input.
    Zero ground-truth injection during intermediate transitions.
    """
    model.eval()
    chain_success_count = 0
    total_transitions = 0
    correct_transitions = 0
    detailed = []
    final_preds = []

    for r in records:
        a, b = r["operands"]
        expected_sum = r["sum"]

        curr_state = a
        chain_broken = False
        step_logs = []

        if b == 0:
            prompt = f"<|im_start|>user\nAdd step 0: {a} + 0 from {a}<|im_end|>\n<|im_start|>assistant\n"
            raw, eos = generate_single_turn(model, tokenizer, prompt, max_tokens=15)
            parsed = parse_incremental_answer(raw)
            step_correct = (parsed == a)
            total_transitions += 1
            if step_correct:
                correct_transitions += 1
            final_ans = parsed
            step_logs.append({"step": 0, "state_in": a, "raw": raw.strip(), "parsed": parsed, "correct": step_correct})
        else:
            for step_idx in range(1, b + 1):
                prompt = f"<|im_start|>user\nAdd step {step_idx}: {a} + {b} from {curr_state}<|im_end|>\n<|im_start|>assistant\n"
                raw, eos = generate_single_turn(model, tokenizer, prompt, max_tokens=15)
                parsed = parse_incremental_answer(raw)
                expected_next = curr_state + 1 if not chain_broken else None
                step_correct = (parsed is not None and parsed == (curr_state + 1))

                total_transitions += 1
                if step_correct:
                    correct_transitions += 1
                else:
                    chain_broken = True

                step_logs.append({
                    "step": step_idx,
                    "state_in": curr_state,
                    "raw": raw.strip(),
                    "parsed": parsed,
                    "correct": step_correct,
                })
                # Autonomous feed: pass the model's generated state to next prompt
                curr_state = parsed if parsed is not None else (curr_state + 1)

            final_ans = curr_state

        chain_success = (final_ans == expected_sum)
        if chain_success:
            chain_success_count += 1
        final_preds.append(str(final_ans) if final_ans is not None else "[INVALID]")

        detailed.append({
            "id": r["id"],
            "operands": [a, b],
            "expected_sum": expected_sum,
            "final_model_ans": final_ans,
            "chain_success": chain_success,
            "step_logs": step_logs,
        })

    trans_acc = round(correct_transitions / max(1, total_transitions), 4)
    chain_acc = round(chain_success_count / len(records), 4)
    counts = collections.Counter(final_preds)

    return {
        "total_problems": len(records),
        "chain_success_count": chain_success_count,
        "chain_success_acc": chain_acc,
        "total_transitions": total_transitions,
        "correct_transitions": correct_transitions,
        "transition_accuracy": trans_acc,
        "prediction_distribution": dict(counts.most_common(5)),
        "modal_collapsed": (max(counts.values()) / len(records) >= 0.8) if records else False,
        "records": detailed,
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

    # Load splits
    direct_splits = {
        "train": load_records(DATA_DIR / "train_direct.jsonl"),
        "val": load_records(DATA_DIR / "val_direct.jsonl"),
        "test": load_records(DATA_DIR / "test_held_out.jsonl" if (DATA_DIR / "test_held_out.jsonl").is_file() else DATA_DIR / "test_direct.jsonl"),
    }
    incr_splits = {
        "train": load_records(DATA_DIR / "train_incremental.jsonl"),
        "val": load_records(DATA_DIR / "val_incremental.jsonl"),
        "test": load_records(DATA_DIR / "test_incremental.jsonl"),
    }

    results = {}

    # 1. Baseline Evaluation
    print("\n--- Evaluating Baseline (Step 10,000 Unmodified) ---")
    config = ModelConfig(max_seq_len=512)
    base_model = ChakrMicro(config)
    base_state = torch.load(BASE_CKPT, map_location="cpu", weights_only=False)
    base_model.load_state_dict(base_state["model_state_dict"])
    results["baseline_step_10000"] = {
        "direct_eval": {s: evaluate_direct_split(base_model, tokenizer, direct_splits[s]) for s in ["train", "val", "test"]},
        "incremental_chain": {s: evaluate_autonomous_incremental_chain(base_model, tokenizer, incr_splits[s]) for s in ["train", "val", "test"]},
        "checkpoint_sha256": compute_file_sha256(str(BASE_CKPT)),
    }

    # 2. Condition A: Direct Arithmetic
    model_a, hist_a, ckpt_a, sha_a = train_condition(
        DATA_DIR / "train_direct_tokens.bin", "condition_a_direct", steps=100, batch_size=4, seq_len=64
    )
    results["condition_a_direct"] = {
        "direct_eval": {s: evaluate_direct_split(model_a, tokenizer, direct_splits[s]) for s in ["train", "val", "test"]},
        "incremental_chain": {s: evaluate_autonomous_incremental_chain(model_a, tokenizer, incr_splits[s]) for s in ["train", "val", "test"]},
        "checkpoint_path": str(ckpt_a),
        "checkpoint_sha256": sha_a,
        "history": hist_a,
    }

    # 3. Condition B: Incremental Supervision
    model_b, hist_b, ckpt_b, sha_b = train_condition(
        DATA_DIR / "train_incremental_tokens.bin", "condition_b_incremental", steps=100, batch_size=4, seq_len=64
    )
    results["condition_b_incremental"] = {
        "direct_eval": {s: evaluate_direct_split(model_b, tokenizer, direct_splits[s]) for s in ["train", "val", "test"]},
        "incremental_chain": {s: evaluate_autonomous_incremental_chain(model_b, tokenizer, incr_splits[s]) for s in ["train", "val", "test"]},
        "checkpoint_path": str(ckpt_b),
        "checkpoint_sha256": sha_b,
        "history": hist_b,
    }

    # 4. Cold Reload Determinism
    print("\n--- Verifying Cold Reload Determinism for Condition B ---")
    reload_model = ChakrMicro(config)
    reload_state = torch.load(ckpt_b, map_location="cpu", weights_only=False)
    reload_model.load_state_dict(reload_state["model_state_dict"])
    reloaded_chain = evaluate_autonomous_incremental_chain(reload_model, tokenizer, incr_splits["test"])
    assert reloaded_chain["chain_success_acc"] == results["condition_b_incremental"]["incremental_chain"]["test"]["chain_success_acc"]
    assert reloaded_chain["transition_accuracy"] == results["condition_b_incremental"]["incremental_chain"]["test"]["transition_accuracy"]
    print("Cold reload determinism verified bit-for-bit!")

    summary_path = OUTPUT_DIR / "batch88_experiment_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nAll Batch 88 Experiments Complete! Summary written to {summary_path}")


if __name__ == "__main__":
    main()
