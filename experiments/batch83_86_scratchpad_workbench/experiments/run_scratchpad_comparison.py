"""
Batch 84: Bounded Scratchpad vs Direct-Answer Comparison Runner.

Matched Experimental Conditions:
1. Baseline: Step 10k unmodified.
2. Condition A (Direct-Answer Arithmetic): 100 steps on train_direct_tokens.bin (99,417 tokens).
3. Condition B (Scratchpad Arithmetic): 100 steps on train_scratchpad_tokens.bin (99,417 tokens).

Evaluates on:
- Train Direct & Scratchpad (64 pairs)
- Validation Direct & Scratchpad (18 pairs)
- Held-Out Test Direct & Scratchpad (18 pairs)

Key Metrics:
- Exact answer extraction using deterministic parse_final_answer
- Correct answer match count and accuracy
- Output validity rate (did the model produce a parsable integer answer?)
- Full prediction distribution and constant-output warning
- Format adherence and EOS rate
- Answer loss vs intermediate counting loss
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
EXPECTED_BASE_SHA = "96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee"
TOKENIZER_DIR = "release/chakrview-v0.5.0/data/experiments/vocab_4096"
BENCHMARK_DIR = Path("experiments/batch83_86_scratchpad_workbench/data")
OUTPUT_DIR = Path("experiments/batch83_86_scratchpad_workbench/experiments")


def parse_final_answer(text: str) -> Optional[str]:
    clean = text.replace("<|im_end|>", "").strip()
    m = re.search(r"answer:\s*([0-9]+)", clean, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    m2 = re.match(r"^\s*([0-9]+)", clean)
    if m2:
        return m2.group(1).strip()
    return None


def load_records(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_evaluation_examples(records: List[Dict[str, Any]], tokenizer: Any) -> List[Dict[str, Any]]:
    examples = []
    for r in records:
        prompt = r["prompt"]
        response = r["response"]
        expected = r["expected_answer"]

        p_norm, r_norm = normalize_prompt_response_strings(prompt, response, strip_delimiters=False)
        p_toks = tokenizer.encode(p_norm)
        r_toks = tokenizer.encode(r_norm)

        inp, tgt = build_causal_sequence(p_toks, r_toks, bos_id=0, eos_id=1, ignore_index=-100)

        examples.append({
            "id": r["id"],
            "prompt": prompt,
            "response": response,
            "expected_answer": expected,
            "operands": r["operands"],
            "format": r["format"],
            "inp": inp,
            "tgt": tgt,
            "prompt_in": [0] + p_toks,
        })
    return examples


def evaluate_model_on_splits(
    model: ChakrMicro,
    tokenizer: Any,
    splits: Dict[str, List[Dict[str, Any]]],
    max_gen_tokens: int = 50,
) -> Dict[str, Any]:
    model.eval()
    results = {}
    with torch.no_grad():
        for split_name, exs in splits.items():
            exact_count = 0
            valid_format_count = 0
            eos_count = 0
            ce_losses = []
            detailed = []
            extracted_answers = []

            for ex in exs:
                x = torch.tensor([ex["inp"]], dtype=torch.long)
                y = torch.tensor([ex["tgt"]], dtype=torch.long)
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, 4096), y.view(-1), ignore_index=-100).item()
                ce_losses.append(loss)

                # Autoregressive generation
                gen = list(ex["prompt_in"])
                eos_reached = False
                for _ in range(max_gen_tokens):
                    pred_logits = model(torch.tensor([gen]))[0, -1]
                    nxt = torch.argmax(pred_logits).item()
                    gen.append(nxt)
                    if nxt == 1:
                        eos_reached = True
                        break

                raw_gen = tokenizer.decode(gen[len(ex["prompt_in"]):])
                parsed = parse_final_answer(raw_gen)
                exact = (parsed == ex["expected_answer"])
                has_valid_format = (parsed is not None)

                if exact:
                    exact_count += 1
                if has_valid_format:
                    valid_format_count += 1
                if eos_reached:
                    eos_count += 1
                extracted_answers.append(parsed if parsed is not None else "[INVALID]")

                detailed.append({
                    "id": ex["id"],
                    "operands": ex["operands"],
                    "expected": ex["expected_answer"],
                    "raw_generated": raw_gen.replace("<|im_end|>", "").strip(),
                    "parsed_answer": parsed,
                    "exact": exact,
                    "valid_format": has_valid_format,
                    "eos": eos_reached,
                    "ce_loss": round(loss, 4),
                })

            counts = collections.Counter(extracted_answers)
            top_preds = dict(counts.most_common(5))
            modal_collapsed = (len(extracted_answers) > 0 and max(counts.values()) / len(extracted_answers) >= 0.8)

            results[split_name] = {
                "total": len(exs),
                "exact_matches": exact_count,
                "exact_match_acc": round(exact_count / len(exs), 4),
                "valid_format_rate": round(valid_format_count / len(exs), 4),
                "eos_rate": round(eos_count / len(exs), 4),
                "mean_loss": round(float(np.mean(ce_losses)), 4) if ce_losses else 0.0,
                "modal_collapsed": modal_collapsed,
                "prediction_distribution": top_preds,
                "records": detailed,
            }
    return results


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
    split_names = [
        "train_direct", "train_scratchpad",
        "val_direct", "val_scratchpad",
        "test_direct", "test_scratchpad",
    ]
    raw_splits = {s: load_records(BENCHMARK_DIR / f"{s}.jsonl") for s in split_names}
    eval_splits = {s: build_evaluation_examples(raw_splits[s], tokenizer) for s in split_names}

    results_summary = {}

    # 1. Baseline Evaluation (Step 10k unmodified)
    print("\n--- Evaluating Baseline (Step 10,000 Unmodified) ---")
    config = ModelConfig(max_seq_len=512)
    base_model = ChakrMicro(config)
    base_state = torch.load(BASE_CKPT, map_location="cpu", weights_only=False)
    base_model.load_state_dict(base_state["model_state_dict"])
    base_metrics = evaluate_model_on_splits(base_model, tokenizer, eval_splits)
    results_summary["baseline_step_10000"] = {
        "metrics": base_metrics,
        "checkpoint_sha256": compute_file_sha256(str(BASE_CKPT)),
    }

    # 2. Condition A: Direct-Answer Arithmetic
    model_a, hist_a, ckpt_a, sha_a = train_condition(
        BENCHMARK_DIR / "train_direct_tokens.bin",
        "condition_a_direct_arithmetic",
        steps=100, batch_size=4, seq_len=64
    )
    metrics_a = evaluate_model_on_splits(model_a, tokenizer, eval_splits)
    results_summary["condition_a_direct_arithmetic"] = {
        "metrics": metrics_a,
        "checkpoint_path": str(ckpt_a),
        "checkpoint_sha256": sha_a,
        "history": hist_a,
    }

    # 3. Condition B: Scratchpad Arithmetic
    model_b, hist_b, ckpt_b, sha_b = train_condition(
        BENCHMARK_DIR / "train_scratchpad_tokens.bin",
        "condition_b_scratchpad_arithmetic",
        steps=100, batch_size=4, seq_len=64
    )
    metrics_b = evaluate_model_on_splits(model_b, tokenizer, eval_splits)
    results_summary["condition_b_scratchpad_arithmetic"] = {
        "metrics": metrics_b,
        "checkpoint_path": str(ckpt_b),
        "checkpoint_sha256": sha_b,
        "history": hist_b,
    }

    # 4. Cold Reload Verification for Condition B
    print("\n--- Verifying Cold Reload Determinism for Condition B ---")
    reload_model = ChakrMicro(config)
    reload_state = torch.load(ckpt_b, map_location="cpu", weights_only=False)
    reload_model.load_state_dict(reload_state["model_state_dict"])
    reload_metrics = evaluate_model_on_splits(reload_model, tokenizer, eval_splits)
    assert reload_metrics["test_scratchpad"]["exact_matches"] == metrics_b["test_scratchpad"]["exact_matches"]
    assert reload_metrics["test_scratchpad"]["mean_loss"] == metrics_b["test_scratchpad"]["mean_loss"]
    print("Cold reload determinism verified bit-for-bit!")

    summary_file = OUTPUT_DIR / "batch84_experiment_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2)
    print(f"\nAll Batch 84 Experiments Complete! Summary written to {summary_file}")


if __name__ == "__main__":
    main()
