"""
Batch 73 Controlled Learning Experiment Runner.

Evaluates:
- Baseline (Untrained Canonical Checkpoint Step 10k)
- Condition A: Uniform Cross-Entropy SFT (100 steps)
- Condition B: Answer-Emphasis SFT (5x answer loss weight, 100 steps)
- Condition C: Answer-Only Diagnostic (Template loss = 0, 100 steps)
- Overfit Control: Small 3-example subset to verify pipeline training capacity

Tracks:
- Train, Validation, Held-Out Test, and Symmetry Diagnostic exact matches
- Answer-only token loss vs template-token loss
- Trajectory snapshots at step 0, 25, 50, 75, 100
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Tuple

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
from chakrview.training.weighted_loss import WeightedCausalLoss
from experiments.batch42_reproducible_evaluation.evaluation.evaluator import (
    load_tokenizer_artifacts,
    compute_file_sha256,
)

BASE_CKPT = Path("experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt")
TOKENIZER_DIR = "release/chakrview-v0.5.0/data/experiments/vocab_4096"
DATA_DIR = Path("experiments/batch71_74_controlled_learning/data")


def load_records(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_examples(
    records: List[Dict[str, Any]],
    tokenizer: Any,
    strategy: str = "uniform",  # "uniform", "answer_emphasis", "answer_only"
    answer_weight: float = 1.0,
) -> List[Dict[str, Any]]:
    examples = []
    for r in records:
        prompt = r["prompt"]
        response = r["response"]
        expected = r["expected_answer"]

        p_norm, r_norm = normalize_prompt_response_strings(prompt, response, strip_delimiters=False)
        p_toks = tokenizer.encode(p_norm)
        r_toks = tokenizer.encode(r_norm)

        inp, tgt = build_causal_sequence(p_toks, r_toks, bos_id=0, eos_id=1, ignore_index=-100)

        # Build weights
        weights = [0.0] * len(inp)
        # Position len(p_toks) is the answer token
        ans_pos = len(p_toks)
        weights[ans_pos] = answer_weight if strategy == "answer_emphasis" else 1.0

        # Template positions
        for p in range(ans_pos + 1, len(inp)):
            if strategy == "answer_only":
                weights[p] = 0.0
                tgt[p] = -100  # mask template in answer_only
            elif strategy == "answer_emphasis":
                weights[p] = 1.0
            else:
                weights[p] = 1.0

        examples.append({
            "id": r["id"],
            "prompt": prompt,
            "response": response,
            "expected_answer": expected,
            "inp": inp,
            "tgt": tgt,
            "weights": weights,
            "prompt_in": [0] + p_toks,
            "ans_pos": ans_pos,
            "tmpl_span": (ans_pos + 1, len(inp)),
        })
    return examples


def evaluate_model_on_splits(
    model: ChakrMicro,
    tokenizer: Any,
    splits: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, Any]:
    model.eval()
    results = {}
    with torch.no_grad():
        for split_name, exs in splits.items():
            exact_count = 0
            eos_count = 0
            ans_losses = []
            tmpl_losses = []
            detailed = []

            for ex in exs:
                x = torch.tensor([ex["inp"]], dtype=torch.long)
                y = torch.tensor([ex["tgt"]], dtype=torch.long)
                logits = model(x)[0]

                # Answer token loss
                a_pos = ex["ans_pos"]
                true_ans_token = tokenizer.encode(ex["expected_answer"])[0]
                ans_ce = F.cross_entropy(logits[a_pos].unsqueeze(0), torch.tensor([true_ans_token])).item()
                ans_losses.append(ans_ce)

                # Template tokens loss
                t_s, t_e = ex["tmpl_span"]
                for p in range(t_s, t_e):
                    true_tok = tokenizer.encode(ex["response"])[p - len(tokenizer.encode(ex["prompt"]))] if p < t_e - 1 else 1
                    ce = F.cross_entropy(logits[p].unsqueeze(0), torch.tensor([true_tok])).item()
                    tmpl_losses.append(ce)

                # Greedy decode
                gen = list(ex["prompt_in"])
                eos_reached = False
                for _ in range(15):
                    pred_logits = model(torch.tensor([gen]))[0, -1]
                    nxt = torch.argmax(pred_logits).item()
                    gen.append(nxt)
                    if nxt == 1:
                        eos_reached = True
                        break

                dec = tokenizer.decode(gen[len(ex["prompt_in"]):])
                clean = dec.replace("<|im_end|>", "").strip()
                exact = (clean == ex["expected_answer"].strip())
                if exact:
                    exact_count += 1
                if eos_reached:
                    eos_count += 1

                detailed.append({
                    "id": ex["id"],
                    "expected": ex["expected_answer"],
                    "generated": clean,
                    "exact": exact,
                    "eos": eos_reached,
                    "ans_loss": round(ans_ce, 4),
                })

            results[split_name] = {
                "total": len(exs),
                "exact_matches": exact_count,
                "exact_match_acc": round(exact_count / len(exs), 4),
                "eos_count": eos_count,
                "eos_rate": round(eos_count / len(exs), 4),
                "mean_answer_loss": round(float(np.mean(ans_losses)), 4),
                "mean_template_loss": round(float(np.mean(tmpl_losses)), 4),
                "records": detailed,
            }
    return results


def run_condition(
    condition_name: str,
    strategy: str,
    answer_weight: float = 1.0,
    steps: int = 100,
    batch_size: int = 2,
    lr: float = 5e-4,
    seed: int = 42,
    output_dir: Path = Path("experiments/batch71_74_controlled_learning/experiments"),
    train_subset_size: int | None = None,
) -> Dict[str, Any]:
    tokenizer, _ = load_tokenizer_artifacts(TOKENIZER_DIR)
    train_recs = load_records(DATA_DIR / "train_arithmetic.jsonl")
    val_recs = load_records(DATA_DIR / "val_arithmetic.jsonl")
    test_recs = load_records(DATA_DIR / "test_held_out.jsonl")
    symm_recs = load_records(DATA_DIR / "symmetry_diagnostic.jsonl")

    if train_subset_size is not None:
        train_recs = train_recs[:train_subset_size]

    train_exs = build_examples(train_recs, tokenizer, strategy=strategy, answer_weight=answer_weight)
    val_exs = build_examples(val_recs, tokenizer, strategy="uniform", answer_weight=1.0)
    test_exs = build_examples(test_recs, tokenizer, strategy="uniform", answer_weight=1.0)
    symm_exs = build_examples(symm_recs, tokenizer, strategy="uniform", answer_weight=1.0)

    eval_splits = {
        "train": train_exs,
        "validation": val_exs,
        "held_out_test": test_exs,
        "symmetry_diagnostic": symm_exs,
    }

    # Load canonical step 10k checkpoint
    ckpt = torch.load(BASE_CKPT, map_location="cpu", weights_only=False)
    model = ChakrMicro(ModelConfig(max_seq_len=512))
    model.load_state_dict(ckpt["model_state_dict"])

    # If steps == 0, evaluate baseline
    if steps == 0:
        base_results = evaluate_model_on_splits(model, tokenizer, eval_splits)
        return {
            "condition": condition_name,
            "strategy": strategy,
            "steps": 0,
            "metrics": base_results,
        }

    loss_fn = WeightedCausalLoss(ignore_index=-100)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_ckpt = output_dir / f"sft_{condition_name}_step_{steps}.pt"

    trajectory = {}
    # Record step 0
    trajectory[0] = evaluate_model_on_splits(model, tokenizer, {"train": train_exs, "val": val_exs})

    start_time = time.perf_counter()
    loss_history = []

    model.train()
    for step in range(1, steps + 1):
        batch_indices = rng.randint(0, len(train_exs), size=batch_size)
        batch = [train_exs[i] for i in batch_indices]

        max_len = max(len(ex["inp"]) for ex in batch)
        pad_id = 2

        padded_inp = [ex["inp"] + [pad_id] * (max_len - len(ex["inp"])) for ex in batch]
        padded_tgt = [ex["tgt"] + [-100] * (max_len - len(ex["tgt"])) for ex in batch]
        padded_w = [ex["weights"] + [0.0] * (max_len - len(ex["weights"])) for ex in batch]

        x = torch.tensor(padded_inp, dtype=torch.long)
        y = torch.tensor(padded_tgt, dtype=torch.long)
        w = torch.tensor(padded_w, dtype=torch.float)

        optimizer.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y, w)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        loss_history.append(loss.item())

        if step in (25, 50, 75, 100):
            model.eval()
            trajectory[step] = evaluate_model_on_splits(model, tokenizer, {"train": train_exs, "val": val_exs})
            model.train()

    elapsed_sec = time.perf_counter() - start_time

    torch.save({
        "sft_step": steps,
        "condition": condition_name,
        "strategy": strategy,
        "model_state_dict": model.state_dict(),
        "train_file_sha256": compute_file_sha256(DATA_DIR / "train_arithmetic.jsonl"),
        "seed": seed,
    }, out_ckpt)

    final_metrics = evaluate_model_on_splits(model, tokenizer, eval_splits)

    return {
        "condition": condition_name,
        "strategy": strategy,
        "answer_weight": answer_weight,
        "steps": steps,
        "elapsed_sec": round(elapsed_sec, 2),
        "initial_loss": round(loss_history[0], 4),
        "final_loss": round(loss_history[-1], 4),
        "checkpoint": str(out_ckpt),
        "checkpoint_sha256": compute_file_sha256(out_ckpt),
        "trajectory": trajectory,
        "final_metrics": final_metrics,
    }


if __name__ == "__main__":
    out_dir = Path("experiments/batch71_74_controlled_learning/experiments")
    all_results = {}

    print("--- 1. Evaluating Baseline Step 10k ---")
    base_res = run_condition("baseline_untrained", strategy="uniform", steps=0, output_dir=out_dir)
    all_results["baseline_untrained"] = base_res
    print("Baseline Val Acc:", base_res["metrics"]["validation"]["exact_match_acc"])

    print("--- 2. Evaluating Condition A: Uniform SFT (100 steps) ---")
    cond_a = run_condition("condition_a_uniform", strategy="uniform", answer_weight=1.0, steps=100, output_dir=out_dir)
    all_results["condition_a_uniform"] = cond_a
    print("Cond A Train Acc:", cond_a["final_metrics"]["train"]["exact_match_acc"], "Val Acc:", cond_a["final_metrics"]["validation"]["exact_match_acc"], "Test Acc:", cond_a["final_metrics"]["held_out_test"]["exact_match_acc"])

    print("--- 3. Evaluating Condition B: Answer-Emphasis SFT (5x weight, 100 steps) ---")
    cond_b = run_condition("condition_b_emphasis", strategy="answer_emphasis", answer_weight=5.0, steps=100, output_dir=out_dir)
    all_results["condition_b_emphasis"] = cond_b
    print("Cond B Train Acc:", cond_b["final_metrics"]["train"]["exact_match_acc"], "Val Acc:", cond_b["final_metrics"]["validation"]["exact_match_acc"], "Test Acc:", cond_b["final_metrics"]["held_out_test"]["exact_match_acc"])

    print("--- 4. Evaluating Condition C: Answer-Only Diagnostic (100 steps) ---")
    cond_c = run_condition("condition_c_answer_only", strategy="answer_only", answer_weight=1.0, steps=100, output_dir=out_dir)
    all_results["condition_c_answer_only"] = cond_c
    print("Cond C Train Acc:", cond_c["final_metrics"]["train"]["exact_match_acc"], "EOS Rate:", cond_c["final_metrics"]["train"]["eos_rate"])

    print("--- 5. Evaluating Overfit Control (3 examples, 50 full-batch steps) ---")
    overfit = run_condition("overfit_control_3ex", strategy="uniform", answer_weight=1.0, steps=50, batch_size=3, lr=1e-3, train_subset_size=3, output_dir=out_dir)
    all_results["overfit_control"] = overfit
    print("Overfit Train Acc:", overfit["final_metrics"]["train"]["exact_match_acc"], f"({overfit['final_metrics']['train']['exact_matches']}/{overfit['final_metrics']['train']['total']})")

    summary_file = out_dir / "batch73_experiment_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nAll experiments complete. Summary saved to {summary_file}")
