"""
Batch 68 Transformation Generalization Experiment Runner.

Evaluates whether ChakrMicro can learn the addition transformation on single-digit additions
(where every answer is strictly a single token) or merely memorizes seen pairs.
Compares:
1. Baseline (uniform loss)
2. Answer-Emphasis (5x loss weight on the single answer token)
Runs across Train (25), Validation (10), and Held-Out Benchmark (10) splits.
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
from chakrview.training.weighted_loss import WeightedCausalLoss
from experiments.batch42_reproducible_evaluation.evaluation.evaluator import (
    load_tokenizer_artifacts,
    compute_file_sha256,
)

BASE_CKPT = Path("experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt")
TOKENIZER_DIR = "release/chakrview-v0.5.0/data/experiments/vocab_4096"
DATA_DIR = Path("experiments/batch67_70_generalization_sprint/data")


def load_dataset_records(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_examples(records: List[Dict[str, Any]], tokenizer: Any, answer_weight: float = 1.0) -> List[Dict[str, Any]]:
    examples = []
    for r in records:
        prompt = r["prompt"]
        response = r["response"]
        expected = r["expected_answer"]

        p_toks = tokenizer.encode(prompt)
        r_toks = tokenizer.encode(response)
        ans_toks = tokenizer.encode(expected)
        assert len(ans_toks) == 1, f"Expected single-token answer: {expected}"
        assert r_toks[0] == ans_toks[0]

        inp = [0] + p_toks + r_toks
        tgt = ([-100] * len(p_toks)) + r_toks + [1]

        weights = [0.0] * len(inp)
        # Position len(p_toks) is the answer token
        weights[len(p_toks)] = answer_weight
        # Template tokens
        for i in range(len(p_toks) + 1, len(inp)):
            weights[i] = 1.0

        examples.append({
            "id": r["id"],
            "prompt": prompt,
            "response": response,
            "expected_answer": expected,
            "inp": inp,
            "tgt": tgt,
            "weights": weights,
            "prompt_in": [0] + p_toks,
            "ans_pos": len(p_toks),
            "tmpl_span": (len(p_toks) + 1, len(inp)),
        })
    return examples


def run_transformation_experiment(
    strategy_name: str,
    answer_weight: float = 1.0,
    steps: int = 100,
    batch_size: int = 2,
    lr: float = 5e-4,
    seed: int = 42,
    output_dir: Path = Path("experiments/batch67_70_generalization_sprint/generalization_runs"),
) -> Dict[str, Any]:
    tokenizer, _ = load_tokenizer_artifacts(TOKENIZER_DIR)
    train_recs = load_dataset_records(DATA_DIR / "train_single_digit.jsonl")
    val_recs = load_dataset_records(DATA_DIR / "val_single_digit.jsonl")
    bench_recs = load_dataset_records(DATA_DIR / "benchmark_single_digit.jsonl")

    train_exs = build_examples(train_recs, tokenizer, answer_weight=answer_weight)
    val_exs = build_examples(val_recs, tokenizer, answer_weight=1.0)
    bench_exs = build_examples(bench_recs, tokenizer, answer_weight=1.0)

    # Load canonical step 10k checkpoint
    ckpt = torch.load(BASE_CKPT, map_location="cpu", weights_only=False)
    model = ChakrMicro(ModelConfig(max_seq_len=512))
    model.load_state_dict(ckpt["model_state_dict"])

    loss_fn = WeightedCausalLoss(ignore_index=-100)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_ckpt = output_dir / f"sft_{strategy_name}_step_{steps}.pt"

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

    elapsed_sec = time.perf_counter() - start_time

    # Save checkpoint
    torch.save({
        "sft_step": steps,
        "strategy": strategy_name,
        "answer_weight": answer_weight,
        "model_state_dict": model.state_dict(),
        "train_file_sha256": compute_file_sha256(DATA_DIR / "train_single_digit.jsonl"),
        "seed": seed,
    }, out_ckpt)

    def eval_split(exs: List[Dict[str, Any]]) -> Dict[str, Any]:
        model.eval()
        exact_count = 0
        eos_count = 0
        ans_losses = []
        tmpl_losses = []
        detailed_records = []

        with torch.no_grad():
            for ex in exs:
                x = torch.tensor([ex["inp"]], dtype=torch.long)
                y = torch.tensor([ex["tgt"]], dtype=torch.long)
                logits = model(x)[0]

                # Answer token CE
                a_pos = ex["ans_pos"]
                ans_ce = F.cross_entropy(logits[a_pos].unsqueeze(0), torch.tensor([y[0, a_pos]])).item()
                ans_losses.append(ans_ce)

                # Template tokens CE
                t_s, t_e = ex["tmpl_span"]
                for p in range(t_s, t_e):
                    ce = F.cross_entropy(logits[p].unsqueeze(0), torch.tensor([y[0, p]])).item()
                    tmpl_losses.append(ce)

                # Greedy decode
                gen = list(ex["prompt_in"])
                eos_reached = False
                for _ in range(15):
                    logits_step = model(torch.tensor([gen]))[0, -1]
                    nxt = torch.argmax(logits_step).item()
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

                detailed_records.append({
                    "id": ex["id"],
                    "expected": ex["expected_answer"],
                    "generated": clean,
                    "raw": dec,
                    "exact_match": exact,
                    "eos": eos_reached,
                    "answer_loss": round(ans_ce, 4),
                })

        return {
            "total": len(exs),
            "exact_matches": exact_count,
            "exact_match_acc": round(exact_count / len(exs), 4),
            "eos_count": eos_count,
            "eos_rate": round(eos_count / len(exs), 4),
            "mean_answer_loss": round(float(np.mean(ans_losses)), 4),
            "mean_template_loss": round(float(np.mean(tmpl_losses)), 4),
            "records": detailed_records,
        }

    return {
        "strategy": strategy_name,
        "answer_weight": answer_weight,
        "steps": steps,
        "elapsed_sec": round(elapsed_sec, 2),
        "initial_loss": round(loss_history[0], 4),
        "final_loss": round(loss_history[-1], 4),
        "checkpoint": str(out_ckpt),
        "train": eval_split(train_exs),
        "val": eval_split(val_exs),
        "bench": eval_split(bench_exs),
    }


if __name__ == "__main__":
    out_dir = Path("experiments/batch67_70_generalization_sprint/generalization_runs")
    results = {}

    configs = [
        ("baseline_uniform", 1.0, 100),
        ("answer_emphasis_5x", 5.0, 100),
    ]

    for name, w, steps in configs:
        print(f"--- Running {name} (steps={steps}, weight={w}) ---")
        res = run_transformation_experiment(name, answer_weight=w, steps=steps, output_dir=out_dir)
        results[name] = res
        print(f"[{name}] Train Acc: {res['train']['exact_match_acc']} ({res['train']['exact_matches']}/{res['train']['total']}) | Ans Loss: {res['train']['mean_answer_loss']}")
        print(f"[{name}] Val Acc:   {res['val']['exact_match_acc']} ({res['val']['exact_matches']}/{res['val']['total']}) | Ans Loss: {res['val']['mean_answer_loss']}")
        print(f"[{name}] Bench Acc: {res['bench']['exact_match_acc']} ({res['bench']['exact_matches']}/{res['bench']['total']}) | Ans Loss: {res['bench']['mean_answer_loss']}")

    summary_file = out_dir / "generalization_comparison_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nAll generalization experiments complete. Saved to {summary_file}")
