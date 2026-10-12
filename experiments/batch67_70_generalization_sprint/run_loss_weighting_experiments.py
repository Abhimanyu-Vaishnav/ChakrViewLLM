"""
Batch 67 Comparative Loss-Weighting Experiment Runner.

Compares 3 loss weighting regimes on the synthetic arithmetic dataset:
Condition A (Baseline): Uniform weight (1.0 on all active targets).
Condition B (Answer-Emphasis): 5.0x weight on answer tokens, 1.0x on template tokens.
Condition C (Answer-Only Diagnostic): 1.0x on answer tokens, 0.0x on template tokens (masked).

Evaluates training loss, answer-only loss, template-only loss, exact match, and EOS completion.
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
TRAIN_DATA = Path("experiments/batch59_62_learning_verification/data/train_synthetic.jsonl")
VAL_DATA = Path("experiments/batch59_62_learning_verification/data/val_synthetic.jsonl")
BENCH_DATA = Path("experiments/batch59_62_learning_verification/data/benchmark_synthetic.jsonl")


def build_weighted_examples(
    jsonl_path: Path,
    tokenizer: Any,
    strategy: str,  # "baseline", "answer_emphasis", "answer_only"
    answer_weight: float = 5.0,
) -> List[Dict[str, Any]]:
    with open(jsonl_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    end_template_toks = tokenizer.encode("<|im_end|>\n")
    template_len = len(end_template_toks)

    examples = []
    for r in records:
        prompt = r["prompt"]
        response = r["response"]
        expected = r["expected_answer"]

        # Prompt & response tokens (exact delimiters)
        p_toks = tokenizer.encode(prompt)
        r_toks = tokenizer.encode(response)

        # Identification of answer vs template tokens
        ans_toks = tokenizer.encode(expected)
        assert r_toks[:len(ans_toks)] == ans_toks
        ans_len = len(ans_toks)

        inp = [0] + p_toks + r_toks
        tgt = ([-100] * len(p_toks)) + r_toks + [1]  # EOS = 1

        # Token weights definition
        weights = [0.0] * len(inp)
        for i in range(len(p_toks), len(p_toks) + ans_len):
            weights[i] = answer_weight if strategy == "answer_emphasis" else 1.0

        for i in range(len(p_toks) + ans_len, len(inp)):
            if strategy == "answer_only":
                weights[i] = 0.0
                tgt[i] = -100  # mask template tokens completely in answer_only
            elif strategy == "answer_emphasis":
                weights[i] = 1.0
            else:  # baseline
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
            "ans_span": (len(p_toks), len(p_toks) + ans_len),
            "template_span": (len(p_toks) + ans_len, len(inp)),
        })

    return examples


def run_experiment(
    strategy: str,
    steps: int = 50,
    batch_size: int = 2,
    lr: float = 1e-4,
    seed: int = 42,
    output_dir: Path = Path("experiments/batch67_70_generalization_sprint/loss_experiments"),
) -> Dict[str, Any]:
    tokenizer, _ = load_tokenizer_artifacts(TOKENIZER_DIR)
    train_exs = build_weighted_examples(TRAIN_DATA, tokenizer, strategy)
    val_exs = build_weighted_examples(VAL_DATA, tokenizer, "baseline")  # standard evaluation
    bench_exs = build_weighted_examples(BENCH_DATA, tokenizer, "baseline")

    # Load canonical base model
    ckpt = torch.load(BASE_CKPT, map_location="cpu", weights_only=False)
    model = ChakrMicro(ModelConfig(max_seq_len=512))
    model.load_state_dict(ckpt["model_state_dict"])

    loss_fn = WeightedCausalLoss(ignore_index=-100)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_ckpt = output_dir / f"sft_{strategy}_step_{steps}.pt"

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
        "strategy": strategy,
        "model_state_dict": model.state_dict(),
        "train_file_sha256": compute_file_sha256(TRAIN_DATA),
        "seed": seed,
    }, out_ckpt)

    # Evaluate exact match and separate answer vs template loss across splits
    def eval_split(exs: List[Dict[str, Any]]) -> Dict[str, Any]:
        model.eval()
        exact_matches = 0
        eos_count = 0
        ans_losses = []
        tmpl_losses = []

        with torch.no_grad():
            for ex in exs:
                # Loss decomposition
                x = torch.tensor([ex["inp"]], dtype=torch.long)
                y = torch.tensor([ex["tgt"]], dtype=torch.long)
                logits = model(x)[0]

                # Answer tokens CE
                a_start, a_end = ex["ans_span"]
                for p in range(a_start, a_end):
                    ce = F.cross_entropy(logits[p].unsqueeze(0), torch.tensor([y[0, p]])).item()
                    ans_losses.append(ce)

                # Template tokens CE
                t_start, t_end = ex["template_span"]
                for p in range(t_start, t_end):
                    # In answer_only, ex["tgt"] was masked (-100), but for EVALUATION we compare against true target:
                    true_tok = tokenizer.encode(ex["response"])[p - len(tokenizer.encode(ex["prompt"]))] if p < t_end - 1 else 1
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
                if clean == ex["expected_answer"].strip():
                    exact_matches += 1
                if eos_reached:
                    eos_count += 1

        return {
            "total": len(exs),
            "exact_matches": exact_matches,
            "exact_match_acc": round(exact_matches / len(exs), 4),
            "eos_count": eos_count,
            "eos_rate": round(eos_count / len(exs), 4),
            "mean_answer_loss": round(float(np.mean(ans_losses)), 4),
            "mean_template_loss": round(float(np.mean(tmpl_losses)), 4),
        }

    train_metrics = eval_split(train_exs)
    val_metrics = eval_split(val_exs)
    bench_metrics = eval_split(bench_exs)

    return {
        "strategy": strategy,
        "steps": steps,
        "elapsed_sec": round(elapsed_sec, 2),
        "initial_loss": round(loss_history[0], 4),
        "final_loss": round(loss_history[-1], 4),
        "checkpoint": str(out_ckpt),
        "train": train_metrics,
        "val": val_metrics,
        "bench": bench_metrics,
    }


if __name__ == "__main__":
    out_base = Path("experiments/batch67_70_generalization_sprint/loss_experiments")
    results = {}
    for strat in ["baseline", "answer_emphasis", "answer_only"]:
        print(f"--- Running condition: {strat} ---")
        res = run_experiment(strat, steps=50, output_dir=out_base)
        results[strat] = res
        print(f"[{strat}] Train Acc: {res['train']['exact_match_acc']} | Ans Loss: {res['train']['mean_answer_loss']} | Tmpl Loss: {res['train']['mean_template_loss']}")
        print(f"[{strat}] Val Acc:   {res['val']['exact_match_acc']} | Ans Loss: {res['val']['mean_answer_loss']} | Tmpl Loss: {res['val']['mean_template_loss']}")

    summary_file = out_base / "loss_comparison_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nAll experiments complete. Summary written to {summary_file}")
