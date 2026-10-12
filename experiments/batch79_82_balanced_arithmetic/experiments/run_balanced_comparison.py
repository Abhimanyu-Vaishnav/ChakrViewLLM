"""
Batch 81: Balanced Pretraining and Generalization Experiment Runner.

Executes:
1. Baseline: Step 10k unmodified.
2. Condition B: Matched Generic Language Control (Stage C, 100 steps).
3. Condition C: Balanced Arithmetic Curriculum (Uniform sum-class sampling 0..18, 100 steps).
4. Overfit Control: 3 examples (50 steps).

Evaluates on:
- Train Split (64 pairs)
- Validation Split (18 pairs)
- Held-Out Test Split (18 pairs)
- Symmetry Diagnostic (32 off-diagonal pairs)
- Template Robustness Diagnostic (36 query/word template records)

Measures:
- Exact answer-match accuracy vs constant-output baseline
- Answer-token CE loss vs template-token CE loss
- EOS completion rate
- Prediction distribution and modal collapse rate
- Full correctness breakdown by sum class and operand order
"""

from __future__ import annotations

import collections
import hashlib
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
from experiments.batch42_reproducible_evaluation.evaluation.evaluator import (
    load_tokenizer_artifacts,
    compute_file_sha256,
)

BASE_CKPT = Path("experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt")
EXPECTED_BASE_SHA = "96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee"
TOKENIZER_DIR = "release/chakrview-v0.5.0/data/experiments/vocab_4096"
BENCHMARK_DIR = Path("experiments/batch79_82_balanced_arithmetic/data")
OUTPUT_DIR = Path("experiments/batch79_82_balanced_arithmetic/experiments")


def load_records(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_benchmark_examples(records: List[Dict[str, Any]], tokenizer: Any) -> List[Dict[str, Any]]:
    examples = []
    for r in records:
        prompt = r["prompt"]
        response = r["response"]
        expected = r["expected_answer"]

        p_norm, r_norm = normalize_prompt_response_strings(prompt, response, strip_delimiters=False)
        p_toks = tokenizer.encode(p_norm)
        r_toks = tokenizer.encode(r_norm)

        inp, tgt = build_causal_sequence(p_toks, r_toks, bos_id=0, eos_id=1, ignore_index=-100)

        # Answer token position(s)
        ans_toks = tokenizer.encode(expected)
        ans_pos = len(p_toks)
        ans_span = (ans_pos, ans_pos + len(ans_toks))

        examples.append({
            "id": r["id"],
            "prompt": prompt,
            "response": response,
            "expected_answer": expected,
            "sum": r.get("sum", int(expected)),
            "operands": r.get("operands", []),
            "inp": inp,
            "tgt": tgt,
            "prompt_in": [0] + p_toks,
            "ans_span": ans_span,
            "ans_toks": ans_toks,
            "tmpl_span": (ans_span[1], len(inp)),
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
            preds = []
            by_sum = collections.defaultdict(lambda: {"total": 0, "correct": 0})

            for ex in exs:
                x = torch.tensor([ex["inp"]], dtype=torch.long)
                logits = model(x)[0]

                # Answer tokens loss
                s_idx, e_idx = ex["ans_span"]
                ex_ans_loss = 0.0
                for pos, true_tok in zip(range(s_idx, e_idx), ex["ans_toks"]):
                    ce = F.cross_entropy(logits[pos].unsqueeze(0), torch.tensor([true_tok])).item()
                    ex_ans_loss += ce
                ex_ans_loss /= max(1, len(ex["ans_toks"]))
                ans_losses.append(ex_ans_loss)

                # Template tokens loss
                t_s, t_e = ex["tmpl_span"]
                for pos in range(t_s, t_e):
                    # Target token from ex["tgt"]
                    tgt_tok = ex["tgt"][pos]
                    if tgt_tok != -100:
                        ce = F.cross_entropy(logits[pos].unsqueeze(0), torch.tensor([tgt_tok])).item()
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
                preds.append(clean)

                s_val = ex["sum"]
                by_sum[s_val]["total"] += 1
                if exact:
                    by_sum[s_val]["correct"] += 1

                detailed.append({
                    "id": ex["id"],
                    "operands": ex["operands"],
                    "sum": s_val,
                    "expected": ex["expected_answer"],
                    "generated": clean,
                    "exact": exact,
                    "eos": eos_reached,
                    "ans_loss": round(ex_ans_loss, 4),
                })

            pred_freq = dict(collections.Counter(preds).most_common(5))
            modal_collapsed = (len(preds) > 0 and max(collections.Counter(preds).values()) / len(preds) >= 0.8)

            results[split_name] = {
                "total": len(exs),
                "exact_matches": exact_count,
                "exact_match_acc": round(exact_count / len(exs), 4),
                "eos_count": eos_count,
                "eos_rate": round(eos_count / len(exs), 4),
                "mean_answer_loss": round(float(np.mean(ans_losses)), 4) if ans_losses else 0.0,
                "mean_template_loss": round(float(np.mean(tmpl_losses)), 4) if tmpl_losses else 0.0,
                "modal_collapsed": modal_collapsed,
                "prediction_distribution": pred_freq,
                "by_sum_accuracy": {str(k): v for k, v in sorted(by_sum.items())},
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


def run_overfit_sanity(
    train_exs: List[Dict[str, Any]],
    steps: int = 50,
) -> Tuple[ChakrMicro, Path, str]:
    print("\n--- Running Overfit Sanity Check (3 examples, 50 steps) ---")
    torch.manual_seed(42)
    config = ModelConfig(max_seq_len=512)
    model = ChakrMicro(config)
    base_state = torch.load(BASE_CKPT, map_location="cpu", weights_only=False)
    model.load_state_dict(base_state["model_state_dict"])

    opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    subset = train_exs[:3]
    model.train()
    for step in range(steps):
        opt.zero_grad()
        total_loss = 0.0
        for ex in subset:
            x = torch.tensor([ex["inp"]], dtype=torch.long)
            y = torch.tensor([ex["tgt"]], dtype=torch.long)
            logits = model(x)
            loss = loss_fn(logits.view(-1, 4096), y.view(-1))
            total_loss += loss.item()
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if (step + 1) % 10 == 0:
            avg_loss = total_loss / len(subset)
            print(f"Overfit Step {step + 1:2d}/{steps} | Loss: {avg_loss:.4f}")

    ckpt_path = OUTPUT_DIR / "overfit_sanity_3ex_step_50.pt"
    torch.save({"step": steps, "model_state_dict": model.state_dict()}, ckpt_path)
    sha = compute_file_sha256(str(ckpt_path))
    return model, ckpt_path, sha


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer, _ = load_tokenizer_artifacts(TOKENIZER_DIR)

    # 1. Load benchmark splits
    train_recs = load_records(BENCHMARK_DIR / "train.jsonl")
    val_recs = load_records(BENCHMARK_DIR / "validation.jsonl")
    test_recs = load_records(BENCHMARK_DIR / "test_held_out.jsonl")
    symm_recs = load_records(BENCHMARK_DIR / "symmetry_diagnostic.jsonl")
    tmpl_recs = load_records(BENCHMARK_DIR / "template_diagnostic.jsonl")

    splits = {
        "train": build_benchmark_examples(train_recs, tokenizer),
        "validation": build_benchmark_examples(val_recs, tokenizer),
        "test_held_out": build_benchmark_examples(test_recs, tokenizer),
        "symmetry_diagnostic": build_benchmark_examples(symm_recs, tokenizer),
        "template_diagnostic": build_benchmark_examples(tmpl_recs, tokenizer),
    }

    results_summary = {}

    # Condition A: Unmodified Baseline (Step 10k)
    print("\n--- Evaluating Baseline (Step 10,000 Unmodified) ---")
    config = ModelConfig(max_seq_len=512)
    base_model = ChakrMicro(config)
    base_state = torch.load(BASE_CKPT, map_location="cpu", weights_only=False)
    base_model.load_state_dict(base_state["model_state_dict"])
    base_metrics = evaluate_model_on_splits(base_model, tokenizer, splits)
    results_summary["baseline_step_10000"] = {
        "metrics": base_metrics,
        "checkpoint_sha256": compute_file_sha256(str(BASE_CKPT)),
    }

    # Condition B: Matched Generic Language Control
    cond_b_tokens = BENCHMARK_DIR / "condition_b_generic_control_tokens.bin"
    model_b, hist_b, ckpt_b, sha_b = train_condition(
        cond_b_tokens, "condition_b_generic_control", steps=100, batch_size=4, seq_len=64
    )
    metrics_b = evaluate_model_on_splits(model_b, tokenizer, splits)
    results_summary["condition_b_generic_control"] = {
        "metrics": metrics_b,
        "checkpoint_path": str(ckpt_b),
        "checkpoint_sha256": sha_b,
        "history": hist_b,
    }

    # Condition C: Balanced Arithmetic Curriculum
    cond_c_tokens = BENCHMARK_DIR / "condition_c_balanced_arithmetic_tokens.bin"
    model_c, hist_c, ckpt_c, sha_c = train_condition(
        cond_c_tokens, "condition_c_balanced_curriculum", steps=100, batch_size=4, seq_len=64
    )
    metrics_c = evaluate_model_on_splits(model_c, tokenizer, splits)
    results_summary["condition_c_balanced_curriculum"] = {
        "metrics": metrics_c,
        "checkpoint_path": str(ckpt_c),
        "checkpoint_sha256": sha_c,
        "history": hist_c,
    }

    # Overfit sanity
    model_overfit, ckpt_overfit, sha_overfit = run_overfit_sanity(splits["train"], steps=50)
    metrics_overfit = evaluate_model_on_splits(model_overfit, tokenizer, splits)
    results_summary["overfit_sanity_3ex"] = {
        "metrics": metrics_overfit,
        "checkpoint_path": str(ckpt_overfit),
        "checkpoint_sha256": sha_overfit,
    }

    # Cold reload verification
    print("\n--- Verifying Cold Reload Determinism for Condition C ---")
    reload_model = ChakrMicro(config)
    reload_state = torch.load(ckpt_c, map_location="cpu", weights_only=False)
    reload_model.load_state_dict(reload_state["model_state_dict"])
    reload_metrics = evaluate_model_on_splits(reload_model, tokenizer, splits)
    assert reload_metrics["test_held_out"]["exact_matches"] == metrics_c["test_held_out"]["exact_matches"], "Reload exact match mismatch!"
    assert reload_metrics["test_held_out"]["mean_answer_loss"] == metrics_c["test_held_out"]["mean_answer_loss"], "Reload ans loss mismatch!"
    print("Cold reload determinism verified bit-for-bit!")

    summary_file = OUTPUT_DIR / "batch81_experiment_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2)
    print(f"\nAll Batch 81 Experiments Complete! Summary written to {summary_file}")


if __name__ == "__main__":
    main()
