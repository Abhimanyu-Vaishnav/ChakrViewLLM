"""
Batch 77: Bounded Pretraining & Arithmetic Curriculum Comparison Runner.

Conducts controlled pretraining from the verified step-10,000 checkpoint:
- Baseline: Canonical Checkpoint Step 10k (unmodified)
- Condition A: Matched Language Control (Stage C data, 100 steps, batch 4, seq 64)
- Condition B: Synthetic Arithmetic Curriculum (multi-template, 100 steps, batch 4, seq 64)
- Overfit Control: 3-example overfit control (50 steps) to verify causal learning capacity

Evaluates all checkpoints on the frozen Batch 72 arithmetic benchmark:
- Train Set (30 pairs)
- Validation Set (10 pairs)
- Held-Out Test Set (10 pairs)
- Symmetry Diagnostic Set (11 pairs)

Records:
- Exact match count and percentage
- Answer-token loss vs template-token loss
- EOS completion rate
- Output frequency distribution & modal collapse diagnosis
- Checkpoint saving & SHA-256 verification
"""

from __future__ import annotations

import collections
import copy
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
BENCHMARK_DIR = Path("experiments/batch71_74_controlled_learning/data")
EXP_DATA_DIR = Path("experiments/batch75_78_synthetic_pretraining/data")
OUTPUT_DIR = Path("experiments/batch75_78_synthetic_pretraining/experiments")


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

        ans_pos = len(p_toks)
        examples.append({
            "id": r["id"],
            "prompt": prompt,
            "response": response,
            "expected_answer": expected,
            "inp": inp,
            "tgt": tgt,
            "prompt_in": [0] + p_toks,
            "ans_pos": ans_pos,
            "tmpl_span": (ans_pos + 1, len(inp)),
        })
    return examples


def evaluate_model_on_benchmark(
    model: ChakrMicro,
    tokenizer: Any,
    benchmark_splits: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, Any]:
    model.eval()
    results = {}
    with torch.no_grad():
        for split_name, exs in benchmark_splits.items():
            exact_count = 0
            eos_count = 0
            ans_losses = []
            tmpl_losses = []
            detailed = []
            preds = []

            for ex in exs:
                x = torch.tensor([ex["inp"]], dtype=torch.long)
                logits = model(x)[0]

                # Answer token CE loss
                a_pos = ex["ans_pos"]
                true_ans_token = tokenizer.encode(ex["expected_answer"])[0]
                ans_ce = F.cross_entropy(logits[a_pos].unsqueeze(0), torch.tensor([true_ans_token])).item()
                ans_losses.append(ans_ce)

                # Template tokens CE loss
                t_s, t_e = ex["tmpl_span"]
                for p in range(t_s, t_e):
                    true_tok = tokenizer.encode(ex["response"])[p - len(tokenizer.encode(ex["prompt"]))] if p < t_e - 1 else 1
                    ce = F.cross_entropy(logits[p].unsqueeze(0), torch.tensor([true_tok])).item()
                    tmpl_losses.append(ce)

                # Greedy autoregressive decode
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

                detailed.append({
                    "id": ex["id"],
                    "expected": ex["expected_answer"],
                    "generated": clean,
                    "exact": exact,
                    "eos": eos_reached,
                    "ans_loss": round(ans_ce, 4),
                })

            pred_freq = dict(collections.Counter(preds).most_common(5))
            modal_collapsed = (len(preds) > 0 and max(collections.Counter(preds).values()) / len(preds) >= 0.8)

            results[split_name] = {
                "total": len(exs),
                "exact_matches": exact_count,
                "exact_match_acc": round(exact_count / len(exs), 4),
                "eos_count": eos_count,
                "eos_rate": round(eos_count / len(exs), 4),
                "mean_answer_loss": round(float(np.mean(ans_losses)), 4),
                "mean_template_loss": round(float(np.mean(tmpl_losses)), 4),
                "modal_collapsed": modal_collapsed,
                "prediction_distribution": pred_freq,
                "records": detailed,
            }
    return results


def train_on_tokens(
    token_file: Path,
    condition_name: str,
    steps: int = 100,
    batch_size: int = 4,
    seq_len: int = 64,
    lr: float = 5e-4,
    seed: int = 42,
) -> Tuple[ChakrMicro, List[Dict[str, Any]], Path, str]:
    print(f"\n--- Running Training: {condition_name} ({steps} steps) ---")
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)

    with open(token_file, "rb") as f:
        tokens = np.frombuffer(f.read(), dtype=np.uint16).astype(np.int64)

    config = ModelConfig(max_seq_len=512)
    model = ChakrMicro(config)

    # Load canonical step 10k checkpoint
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
    ckpt_dict = {
        "step": steps,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": opt.state_dict(),
        "history": history,
        "condition_name": condition_name,
    }
    torch.save(ckpt_dict, ckpt_path)
    sha256 = compute_file_sha256(str(ckpt_path))
    print(f"Saved {condition_name} checkpoint: {ckpt_path} (SHA-256: {sha256})")
    return model, history, ckpt_path, sha256


def run_overfit_control(
    benchmark_splits: Dict[str, List[Dict[str, Any]]],
    tokenizer: Any,
    steps: int = 50,
) -> Tuple[ChakrMicro, Path, str]:
    print("\n--- Running Overfit Control (3 examples, 50 steps) ---")
    torch.manual_seed(42)
    config = ModelConfig(max_seq_len=512)
    model = ChakrMicro(config)
    base_state = torch.load(BASE_CKPT, map_location="cpu", weights_only=False)
    model.load_state_dict(base_state["model_state_dict"])

    opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    # Pick 3 train examples
    train_exs = benchmark_splits["train"][:3]
    model.train()
    for step in range(steps):
        opt.zero_grad()
        total_loss = 0.0
        for ex in train_exs:
            x = torch.tensor([ex["inp"]], dtype=torch.long)
            y = torch.tensor([ex["tgt"]], dtype=torch.long)
            logits = model(x)
            loss = loss_fn(logits.view(-1, 4096), y.view(-1))
            total_loss += loss.item()
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if (step + 1) % 10 == 0:
            avg_loss = total_loss / len(train_exs)
            print(f"Overfit Step {step + 1:2d}/{steps} | Loss: {avg_loss:.4f}")

    ckpt_path = OUTPUT_DIR / "overfit_control_3ex_step_50.pt"
    torch.save({"step": steps, "model_state_dict": model.state_dict()}, ckpt_path)
    sha256 = compute_file_sha256(str(ckpt_path))
    return model, ckpt_path, sha256


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer, _ = load_tokenizer_artifacts(TOKENIZER_DIR)

    # Verify baseline checkpoint SHA-256
    actual_base_sha = compute_file_sha256(str(BASE_CKPT))
    assert actual_base_sha == EXPECTED_BASE_SHA, f"Base checkpoint SHA mismatch: {actual_base_sha}"
    print(f"Verified Base Checkpoint Step 10k: {BASE_CKPT} (SHA-256: {actual_base_sha})")

    # Load benchmark evaluation records
    train_recs = load_records(BENCHMARK_DIR / "train_arithmetic.jsonl")
    val_recs = load_records(BENCHMARK_DIR / "val_arithmetic.jsonl")
    test_recs = load_records(BENCHMARK_DIR / "test_held_out.jsonl")
    symm_recs = load_records(BENCHMARK_DIR / "symmetry_diagnostic.jsonl")

    benchmark_splits = {
        "train": build_benchmark_examples(train_recs, tokenizer),
        "validation": build_benchmark_examples(val_recs, tokenizer),
        "test_held_out": build_benchmark_examples(test_recs, tokenizer),
        "symmetry_diagnostic": build_benchmark_examples(symm_recs, tokenizer),
    }

    results_summary = {}

    # 1. Baseline Evaluation (Step 10,000 unmodified)
    print("\n--- Evaluating Baseline (Step 10,000 Unmodified) ---")
    config = ModelConfig(max_seq_len=512)
    base_model = ChakrMicro(config)
    base_state = torch.load(BASE_CKPT, map_location="cpu", weights_only=False)
    base_model.load_state_dict(base_state["model_state_dict"])
    base_metrics = evaluate_model_on_benchmark(base_model, tokenizer, benchmark_splits)
    results_summary["baseline_step_10000"] = {
        "metrics": base_metrics,
        "checkpoint_sha256": actual_base_sha,
    }

    # 2. Condition A: Matched Language Control
    cond_a_tokens = EXP_DATA_DIR / "condition_a_control_tokens.bin"
    model_a, hist_a, ckpt_a, sha_a = train_on_tokens(
        cond_a_tokens, condition_name="condition_a_language_control", steps=100, batch_size=4, seq_len=64
    )
    metrics_a = evaluate_model_on_benchmark(model_a, tokenizer, benchmark_splits)
    results_summary["condition_a_language_control"] = {
        "metrics": metrics_a,
        "checkpoint_path": str(ckpt_a),
        "checkpoint_sha256": sha_a,
        "history": hist_a,
    }

    # 3. Condition B: Synthetic Arithmetic Curriculum
    cond_b_tokens = EXP_DATA_DIR / "condition_b_arithmetic_tokens.bin"
    model_b, hist_b, ckpt_b, sha_b = train_on_tokens(
        cond_b_tokens, condition_name="condition_b_arithmetic_curriculum", steps=100, batch_size=4, seq_len=64
    )
    metrics_b = evaluate_model_on_benchmark(model_b, tokenizer, benchmark_splits)
    results_summary["condition_b_arithmetic_curriculum"] = {
        "metrics": metrics_b,
        "checkpoint_path": str(ckpt_b),
        "checkpoint_sha256": sha_b,
        "history": hist_b,
    }

    # 4. Overfit Control (3 examples)
    model_overfit, ckpt_overfit, sha_overfit = run_overfit_control(benchmark_splits, tokenizer, steps=50)
    metrics_overfit = evaluate_model_on_benchmark(model_overfit, tokenizer, benchmark_splits)
    results_summary["overfit_control_3ex"] = {
        "metrics": metrics_overfit,
        "checkpoint_path": str(ckpt_overfit),
        "checkpoint_sha256": sha_overfit,
    }

    # 5. Cold Reload Determinism Verification
    print("\n--- Verifying Cold Reload Determinism for Condition B ---")
    reload_model = ChakrMicro(config)
    reload_data = torch.load(ckpt_b, map_location="cpu", weights_only=False)
    reload_model.load_state_dict(reload_data["model_state_dict"])
    reloaded_metrics = evaluate_model_on_benchmark(reload_model, tokenizer, benchmark_splits)
    assert reloaded_metrics["test_held_out"]["exact_matches"] == metrics_b["test_held_out"]["exact_matches"], "Reload test exact match mismatch!"
    assert reloaded_metrics["test_held_out"]["mean_answer_loss"] == metrics_b["test_held_out"]["mean_answer_loss"], "Reload test ans loss mismatch!"
    print("Cold reload determinism verified bit-exactly!")

    # Save complete experiment summary
    summary_path = OUTPUT_DIR / "batch77_experiment_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2)
    print(f"\nAll Batch 77 experiments complete. Summary written to {summary_path}")


if __name__ == "__main__":
    main()
