"""
UCIP-25 Model Evaluation Engine:
Evaluates Step-800 and UCIP-20 checkpoints across all 1,000 UCIP-23 benchmark records.
Generates predictions under greedy decoding (T=0.0, max_new_tokens=16).
Records:
- Raw predictions and generated tokens
- Exact match scoring
- Per-task accuracy and confidence intervals
- Per-split accuracy
- Error categorization
"""

from __future__ import annotations
import glob
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer.serialization import load_tokenizer_artifacts


def load_model(checkpoint_path: Path, device: str = "cpu") -> ChakrMicro:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ModelConfig(**ckpt["config"])
    model = ChakrMicro(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def generate_greedy(
    model: ChakrMicro,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int = 16,
    device: str = "cpu",
    bos_id: int = 0,
    eos_id: int = 1,
) -> Tuple[List[int], str]:
    inp = [bos_id] + tokenizer.encode(prompt)
    curr = torch.tensor([inp], dtype=torch.long, device=device)
    generated: List[int] = []

    with torch.no_grad():
        for _ in range(max_new_tokens):
            if curr.shape[1] >= model.config.max_seq_len:
                break
            logits = model(curr)[0, -1, :]
            nxt = int(torch.argmax(logits).item())
            if nxt == eos_id:
                break
            generated.append(nxt)
            curr = torch.cat([curr, torch.tensor([[nxt]], dtype=torch.long, device=device)], dim=1)

    text = tokenizer.decode(generated, errors="replace").strip()
    return generated, text


def compute_wilson_ci(k: int, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    z = 1.95996  # for 95%
    p = k / n
    denominator = 1 + z**2 / n
    centre_adjusted_probability = p + z**2 / (2 * n)
    adjusted_std = math.sqrt((p * (1 - p) + z**2 / (4 * n)) / n)
    lower = (centre_adjusted_probability - z * adjusted_std) / denominator
    upper = (centre_adjusted_probability + z * adjusted_std) / denominator
    return max(0.0, lower), min(1.0, upper)


def run_evaluation():
    device = "cpu"
    tok, _ = load_tokenizer_artifacts(ROOT / "data/experiments/vocab_4096")

    # 1. Load benchmark dataset
    records = []
    dataset_files = sorted(glob.glob("experiments/ucip23/datasets/task_*.jsonl"))
    for fpath in dataset_files:
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))

    print(f"Loaded {len(records)} benchmark records across Tasks A-H.")

    # Checkpoints to evaluate
    checkpoints = {
        "step800": Path("experiments/ucip13/checkpoints/primary_seed42/checkpoint_step_800.pt"),
        "ucip20_best": Path("experiments/ucip20/checkpoints/checkpoint_best.pt")
    }

    all_scores = {}

    for ckpt_key, ckpt_path in checkpoints.items():
        print(f"\nEvaluating Checkpoint: {ckpt_key} ({ckpt_path})...")
        model = load_model(ckpt_path, device=device)

        predictions = []
        task_correct = Counter()
        task_total = Counter()
        split_correct = Counter()
        split_total = Counter()
        error_categories = Counter()
        sample_errors = []

        for idx, rec in enumerate(records):
            if (idx + 1) % 200 == 0:
                print(f"  Processed {idx + 1}/{len(records)} records...")

            rid = rec["id"]
            t_type = rec["task_type"]
            split = rec.get("split", "unknown")
            prompt = rec["prompt"]
            target = rec["target_response"].strip()

            gen_toks, gen_text = generate_greedy(model, tok, prompt, max_new_tokens=16, device=device)

            # Strict Exact Match
            is_correct = (gen_text.strip().lower() == target.lower())

            task_total[t_type] += 1
            split_total[split] += 1
            if is_correct:
                task_correct[t_type] += 1
                split_correct[split] += 1
            else:
                # Classify error
                if not gen_text.strip():
                    cat = "Empty output"
                elif t_type == "task_h_abstention" and "abstain" not in gen_text.lower() and "not provided" not in gen_text.lower():
                    cat = "Failure to abstain"
                elif t_type == "task_d_multi_relation_binding":
                    # Check if model returned another value belonging to the entity
                    cat = "Relational crosstalk or wrong attribute"
                elif t_type == "task_g_distractor_resistance":
                    # Check if distractor value was emitted
                    cat = "Distractor intrusion or wrong value"
                else:
                    cat = "Incorrect value or attractor output"

                error_categories[cat] += 1
                if len(sample_errors) < 20:
                    sample_errors.append({
                        "id": rid,
                        "task_type": t_type,
                        "prompt_preview": prompt[:80] + "...",
                        "target": target,
                        "prediction": gen_text,
                        "category": cat
                    })

            predictions.append({
                "id": rid,
                "task_type": t_type,
                "split": split,
                "target": target,
                "prediction": gen_text,
                "prediction_tokens": gen_toks,
                "correct": is_correct
            })

        # Save raw predictions
        pred_file = f"experiments/ucip25/predictions/{ckpt_key}_predictions.json"
        with open(pred_file, "w", encoding="utf-8") as f:
            json.dump(predictions, f, indent=2)

        # Compute task metrics
        task_metrics = {}
        for t in sorted(task_total.keys()):
            k = task_correct[t]
            n = task_total[t]
            acc = k / n if n > 0 else 0.0
            ci_low, ci_high = compute_wilson_ci(k, n)
            task_metrics[t] = {
                "correct": k,
                "total": n,
                "accuracy": acc,
                "ci_95": [round(ci_low, 4), round(ci_high, 4)]
            }

        # Compute split metrics
        split_metrics = {}
        for s in sorted(split_total.keys()):
            k = split_correct[s]
            n = split_total[s]
            split_metrics[s] = {
                "correct": k,
                "total": n,
                "accuracy": k / n if n > 0 else 0.0
            }

        overall_correct = sum(task_correct.values())
        overall_total = sum(task_total.values())
        overall_acc = overall_correct / overall_total
        ov_low, ov_high = compute_wilson_ci(overall_correct, overall_total)

        summary = {
            "checkpoint": ckpt_key,
            "overall_accuracy": overall_acc,
            "overall_ci_95": [round(ov_low, 4), round(ov_high, 4)],
            "task_metrics": task_metrics,
            "split_metrics": split_metrics,
            "error_categories": dict(error_categories),
            "sample_errors": sample_errors
        }

        all_scores[ckpt_key] = summary

        score_file = f"experiments/ucip25/scores/{ckpt_key}_scores.json"
        with open(score_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print(f"Done {ckpt_key}: Overall Accuracy = {overall_acc*100:.2f}% ({overall_correct}/{overall_total})")
        for t, m in task_metrics.items():
            print(f"  {t}: {m['accuracy']*100:.2f}% (CI: {m['ci_95'][0]*100:.1f}% - {m['ci_95'][1]*100:.1f}%)")

    # Save consolidated score summary
    consolidated_file = "experiments/ucip25/scores/consolidated_scores.json"
    with open(consolidated_file, "w", encoding="utf-8") as f:
        json.dump(all_scores, f, indent=2)

    print("\nModel Evaluation Complete. All scores and predictions preserved.")


if __name__ == "__main__":
    run_evaluation()
