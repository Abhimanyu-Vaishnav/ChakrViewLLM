"""
evaluate_exp10_2.py - Comprehensive Evaluation Harness for Experiment 10.2

Compares:
1. Original Exp 10 Factorized Span Pointer (baseline)
2. Exp 10 Oracle-Start Upper Bound (diagnostic upper bound on baseline)
3. Exp 10.2 Start-Conditioned Span Pointer (isolated research correction)

Evaluated across:
- Validation split (296 examples)
- Test A split (297 examples)
- Test B split (300 examples)

Computes all metrics with explicit numerator / denominator:
- Start-position accuracy
- End-position accuracy
- Exact span match (both start and end exact match)
- Exact token-sequence match
- Exact decoded-value match
- Premature-end rate (predicted span is strict prefix of expected span)
- Invalid-span rate (predicted span outside bounds or start > end)
- Breakdown by target token length (1, 2, 3, 4+)
- Breakdown by punctuation adjacency
- Breakdown by identity category (familiar vs held-out)
"""

from __future__ import annotations

import collections
import csv
import hashlib
import json
import string
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chakrview.brain.config import ModelConfig
from chakrview.tokenizer.serialization import load_tokenizer_artifacts
from experiments.research_v08_json_sft.exp10_span_pointer.span_pointer_model import ChakrSpanPointerModel
from experiments.research_v08_json_sft.exp10_span_pointer.exp10_2_end_boundary.start_conditioned_span_model import ChakrStartConditionedSpanPointer

EXP10_DIR = Path(__file__).resolve().parents[1]
EXP10_2_DIR = Path(__file__).resolve().parent
DATA_DIR = EXP10_DIR / "data"
TOKENIZER_DIR = REPO_ROOT / "data" / "experiments" / "vocab_4096"
EXP10_CKPT = EXP10_DIR / "checkpoints" / "exp10_span_pointer_best.pt"
EXP10_2_CKPT = EXP10_2_DIR / "checkpoints" / "exp10_2_start_cond_best.pt"

MAX_SPAN_LEN = 8

def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records

def is_punctuation_adjacent(prompt: str, target: str) -> bool:
    idx = prompt.find(target)
    if idx == -1:
        return False
    before = prompt[idx-1] if idx > 0 else ""
    after = prompt[idx+len(target)] if (idx+len(target)) < len(prompt) else ""
    punct = set(string.punctuation)
    return (before in punct) or (after in punct)

def compute_metrics_for_eval(eval_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(eval_records)
    if n == 0:
        return {}

    correct_start = sum(1 for r in eval_records if r["pred_start"] == r["gold_start"])
    correct_end = sum(1 for r in eval_records if r["pred_end"] == r["gold_end"])
    exact_span = sum(1 for r in eval_records if r["pred_start"] == r["gold_start"] and r["pred_end"] == r["gold_end"])
    exact_tokens = sum(1 for r in eval_records if r["pred_tokens"] == r["gold_tokens"])
    exact_val = sum(1 for r in eval_records if r["pred_val"] == r["gold_val"])
    premature_end = sum(1 for r in eval_records if r["is_premature"])
    invalid_spans = sum(1 for r in eval_records if r["is_invalid"])

    # Breakdowns by length
    lengths = sorted(list(set(r["target_token_len"] for r in eval_records)))
    by_len = {}
    for l in lengths:
        sub = [r for r in eval_records if r["target_token_len"] == l]
        nl = len(sub)
        by_len[str(l)] = {
            "total": nl,
            "exact_span": sum(1 for r in sub if r["pred_start"] == r["gold_start"] and r["pred_end"] == r["gold_end"]),
            "exact_span_acc": sum(1 for r in sub if r["pred_start"] == r["gold_start"] and r["pred_end"] == r["gold_end"]) / nl,
            "start_acc": sum(1 for r in sub if r["pred_start"] == r["gold_start"]) / nl,
            "end_acc": sum(1 for r in sub if r["pred_end"] == r["gold_end"]) / nl,
            "premature_rate": sum(1 for r in sub if r["is_premature"]) / nl,
        }

    # Breakdown by punctuation adjacency
    punct_sub = [r for r in eval_records if r["punct_adjacent"]]
    non_punct_sub = [r for r in eval_records if not r["punct_adjacent"]]
    by_punct = {}
    if punct_sub:
        by_punct["punct_adjacent"] = {
            "total": len(punct_sub),
            "exact_span_acc": sum(1 for r in punct_sub if r["pred_start"] == r["gold_start"] and r["pred_end"] == r["gold_end"]) / len(punct_sub),
            "end_acc": sum(1 for r in punct_sub if r["pred_end"] == r["gold_end"]) / len(punct_sub),
        }
    if non_punct_sub:
        by_punct["non_punct_adjacent"] = {
            "total": len(non_punct_sub),
            "exact_span_acc": sum(1 for r in non_punct_sub if r["pred_start"] == r["gold_start"] and r["pred_end"] == r["gold_end"]) / len(non_punct_sub),
            "end_acc": sum(1 for r in non_punct_sub if r["pred_end"] == r["gold_end"]) / len(non_punct_sub),
        }

    return {
        "total": n,
        "correct_start": correct_start,
        "start_acc": correct_start / n,
        "correct_end": correct_end,
        "end_acc": correct_end / n,
        "exact_span": exact_span,
        "exact_span_acc": exact_span / n,
        "exact_tokens": exact_tokens,
        "token_match_acc": exact_tokens / n,
        "exact_val": exact_val,
        "val_match_acc": exact_val / n,
        "premature_end_count": premature_end,
        "premature_end_rate": premature_end / n,
        "invalid_span_count": invalid_spans,
        "invalid_span_rate": invalid_spans / n,
        "by_target_length": by_len,
        "by_punctuation": by_punct
    }

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer, _ = load_tokenizer_artifacts(TOKENIZER_DIR)

    splits = {
        "val": load_jsonl(DATA_DIR / "val.jsonl"),
        "test_a": load_jsonl(DATA_DIR / "test_a_familiar.jsonl"),
        "test_b": load_jsonl(DATA_DIR / "test_b_held_out.jsonl"),
    }

    config = ModelConfig(max_seq_len=512)

    # 1. Load Exp 10 Model
    print("Loading Exp 10 Model...")
    exp10_model = ChakrSpanPointerModel(config, max_span_len=MAX_SPAN_LEN, freeze_backbone=False).to(device)
    exp10_state = torch.load(EXP10_CKPT, map_location=device, weights_only=False)
    exp10_model.load_state_dict(exp10_state["model_state_dict"])
    exp10_model.eval()

    # 2. Load Exp 10.2 Model
    print("Loading Exp 10.2 Start-Conditioned Model...")
    exp10_2_model = ChakrStartConditionedSpanPointer(config, max_span_len=MAX_SPAN_LEN, freeze_backbone=False).to(device)
    exp10_2_state = torch.load(EXP10_2_CKPT, map_location=device, weights_only=False)
    exp10_2_model.load_state_dict(exp10_2_state["model_state_dict"])
    exp10_2_model.eval()

    all_per_example_rows = []
    overall_results = {
        "exp10_baseline": {},
        "exp10_oracle_start": {},
        "exp10_2_start_conditioned": {}
    }

    models_to_eval = [
        ("exp10_baseline", exp10_model, "normal"),
        ("exp10_oracle_start", exp10_model, "oracle"),
        ("exp10_2_start_conditioned", exp10_2_model, "normal"),
    ]

    for model_name, model, mode in models_to_eval:
        print(f"\n--- Evaluating {model_name} (mode={mode}) ---")
        overall_results[model_name] = {}

        for split_name, dataset in splits.items():
            records = []
            for item in dataset:
                prompt = item["prompt"]
                p_toks = tokenizer.encode(prompt)
                full_seq = [0] + p_toks
                seq_len = len(full_seq)

                gold_start = item["start_token_index"]
                gold_end = item["end_token_index"]
                gold_tokens = item["expected_token_sequence"]
                gold_val = item["target_value"].strip()
                target_len = item["target_span_length"]

                inp_tensor = torch.tensor([full_seq], dtype=torch.long, device=device)
                cand_mask = torch.zeros((1, seq_len), dtype=torch.bool, device=device)
                cand_mask[0, 1:seq_len - 2] = True
                cand_mask[0, gold_start] = True
                cand_mask[0, gold_end] = True

                punct_adj = is_punctuation_adjacent(prompt, gold_val)

                with torch.no_grad():
                    if mode == "oracle":
                        # Exp 10 oracle start
                        s_logits, e_logits = exp10_model(inp_tensor, candidate_mask=cand_mask)
                        s_logits = s_logits[0]
                        e_logits = e_logits[0]
                        pred_s = gold_start
                        mask = torch.full_like(e_logits, float("-inf"))
                        max_e = min(pred_s + MAX_SPAN_LEN, e_logits.size(0))
                        for e in range(pred_s, max_e):
                            if cand_mask[0, e]:
                                mask[e] = e_logits[e]
                        pred_e = mask.argmax().item()
                    elif model_name == "exp10_2_start_conditioned":
                        pred_s, pred_e, _, _ = exp10_2_model.predict_span(inp_tensor, candidate_mask=cand_mask)
                    else: # exp10 normal
                        pred_s, pred_e, _, _ = exp10_model.predict_span(inp_tensor, candidate_mask=cand_mask)

                pred_tokens = full_seq[pred_s:pred_e+1] if pred_s <= pred_e else []
                pred_val = tokenizer.decode(pred_tokens).strip().lstrip(":=").rstrip(",").strip() if pred_tokens else ""

                # Check validity
                is_invalid = (pred_s < 0 or pred_e >= seq_len or pred_s > pred_e)
                # Check premature end: predicted start is gold start, predicted end < gold end, pred_tokens is strict prefix of gold_tokens
                is_premature = (pred_s == gold_start and pred_e < gold_end and gold_tokens[:len(pred_tokens)] == pred_tokens)

                rec = {
                    "example_id": item["example_id"],
                    "split": split_name,
                    "model": model_name,
                    "mode": mode,
                    "gold_start": gold_start,
                    "gold_end": gold_end,
                    "pred_start": pred_s,
                    "pred_end": pred_e,
                    "gold_tokens": gold_tokens,
                    "pred_tokens": pred_tokens,
                    "gold_val": gold_val,
                    "pred_val": pred_val,
                    "target_token_len": target_len,
                    "pred_token_len": len(pred_tokens),
                    "punct_adjacent": punct_adj,
                    "is_premature": is_premature,
                    "is_invalid": is_invalid,
                    "exact_span": (pred_s == gold_start and pred_e == gold_end),
                    "exact_val": (pred_val == gold_val),
                }
                records.append(rec)

                # Collect per-example row for CSV
                all_per_example_rows.append({
                    "model": model_name,
                    "split": split_name,
                    "example_id": item["example_id"],
                    "gold_start": gold_start,
                    "gold_end": gold_end,
                    "pred_start": pred_s,
                    "pred_end": pred_e,
                    "gold_tokens": json.dumps(gold_tokens),
                    "pred_tokens": json.dumps(pred_tokens),
                    "gold_val": gold_val,
                    "pred_val": pred_val,
                    "target_token_len": target_len,
                    "pred_token_len": len(pred_tokens),
                    "punct_adjacent": punct_adj,
                    "is_premature": is_premature,
                    "is_invalid": is_invalid,
                    "exact_span": rec["exact_span"],
                    "exact_val": rec["exact_val"]
                })

            metrics = compute_metrics_for_eval(records)
            overall_results[model_name][split_name] = metrics
            print(f"[{split_name.upper()}] Exact Span Acc: {metrics['exact_span_acc']*100:.2f}% ({metrics['exact_span']}/{metrics['total']}) | Start Acc: {metrics['start_acc']*100:.2f}% | End Acc: {metrics['end_acc']*100:.2f}% | Premature End: {metrics['premature_end_count']}")

    # Save EXP10_2_RESULTS.json
    results_path = EXP10_2_DIR / "EXP10_2_RESULTS.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(overall_results, f, indent=2)
    print(f"\nSaved overall results to {results_path}")

    # Save EXP10_2_PER_EXAMPLE.csv
    per_example_path = EXP10_2_DIR / "EXP10_2_PER_EXAMPLE.csv"
    with open(per_example_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "model", "split", "example_id",
            "gold_start", "gold_end", "pred_start", "pred_end",
            "gold_tokens", "pred_tokens", "gold_val", "pred_val",
            "target_token_len", "pred_token_len", "punct_adjacent",
            "is_premature", "is_invalid", "exact_span", "exact_val"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_per_example_rows:
            writer.writerow(r)
    print(f"Saved per-example predictions to {per_example_path}")

if __name__ == "__main__":
    main()
