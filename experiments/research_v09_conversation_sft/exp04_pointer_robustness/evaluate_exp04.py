"""
evaluate_exp04.py
ChakrView Exp04 — Independent Pointer Robustness Evaluation Script.

Executes the frozen Exp15 field-conditioned pointer on the 120-example
independent robustness challenge dataset.

Reports:
- Overall exact-span accuracy (numerator / denominator)
- Category-level accuracy
- Genuine-abstention accuracy, precision, recall, and confusion matrix
- Start and end boundary accuracy
- Positional and prompt-length breakdown
- Per-example diagnostic results into PER_EXAMPLE_RESULTS.jsonl
- Emits CATEGORY_METRICS.json and ROBUSTNESS_REPORT.md
"""

from __future__ import annotations

import json
import time
from pathlib import Path
import sys
from typing import Dict, Any, List

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

EXP04_DIR = Path(__file__).resolve().parent
DATA_DIR = EXP04_DIR / "data"
CANONICAL_TOK_DIR = REPO_ROOT / "data" / "experiments" / "vocab_4096"
EXP15_CKPT_PATH = REPO_ROOT / "experiments" / "research_v09_conversation_sft" / "exp03_field_conditioned_pointer" / "checkpoints" / "pointer_best.pt"

from chakrview.tokenizer.serialization import load_tokenizer_artifacts
from experiments.research_v09_conversation_sft.exp03_field_conditioned_pointer.pointer_attention import (
    FieldConditionedSpanPointer,
    build_field_conditioned_pointer
)

def run_evaluation():
    print("=" * 70)
    print("CHAKRVIEW EXP04: INDEPENDENT POINTER ROBUSTNESS EVALUATION")
    print("=" * 70)

    tokenizer, _ = load_tokenizer_artifacts(CANONICAL_TOK_DIR, verify_checksums=False)

    # Load 120 challenge examples
    dataset_path = DATA_DIR / "exp04_challenge.jsonl"
    records = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    total_samples = len(records)
    print(f"Loaded {total_samples} challenge examples from {dataset_path.name}")

    # Build model and load trained Exp15 pointer weights
    model = build_field_conditioned_pointer(device="cpu")
    ckpt = torch.load(EXP15_CKPT_PATH, map_location="cpu", weights_only=False)
    model.field_embedding.load_state_dict(ckpt["field_embedding"])
    model.query_start.load_state_dict(ckpt["query_start"])
    model.key_start.load_state_dict(ckpt["key_start"])
    model.query_end.load_state_dict(ckpt["query_end"])
    model.key_end.load_state_dict(ckpt["key_end"])
    model.eval()
    print("Successfully loaded frozen Exp15 pointer checkpoint.")

    start_time = time.time()

    per_example_results = []
    cat_stats: Dict[str, Dict[str, int]] = {}

    exact_matches = 0
    start_matches = 0
    end_matches = 0

    # Abstention confusion matrix
    # Gold abstention: [0, 0]
    # Gold answerable: [s, e] with s > 0
    tp_abstain = 0  # Gold=Abstain, Pred=Abstain
    fp_abstain = 0  # Gold=Answerable, Pred=Abstain
    fn_abstain = 0  # Gold=Abstain, Pred=NonEmpty
    tn_abstain = 0  # Gold=Answerable, Pred=NonEmpty

    length_bands = {"short (<100)": {"total": 0, "correct": 0}, "medium (100-200)": {"total": 0, "correct": 0}, "long (>200)": {"total": 0, "correct": 0}}

    for rec in records:
        ex_id = rec["example_id"]
        cat = rec["category"]
        field_id = rec["field_id"]
        input_ids = torch.tensor(rec["input_ids"], dtype=torch.long)
        mask = torch.tensor(rec["candidate_mask"], dtype=torch.bool)
        gold_s = rec["start_pos"]
        gold_e = rec["end_pos"]
        target_text = rec["target_text"]
        is_abstention = rec["is_abstention"]

        if cat not in cat_stats:
            cat_stats[cat] = {"total": 0, "correct": 0, "abstentions": 0, "correct_abstentions": 0}
        cat_stats[cat]["total"] += 1
        if is_abstention:
            cat_stats[cat]["abstentions"] += 1

        seq_len = input_ids.shape[0]
        if seq_len < 100:
            l_band = "short (<100)"
        elif seq_len <= 200:
            l_band = "medium (100-200)"
        else:
            l_band = "long (>200)"
        length_bands[l_band]["total"] += 1

        # Predict span
        pred_s, pred_e, extracted_tokens, conf = model.predict_span(
            input_ids, field_id, mask, use_field_conditioning=True
        )

        is_exact = (pred_s == gold_s and pred_e == gold_e)
        is_start = (pred_s == gold_s)
        is_end = (pred_e == gold_e)

        if is_exact:
            exact_matches += 1
            cat_stats[cat]["correct"] += 1
            length_bands[l_band]["correct"] += 1

        if is_start:
            start_matches += 1
        if is_end:
            end_matches += 1

        # Abstention matrix
        pred_is_abstain = (pred_s == 0 and pred_e == 0)
        if is_abstention and pred_is_abstain:
            tp_abstain += 1
            cat_stats[cat]["correct_abstentions"] += 1
        elif not is_abstention and pred_is_abstain:
            fp_abstain += 1
        elif is_abstention and not pred_is_abstain:
            fn_abstain += 1
        else:
            tn_abstain += 1

        pred_text = "[NO_ANSWER]" if pred_is_abstain else tokenizer.decode(extracted_tokens).strip()

        failure_reason = None
        if not is_exact:
            if is_abstention and not pred_is_abstain:
                failure_reason = f"False Positive Span: Expected [0,0] Abstention, but predicted [{pred_s}, {pred_e}] ('{pred_text}')"
            elif not is_abstention and pred_is_abstain:
                failure_reason = f"False Negative Abstention: Expected [{gold_s}, {gold_e}] ('{target_text}'), but model abstained at [0,0]"
            else:
                failure_reason = f"Span Boundary Error: Predicted [{pred_s}, {pred_e}] ('{pred_text}') != Gold [{gold_s}, {gold_e}] ('{target_text}')"

        per_example_results.append({
            "example_id": ex_id,
            "category": cat,
            "field_id": field_id,
            "prompt_length_tokens": seq_len,
            "gold_span": [gold_s, gold_e],
            "pred_span": [pred_s, pred_e],
            "gold_text": target_text if not is_abstention else "[NO_ANSWER]",
            "pred_text": pred_text,
            "confidence": conf,
            "is_abstention": is_abstention,
            "pred_is_abstain": pred_is_abstain,
            "exact_match": is_exact,
            "start_match": is_start,
            "end_match": is_end,
            "failure_reason": failure_reason
        })

    duration = time.time() - start_time

    # Compute metrics
    overall_acc = exact_matches / total_samples
    start_acc = start_matches / total_samples
    end_acc = end_matches / total_samples

    total_abstain_cases = tp_abstain + fn_abstain
    abstain_recall = tp_abstain / total_abstain_cases if total_abstain_cases > 0 else 0.0
    abstain_prec = tp_abstain / (tp_abstain + fp_abstain) if (tp_abstain + fp_abstain) > 0 else 0.0
    abstain_acc = (tp_abstain + tn_abstain) / total_samples

    print("\n" + "=" * 70)
    print("EXP04 CHALLENGE EVALUATION RESULTS")
    print("=" * 70)
    print(f"Overall Exact-Span Accuracy : {exact_matches}/{total_samples} ({overall_acc * 100:.2f}%)")
    print(f"Start-Boundary Accuracy     : {start_matches}/{total_samples} ({start_acc * 100:.2f}%)")
    print(f"End-Boundary Accuracy       : {end_matches}/{total_samples} ({end_acc * 100:.2f}%)")
    print(f"Abstention Precision        : {abstain_prec * 100:.2f}% ({tp_abstain}/{tp_abstain + fp_abstain})")
    print(f"Abstention Recall           : {abstain_recall * 100:.2f}% ({tp_abstain}/{total_abstain_cases})")
    print(f"Pre-Registered Threshold    : 75.00%")
    print(f"Threshold Met?              : {'YES' if overall_acc >= 0.75 else 'NO'}")
    print("=" * 70)

    # Write per-example results
    per_example_file = EXP04_DIR / "PER_EXAMPLE_RESULTS.jsonl"
    with open(per_example_file, "w", encoding="utf-8") as f:
        for r in per_example_results:
            f.write(json.dumps(r) + "\n")

    # Category metrics payload
    category_metrics = {
        "experiment": "Exp04 Independent Pointer Robustness Challenge",
        "benchmark_sha256": "11c0b4bfc33410e113c9c112f35e50c8a9b4446ff93fb930f0f837b0ee9d9748",
        "total_samples": total_samples,
        "exact_matches": exact_matches,
        "overall_exact_accuracy": overall_acc,
        "acceptance_threshold": 0.75,
        "passed_acceptance_criteria": bool(overall_acc >= 0.75),
        "start_boundary_accuracy": start_acc,
        "end_boundary_accuracy": end_acc,
        "abstention_metrics": {
            "true_positives": tp_abstain,
            "false_positives": fp_abstain,
            "false_negatives": fn_abstain,
            "true_negatives": tn_abstain,
            "precision": abstain_prec,
            "recall": abstain_recall,
            "accuracy": abstain_acc
        },
        "length_band_breakdown": length_bands,
        "category_breakdown": cat_stats,
        "eval_duration_sec": duration
    }
    with open(EXP04_DIR / "CATEGORY_METRICS.json", "w", encoding="utf-8") as f:
        json.dump(category_metrics, f, indent=2)

    # Generate ROBUSTNESS_REPORT.md
    with open(EXP04_DIR / "ROBUSTNESS_REPORT.md", "w", encoding="utf-8") as f:
        f.write("# ChakrView Exp04 — Independent Pointer Robustness Report\n\n")
        f.write(f"**Status**: `{'ROBUSTNESS THRESHOLD MET — HUMAN REVIEW STILL REQUIRED' if overall_acc >= 0.75 else 'ROBUSTNESS THRESHOLD NOT MET'}`\n")
        f.write(f"**Date**: 2026-10-22\n")
        f.write(f"**Benchmark SHA-256**: `11c0b4bfc33410e113c9c112f35e50c8a9b4446ff93fb930f0f837b0ee9d9748`\n\n")
        f.write(f"## 1. Executive Summary\n\n")
        f.write(f"- Evaluated **{total_samples} independent conversational examples** across 10 categories.\n")
        f.write(f"- Overall Exact-Span Accuracy: **{exact_matches}/{total_samples} ({overall_acc*100:.2f}%)**.\n")
        f.write(rf"- Pre-registered threshold ($\ge 75.0\%$): **{'PASSED' if overall_acc >= 0.75 else 'FAILED'}**.\n\n")
        f.write(f"## 2. Category Performance Breakdown\n\n")
        f.write(f"| Category | Field ID | Total | Correct | Accuracy | Abstentions Correct |\n")
        f.write(f"| :--- | :---: | :---: | :---: | :---: | :---: |\n")
        for c, s in cat_stats.items():
            f.write(f"| `{c}` | - | {s['total']} | {s['correct']} | {s['correct']/s['total']*100:.1f}% | {s['correct_abstentions']}/{s['abstentions']} |\n")
        f.write(f"\n## 3. Abstention Analysis\n\n")
        f.write(f"- True Positives: {tp_abstain}\n- False Positives: {fp_abstain}\n- False Negatives: {fn_abstain}\n- True Negatives: {tn_abstain}\n")
        f.write(f"- Abstention Precision: **{abstain_prec*100:.2f}%**\n")
        f.write(f"- Abstention Recall: **{abstain_recall*100:.2f}%**\n\n")
        f.write(f"## 4. Diagnostics by Context Length\n\n")
        for b_name, b_stats in length_bands.items():
            pct = (b_stats['correct']/b_stats['total']*100) if b_stats['total'] > 0 else 0.0
            f.write(f"- `{b_name}`: {b_stats['correct']}/{b_stats['total']} ({pct:.1f}%)\n")

    print(f"Generated PER_EXAMPLE_RESULTS.jsonl, CATEGORY_METRICS.json, and ROBUSTNESS_REPORT.md")

if __name__ == "__main__":
    run_evaluation()
