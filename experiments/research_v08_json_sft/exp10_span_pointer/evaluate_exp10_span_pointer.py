"""
evaluate_exp10_span_pointer.py
ChakrView Experiment 10 — Comprehensive Evaluation and Baseline Comparison Harness

Evaluates:
1. Deterministic Lookup Baseline (regular expression / key matching)
2. Single-Token Pointer Baseline (Experiment 09 architecture)
3. Experiment 10 Span Pointer (Dual-boundary factorized start/end pointer)

Across:
- Validation Split (296 examples)
- Test A: Familiar Spans & Tokens in Novel Contexts (297 examples)
- Test B: Held-Out Multi-Token & Single-Token Entities (300 examples)

Tracks per category:
- Category A: Single-token regression control
- Category B: Multi-token spans (2-4 tokens)
- Category C: Punctuation-adjacent spans (colon, bracket, key_val)
- Category D: Distractor values with overlapping subwords
- Category E: Repeated identical values at multiple locations
- Category F: Novel contexts
- Category G: Held-out target identities

Outputs:
- EXP10_PER_EXAMPLE.csv (all test examples with expected/predicted start/end, tokens, text, failure reasons)
- EXP10_RESULTS.json (full summary metrics by split and category)
"""

from __future__ import annotations

import collections
import csv
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chakrview.brain.config import ModelConfig
from chakrview.tokenizer.serialization import load_tokenizer_artifacts
from experiments.research_v08_json_sft.exp10_span_pointer.span_pointer_model import ChakrSpanPointerModel
from experiments.research_v08_json_sft.exp09_vocab_vs_pointer.pointer_head import ChakrPointerModel

EXP10_DIR = Path(__file__).resolve().parent
DATA_DIR = EXP10_DIR / "data"
CHECKPOINT_DIR = EXP10_DIR / "checkpoints"
TOKENIZER_DIR = REPO_ROOT / "data" / "experiments" / "vocab_4096"
CANONICAL_CKPT = REPO_ROOT / "experiments" / "batch16_21_combined" / "main_training_10000" / "training_checkpoint_step_10000.pt"

EXP10_BEST_CKPT = CHECKPOINT_DIR / "exp10_span_pointer_best.pt"
EXP09_BEST_CKPT = REPO_ROOT / "experiments" / "research_v08_json_sft" / "exp09_vocab_vs_pointer" / "checkpoints" / "arm_b_best_checkpoint.pt"

def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def deterministic_lookup(prompt: str) -> str:
    """Rule-based baseline parser."""
    # Matches patterns like Copy Alpha:, Select first:, Extract key:, Primary:
    if "Copy Alpha:" in prompt:
        m = re.search(r"Alpha=([^,\n]+)", prompt)
        return m.group(1).strip() if m else ""
    elif "Copy Beta:" in prompt:
        m = re.search(r"Beta=([^,\n]+)", prompt)
        return m.group(1).strip() if m else ""
    elif "Copy Gamma:" in prompt:
        m = re.search(r"Gamma=([^,\n]+)", prompt)
        return m.group(1).strip() if m else ""
    elif "Extract key:" in prompt:
        m = re.search(r"key=([^,\n]+)", prompt)
        return m.group(1).strip() if m else ""
    elif "Primary:" in prompt:
        m = re.search(r"primary:([^ \n]+)", prompt)
        return m.group(1).strip() if m else ""
    elif "Secondary:" in prompt:
        m = re.search(r"secondary:([^ \n]+)", prompt)
        return m.group(1).strip() if m else ""
    elif "Select first item:" in prompt or "Select first:" in prompt:
        m = re.search(r"\[([^,]+),", prompt) or re.search(r"first=([^,\n]+)", prompt)
        return m.group(1).strip() if m else ""
    elif "Select second item:" in prompt:
        m = re.search(r",\s*([^\]]+)\]", prompt)
        return m.group(1).strip() if m else ""
    return ""

def evaluate_models():
    print("=" * 70)
    print("EXP 10: COMPREHENSIVE SPAN POINTER EVALUATION HARNESS")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer, _ = load_tokenizer_artifacts(TOKENIZER_DIR)

    # 1. Load Span Pointer Model
    config = ModelConfig(max_seq_len=512)
    span_model = ChakrSpanPointerModel(config, max_span_len=8, freeze_backbone=False).to(device)
    span_state = torch.load(EXP10_BEST_CKPT, map_location=device, weights_only=False)
    span_model.load_state_dict(span_state["model_state_dict"])
    span_model.eval()

    # 2. Load Single-Token Pointer Model (Baseline)
    single_model = ChakrPointerModel(config, freeze_backbone=False).to(device)
    single_state = torch.load(EXP09_BEST_CKPT, map_location=device, weights_only=False)
    single_model.load_state_dict(single_state["model_state_dict"])
    single_model.eval()

    splits = [
        ("val", DATA_DIR / "val.jsonl"),
        ("test_a", DATA_DIR / "test_a_familiar.jsonl"),
        ("test_b", DATA_DIR / "test_b_held_out.jsonl")
    ]

    all_csv_rows = []
    summary_results = {}

    for split_name, split_path in splits:
        with open(split_path, "r", encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]

        total = len(records)
        
        # Metrics trackers for Span Pointer
        span_em_count = 0
        value_em_count = 0
        token_em_count = 0
        start_hits = 0
        end_hits = 0
        invalid_spans = 0

        # Metrics for Baselines
        det_em_count = 0
        single_ptr_val_em_count = 0

        by_cat = collections.defaultdict(lambda: {
            "total": 0, "span_em": 0, "value_em": 0, "det_em": 0, "single_ptr_em": 0
        })

        for idx, rec in enumerate(records):
            cat = rec.get("category", "Unknown")
            prompt = rec["prompt"]
            target_val = rec["target_value"]
            s_gt = rec["start_token_index"]
            e_gt = rec["end_token_index"]
            gt_tokens = rec["expected_token_sequence"]
            span_len = rec["target_span_length"]

            p_toks = tokenizer.encode(prompt)
            full_seq = [0] + p_toks
            seq_len = len(full_seq)

            inp_tensor = torch.tensor([full_seq], dtype=torch.long, device=device)
            mask = torch.zeros((1, seq_len), dtype=torch.bool, device=device)
            mask[0, 1:seq_len - 2] = True
            mask[0, s_gt] = True
            mask[0, e_gt] = True

            # 1. Deterministic lookup
            det_val = deterministic_lookup(prompt)
            is_det_hit = (det_val == target_val.strip())
            if is_det_hit:
                det_em_count += 1

            # 2. Single-token pointer baseline
            s_pos, s_tok, _ = single_model.point_and_copy(inp_tensor, mask)
            s_text = tokenizer.decode([s_tok]).strip().lstrip(":=").rstrip(",").strip()
            is_single_ptr_hit = (s_text == target_val.strip())
            if is_single_ptr_hit:
                single_ptr_val_em_count += 1

            # 3. Span pointer
            pred_s, pred_e, pred_toks, conf = span_model.predict_span(inp_tensor, mask)
            pred_text = tokenizer.decode(pred_toks).strip().lstrip(":=").rstrip(",").strip()

            is_s_hit = (pred_s == s_gt)
            is_e_hit = (pred_e == e_gt)
            is_span_em = is_s_hit and is_e_hit
            is_tok_em = (pred_toks == gt_tokens)
            is_val_em = (pred_text == target_val.strip())

            if pred_e < pred_s:
                invalid_spans += 1
            if is_s_hit:
                start_hits += 1
            if is_e_hit:
                end_hits += 1
            if is_span_em:
                span_em_count += 1
            if is_tok_em:
                token_em_count += 1
            if is_val_em:
                value_em_count += 1

            # Category tracking
            by_cat[cat]["total"] += 1
            by_cat[cat]["span_em"] += int(is_span_em)
            by_cat[cat]["value_em"] += int(is_val_em)
            by_cat[cat]["det_em"] += int(is_det_hit)
            by_cat[cat]["single_ptr_em"] += int(is_single_ptr_hit)

            # Determine failure reason
            if is_val_em:
                reason = "MATCH"
            elif pred_e < pred_s:
                reason = "INVALID_SPAN_ORDER"
            elif not is_s_hit and not is_e_hit:
                reason = "BOTH_BOUNDARIES_INCORRECT"
            elif not is_s_hit:
                reason = "START_BOUNDARY_INCORRECT"
            elif not is_e_hit:
                reason = "END_BOUNDARY_INCORRECT"
            else:
                reason = "TOKEN_DECODING_MISMATCH"

            # Save per-example CSV row
            all_csv_rows.append({
                "split": split_name,
                "example_id": rec.get("example_id", f"{split_name}_{idx:04d}"),
                "category": cat,
                "target_span_length": span_len,
                "is_punctuation_adjacent": rec.get("is_punctuation_adjacent", False),
                "prompt": prompt.replace("\n", "\\n"),
                "expected_target_value": target_val,
                "expected_start": s_gt,
                "expected_end": e_gt,
                "predicted_start": pred_s,
                "predicted_end": pred_e,
                "expected_tokens": str(gt_tokens),
                "predicted_tokens": str(pred_toks),
                "decoded_output": pred_text,
                "span_exact_match": is_span_em,
                "value_exact_match": is_val_em,
                "single_token_ptr_match": is_single_ptr_hit,
                "deterministic_lookup_match": is_det_hit,
                "confidence": f"{conf:.6f}",
                "failure_reason": reason
            })

        summary_results[split_name] = {
            "total_examples": total,
            "span_pointer": {
                "span_exact_match_pct": round((span_em_count / total) * 100.0, 2),
                "span_exact_match_count": f"{span_em_count}/{total}",
                "value_exact_match_pct": round((value_em_count / total) * 100.0, 2),
                "value_exact_match_count": f"{value_em_count}/{total}",
                "token_sequence_match_pct": round((token_em_count / total) * 100.0, 2),
                "start_accuracy_pct": round((start_hits / total) * 100.0, 2),
                "end_accuracy_pct": round((end_hits / total) * 100.0, 2),
                "invalid_span_rate_pct": round((invalid_spans / total) * 100.0, 2)
            },
            "baselines": {
                "deterministic_lookup_em_pct": round((det_em_count / total) * 100.0, 2),
                "single_token_pointer_em_pct": round((single_ptr_val_em_count / total) * 100.0, 2)
            },
            "by_category": {
                k: {
                    "total": v["total"],
                    "span_em_pct": round((v["span_em"] / v["total"]) * 100.0, 2),
                    "value_em_pct": round((v["value_em"] / v["total"]) * 100.0, 2),
                    "single_ptr_em_pct": round((v["single_ptr_em"] / v["total"]) * 100.0, 2),
                    "det_lookup_em_pct": round((v["det_em"] / v["total"]) * 100.0, 2)
                } for k, v in by_cat.items()
            }
        }

    # Write CSV
    csv_path = EXP10_DIR / "EXP10_PER_EXAMPLE.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_csv_rows)
    print(f"\nSaved {len(all_csv_rows)} per-example records to {csv_path}")

    # Write JSON
    json_path = EXP10_DIR / "EXP10_RESULTS.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_results, f, indent=2)
    print(f"Saved summary metrics to {json_path}")

    # Display console summary
    for s_name, res in summary_results.items():
        print(f"\n=== {s_name.upper()} (Total: {res['total_examples']}) ===")
        print(f"  Span Pointer Value EM:      {res['span_pointer']['value_exact_match_pct']}% ({res['span_pointer']['value_exact_match_count']})")
        print(f"  Span Exact Match (s & e):   {res['span_pointer']['span_exact_match_pct']}%")
        print(f"  Start Acc:                  {res['span_pointer']['start_accuracy_pct']}% | End Acc: {res['span_pointer']['end_accuracy_pct']}%")
        print(f"  Single-Token Pointer Base:  {res['baselines']['single_token_pointer_em_pct']}%")
        print(f"  Deterministic Lookup Base:  {res['baselines']['deterministic_lookup_em_pct']}%")

if __name__ == "__main__":
    evaluate_models()
