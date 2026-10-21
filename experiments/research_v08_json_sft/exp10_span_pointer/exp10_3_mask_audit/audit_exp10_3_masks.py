"""
audit_exp10_3_masks.py
ChakrView Experiment 10.3 — Evaluation Mask Leakage Audit Harness

Audits:
- Experiment 10 evaluation & training masks
- Experiment 10.1 metric audit masks
- Experiment 10.2 failure audit & evaluation masks
- Corrected Pure Inference Mask (constructing eligibility strictly without gold labels)
- Corrected Oracle-Start Diagnostic (intervening ONLY on start position, zero end unlocking)

Computes side-by-side comparison of original vs corrected metrics across:
- Validation (296 examples)
- Test A (297 examples)
- Test B (300 examples)

Tracks:
- Start-position accuracy
- End-position accuracy
- Exact span match
- Exact token sequence match
- Exact decoded value match
- Premature-end rate
- Invalid / unrepresentable target rate
- Count of examples where gold start/end was excluded by the pure mask
- Number of predictions that changed under pure mask vs original mask
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

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chakrview.brain.config import ModelConfig
from chakrview.tokenizer.serialization import load_tokenizer_artifacts
from experiments.research_v08_json_sft.exp10_span_pointer.span_pointer_model import ChakrSpanPointerModel
from experiments.research_v08_json_sft.exp10_span_pointer.exp10_2_end_boundary.start_conditioned_span_model import ChakrStartConditionedSpanPointer

EXP10_DIR = Path(__file__).resolve().parents[1]
EXP10_3_DIR = Path(__file__).resolve().parent
DATA_DIR = EXP10_DIR / "data"
TOKENIZER_DIR = REPO_ROOT / "data" / "experiments" / "vocab_4096"
EXP10_CKPT = EXP10_DIR / "checkpoints" / "exp10_span_pointer_best.pt"
EXP10_2_CKPT = EXP10_DIR / "exp10_2_end_boundary" / "checkpoints" / "exp10_2_start_cond_best.pt"

MAX_SPAN_LEN = 8

def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

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

def build_original_mask(seq_len: int, s_gt: int, e_gt: int, device: str = "cpu") -> torch.Tensor:
    """Original mask construction from exp10 and exp10.2: 1:seq_len-2 with gold start and end explicitly set."""
    mask = torch.zeros((1, seq_len), dtype=torch.bool, device=device)
    mask[0, 1:seq_len - 2] = True
    mask[0, s_gt] = True
    mask[0, e_gt] = True
    return mask

def build_pure_inference_mask(seq_len: int, device: str = "cpu") -> torch.Tensor:
    """Corrected mask construction: candidate eligibility strictly derived from sequence length without gold labels."""
    mask = torch.zeros((1, seq_len), dtype=torch.bool, device=device)
    # Positions 1 to seq_len - 3 are prompt body (excluding BOS at 0 and final prompt punctuation ': ' at seq_len - 2 and seq_len - 1)
    mask[0, 1:seq_len - 2] = True
    return mask

def evaluate_split(
    model: Any,
    dataset: List[Dict[str, Any]],
    tokenizer: Any,
    split_name: str,
    device: str,
    is_start_conditioned: bool = False
) -> Dict[str, Any]:
    total = len(dataset)
    
    # Trackers for Original Mask
    orig_results = {
        "start_correct": 0, "end_correct": 0, "exact_span": 0,
        "exact_tokens": 0, "exact_val": 0, "premature_count": 0, "invalid_count": 0
    }
    # Trackers for Pure Inference Mask
    pure_results = {
        "start_correct": 0, "end_correct": 0, "exact_span": 0,
        "exact_tokens": 0, "exact_val": 0, "premature_count": 0, "invalid_count": 0
    }
    # Trackers for Corrected Oracle-Start Mode
    oracle_results = {
        "start_correct": total, "end_correct": 0, "exact_span": 0,
        "exact_tokens": 0, "exact_val": 0, "premature_count": 0, "invalid_count": 0
    }

    mask_strictly_identical_count = 0
    gold_excluded_by_pure_mask_count = 0
    predictions_differ_count = 0
    oracle_predictions_differ_count = 0

    per_example_rows = []

    for item in dataset:
        prompt = item["prompt"]
        p_toks = tokenizer.encode(prompt)
        full_seq = [0] + p_toks
        seq_len = len(full_seq)

        gold_s = item["start_token_index"]
        gold_e = item["end_token_index"]
        gold_tokens = item["expected_token_sequence"]
        gold_val = item["target_value"].strip()
        target_len = item["target_span_length"]
        punct_adj = is_punctuation_adjacent(prompt, gold_val)

        inp_tensor = torch.tensor([full_seq], dtype=torch.long, device=device)

        m_orig = build_original_mask(seq_len, gold_s, gold_e, device=device)
        m_pure = build_pure_inference_mask(seq_len, device=device)

        is_mask_identical = torch.equal(m_orig, m_pure)
        if is_mask_identical:
            mask_strictly_identical_count += 1

        gold_in_pure = (m_pure[0, gold_s].item() and m_pure[0, gold_e].item())
        if not gold_in_pure:
            gold_excluded_by_pure_mask_count += 1

        with torch.no_grad():
            # 1. Original Inference
            s_orig, e_orig, toks_orig, _ = model.predict_span(inp_tensor, candidate_mask=m_orig)
            val_orig = tokenizer.decode(toks_orig).strip().lstrip(":=").rstrip(",").strip() if toks_orig else ""

            # 2. Pure Inference (no gold labels)
            s_pure, e_pure, toks_pure, _ = model.predict_span(inp_tensor, candidate_mask=m_pure)
            val_pure = tokenizer.decode(toks_pure).strip().lstrip(":=").rstrip(",").strip() if toks_pure else ""

            # 3. Corrected Oracle-Start Mode (only intervene on start, candidate mask is pure)
            if is_start_conditioned:
                s_orc, e_orc, toks_orc, _ = model.predict_span(inp_tensor, candidate_mask=m_pure, oracle_start=gold_s)
            else:
                s_logits, e_logits = model(inp_tensor, candidate_mask=m_pure)
                e_logits = e_logits[0]
                s_orc = gold_s
                cand_sub = torch.full_like(e_logits, float("-inf"))
                max_e = min(s_orc + MAX_SPAN_LEN, seq_len)
                for e in range(s_orc, max_e):
                    if m_pure[0, e]:
                        cand_sub[e] = e_logits[e]
                e_orc = cand_sub.argmax().item()
                toks_orc = full_seq[s_orc:e_orc+1] if s_orc <= e_orc else []

            val_orc = tokenizer.decode(toks_orc).strip().lstrip(":=").rstrip(",").strip() if toks_orc else ""

        # Check prediction difference
        pred_differs = (s_orig != s_pure or e_orig != e_pure)
        if pred_differs:
            predictions_differ_count += 1

        # Check premature
        prem_orig = (s_orig == gold_s and e_orig < gold_e and gold_tokens[:len(toks_orig)] == toks_orig)
        prem_pure = (s_pure == gold_s and e_pure < gold_e and gold_tokens[:len(toks_pure)] == toks_pure)
        prem_orc = (s_orc == gold_s and e_orc < gold_e and gold_tokens[:len(toks_orc)] == toks_orc)

        # Update Original
        if s_orig == gold_s: orig_results["start_correct"] += 1
        if e_orig == gold_e: orig_results["end_correct"] += 1
        if s_orig == gold_s and e_orig == gold_e: orig_results["exact_span"] += 1
        if toks_orig == gold_tokens: orig_results["exact_tokens"] += 1
        if val_orig == gold_val: orig_results["exact_val"] += 1
        if prem_orig: orig_results["premature_count"] += 1
        if s_orig > e_orig: orig_results["invalid_count"] += 1

        # Update Pure
        if s_pure == gold_s: pure_results["start_correct"] += 1
        if e_pure == gold_e: pure_results["end_correct"] += 1
        if s_pure == gold_s and e_pure == gold_e: pure_results["exact_span"] += 1
        if toks_pure == gold_tokens: pure_results["exact_tokens"] += 1
        if val_pure == gold_val: pure_results["exact_val"] += 1
        if prem_pure: pure_results["premature_count"] += 1
        if s_pure > e_pure: pure_results["invalid_count"] += 1

        # Update Oracle
        if e_orc == gold_e: oracle_results["end_correct"] += 1
        if s_orc == gold_s and e_orc == gold_e: oracle_results["exact_span"] += 1
        if toks_orc == gold_tokens: oracle_results["exact_tokens"] += 1
        if val_orc == gold_val: oracle_results["exact_val"] += 1
        if prem_orc: oracle_results["premature_count"] += 1
        if s_orc > e_orc: oracle_results["invalid_count"] += 1

        per_example_rows.append({
            "split": split_name,
            "example_id": item["example_id"],
            "gold_start": gold_s,
            "gold_end": gold_e,
            "target_len": target_len,
            "gold_val": gold_val,
            "orig_pred_start": s_orig,
            "orig_pred_end": e_orig,
            "orig_val": val_orig,
            "pure_pred_start": s_pure,
            "pure_pred_end": e_pure,
            "pure_val": val_pure,
            "oracle_pred_start": s_orc,
            "oracle_pred_end": e_orc,
            "oracle_val": val_orc,
            "mask_strictly_identical": is_mask_identical,
            "gold_in_pure_mask": gold_in_pure,
            "prediction_changed": pred_differs,
            "orig_exact_span": (s_orig == gold_s and e_orig == gold_e),
            "pure_exact_span": (s_pure == gold_s and e_pure == gold_e),
            "oracle_exact_span": (s_orc == gold_s and e_orc == gold_e),
            "pure_is_premature": prem_pure
        })

    def calc_rates(d: Dict[str, int]) -> Dict[str, Any]:
        return {
            "total": total,
            "start_acc": d["start_correct"] / total,
            "start_count": d["start_correct"],
            "end_acc": d["end_correct"] / total,
            "end_count": d["end_correct"],
            "exact_span_em": d["exact_span"] / total,
            "exact_span_count": d["exact_span"],
            "token_match_acc": d["exact_tokens"] / total,
            "val_match_acc": d["exact_val"] / total,
            "premature_rate": d["premature_count"] / total,
            "premature_count": d["premature_count"],
            "invalid_rate": d["invalid_count"] / total,
            "invalid_count": d["invalid_count"],
        }

    return {
        "total": total,
        "mask_audit_stats": {
            "masks_strictly_identical": mask_strictly_identical_count,
            "masks_identical_pct": (mask_strictly_identical_count / total) * 100,
            "gold_excluded_by_pure_mask": gold_excluded_by_pure_mask_count,
            "predictions_changed_count": predictions_differ_count,
            "predictions_changed_pct": (predictions_differ_count / total) * 100
        },
        "original_mask_metrics": calc_rates(orig_results),
        "pure_mask_metrics": calc_rates(pure_results),
        "corrected_oracle_metrics": calc_rates(oracle_results),
        "per_example_rows": per_example_rows
    }

def main():
    print("=" * 70)
    print("EXP 10.3: EVALUATION MASK LEAKAGE AUDIT HARNESS")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer, _ = load_tokenizer_artifacts(TOKENIZER_DIR)

    # 1. Record Hashes
    integrity_data = {
        "canonical_checkpoint": {
            "path": "experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt",
            "sha256": compute_sha256(REPO_ROOT / "experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt")
        },
        "tokenizer_vocab": {
            "path": "data/experiments/vocab_4096/vocab.json",
            "sha256": compute_sha256(TOKENIZER_DIR / "vocab.json")
        },
        "tokenizer_merges": {
            "path": "data/experiments/vocab_4096/merges.json",
            "sha256": compute_sha256(TOKENIZER_DIR / "merges.json")
        },
        "session_runtime": {
            "path": "chakrview/runtime/session.py",
            "sha256": compute_sha256(REPO_ROOT / "chakrview/runtime/session.py")
        },
        "exp10_best_checkpoint": {
            "path": "experiments/research_v08_json_sft/exp10_span_pointer/checkpoints/exp10_span_pointer_best.pt",
            "sha256": compute_sha256(EXP10_CKPT)
        },
        "exp10_2_best_checkpoint": {
            "path": "experiments/research_v08_json_sft/exp10_span_pointer/exp10_2_end_boundary/checkpoints/exp10_2_start_cond_best.pt",
            "sha256": compute_sha256(EXP10_2_CKPT)
        }
    }

    with open(EXP10_3_DIR / "EXP10_3_INTEGRITY.json", "w", encoding="utf-8") as f:
        json.dump(integrity_data, f, indent=2)
    print("Recorded integrity hashes in EXP10_3_INTEGRITY.json")

    # 2. Load Models
    config = ModelConfig(max_seq_len=512)
    exp10_model = ChakrSpanPointerModel(config, max_span_len=MAX_SPAN_LEN, freeze_backbone=False).to(device)
    exp10_model.load_state_dict(torch.load(EXP10_CKPT, map_location=device, weights_only=False)["model_state_dict"])
    exp10_model.eval()

    exp10_2_model = ChakrStartConditionedSpanPointer(config, max_span_len=MAX_SPAN_LEN, freeze_backbone=False).to(device)
    exp10_2_model.load_state_dict(torch.load(EXP10_2_CKPT, map_location=device, weights_only=False)["model_state_dict"])
    exp10_2_model.eval()

    splits = {
        "val": load_jsonl(DATA_DIR / "val.jsonl"),
        "test_a": load_jsonl(DATA_DIR / "test_a_familiar.jsonl"),
        "test_b": load_jsonl(DATA_DIR / "test_b_held_out.jsonl"),
    }

    audit_summary = {
        "exp10_factorized": {},
        "exp10_2_start_conditioned": {}
    }

    all_csv_rows = []

    for model_name, model, is_sc in [
        ("exp10_factorized", exp10_model, False),
        ("exp10_2_start_conditioned", exp10_2_model, True)
    ]:
        print(f"\nEvaluating Model: {model_name}")
        audit_summary[model_name] = {}
        for split_name, dataset in splits.items():
            res = evaluate_split(model, dataset, tokenizer, split_name, device, is_start_conditioned=is_sc)
            audit_summary[model_name][split_name] = {
                "mask_audit_stats": res["mask_audit_stats"],
                "original_mask_metrics": res["original_mask_metrics"],
                "pure_mask_metrics": res["pure_mask_metrics"],
                "corrected_oracle_metrics": res["corrected_oracle_metrics"]
            }

            for row in res["per_example_rows"]:
                row["model"] = model_name
                all_csv_rows.append(row)

            m_stats = res["mask_audit_stats"]
            p_m = res["pure_mask_metrics"]
            o_m = res["original_mask_metrics"]
            print(f"  [{split_name.upper()}] Identical Masks: {m_stats['masks_strictly_identical']}/{res['total']} ({m_stats['masks_identical_pct']:.1f}%) | Gold Excluded: {m_stats['gold_excluded_by_pure_mask']} | Preds Changed: {m_stats['predictions_changed_count']}")
            print(f"    Orig Span EM: {o_m['exact_span_em']*100:.2f}% | Pure Span EM: {p_m['exact_span_em']*100:.2f}%")

    with open(EXP10_3_DIR / "EXP10_3_MASK_AUDIT_RESULTS.json", "w", encoding="utf-8") as f:
        json.dump(audit_summary, f, indent=2)
    print(f"\nSaved results to {EXP10_3_DIR / 'EXP10_3_MASK_AUDIT_RESULTS.json'}")

    # Write CSV
    csv_path = EXP10_3_DIR / "EXP10_3_PER_EXAMPLE.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "model", "split", "example_id",
            "gold_start", "gold_end", "target_len", "gold_val",
            "orig_pred_start", "orig_pred_end", "orig_val",
            "pure_pred_start", "pure_pred_end", "pure_val",
            "oracle_pred_start", "oracle_pred_end", "oracle_val",
            "mask_strictly_identical", "gold_in_pure_mask", "prediction_changed",
            "orig_exact_span", "pure_exact_span", "oracle_exact_span", "pure_is_premature"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_csv_rows:
            writer.writerow(r)
    print(f"Saved per-example details to {csv_path}")

if __name__ == "__main__":
    main()
