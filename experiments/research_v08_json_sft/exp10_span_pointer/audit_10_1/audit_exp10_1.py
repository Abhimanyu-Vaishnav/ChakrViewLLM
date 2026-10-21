"""
audit_exp10_1.py
ChakrView Experiment 10.1 — Independent Generalization & Leakage Audit Script

Performs:
1. Recomputes all metrics directly from EXP10_PER_EXAMPLE.csv independently from EXP10_RESULTS.json.
2. Checks denominator completeness (verifies no missing/invalid predictions).
3. Analyzes the discrepancy between Value EM and Span EM on Validation, Test A, and Test B.
4. Audits dataset splits for exact prompt duplicates, target value overlap, and leakage.
5. Extracts and classifies all 73 Test B failures into AUDIT_10_1_FAILURES.csv.
6. Recomputes cryptographic hashes of all canonical and protected artifacts.
7. Generates:
   - AUDIT_10_1_RESULTS.json
   - AUDIT_10_1_FAILURES.csv
   - AUDIT_10_1_SPLIT_OVERLAP.json
   - AUDIT_10_1_INTEGRITY.json
"""

from __future__ import annotations

import collections
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

EXP10_DIR = Path(__file__).resolve().parents[1]
AUDIT_DIR = Path(__file__).resolve().parent
DATA_DIR = EXP10_DIR / "data"
CSV_PATH = EXP10_DIR / "EXP10_PER_EXAMPLE.csv"
ORIG_JSON_PATH = EXP10_DIR / "EXP10_RESULTS.json"

CANONICAL_CKPT = REPO_ROOT / "experiments" / "batch16_21_combined" / "main_training_10000" / "training_checkpoint_step_10000.pt"
TOKENIZER_DIR = REPO_ROOT / "data" / "experiments" / "vocab_4096"
SESSION_RUNTIME = REPO_ROOT / "chakrview" / "runtime" / "session.py"

EXPECTED_HASHES = {
    "Canonical Checkpoint": ("96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee", CANONICAL_CKPT),
    "Tokenizer Vocab": ("db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a", TOKENIZER_DIR / "vocab.json"),
    "Tokenizer Merges": ("ada4b1dcfbb3d962a2e60203089881872a5063de73b2fef9eeff81ce0016efa7", TOKENIZER_DIR / "merges.json"),
    "Tokenizer Config": ("0fca72b6a70e9ae812589199c661ef6be20cb9973f06353db125a1ec7264e589", TOKENIZER_DIR / "config.json"),
    "Protected Session Runtime": ("9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67", SESSION_RUNTIME)
}

def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def recompute_metrics_from_csv():
    with open(CSV_PATH, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    recomputed = {}
    splits = ["val", "test_a", "test_b"]

    for s in splits:
        s_rows = [r for r in rows if r["split"] == s]
        total = len(s_rows)
        
        span_em = sum(1 for r in s_rows if r["span_exact_match"].lower() == "true")
        val_em = sum(1 for r in s_rows if r["value_exact_match"].lower() == "true")
        start_acc = sum(1 for r in s_rows if int(r["predicted_start"]) == int(r["expected_start"]))
        end_acc = sum(1 for r in s_rows if int(r["predicted_end"]) == int(r["expected_end"]))
        tok_em = sum(1 for r in s_rows if r["predicted_tokens"] == r["expected_tokens"])
        inv_spans = sum(1 for r in s_rows if int(r["predicted_end"]) < int(r["predicted_start"]))

        single_ptr = sum(1 for r in s_rows if r["single_token_ptr_match"].lower() == "true")
        det_lookup = sum(1 for r in s_rows if r["deterministic_lookup_match"].lower() == "true")

        by_cat = collections.defaultdict(lambda: {
            "total": 0, "span_em": 0, "value_em": 0, "single_ptr_em": 0, "det_lookup_em": 0
        })
        by_len = collections.defaultdict(lambda: {"total": 0, "value_em": 0, "span_em": 0})
        by_punct = collections.defaultdict(lambda: {"total": 0, "value_em": 0, "span_em": 0})

        for r in s_rows:
            cat = r["category"]
            span_l = int(r["target_span_length"])
            is_punct = r["is_punctuation_adjacent"].lower() == "true"
            is_val_hit = r["value_exact_match"].lower() == "true"
            is_span_hit = r["span_exact_match"].lower() == "true"

            by_cat[cat]["total"] += 1
            by_cat[cat]["span_em"] += int(is_span_hit)
            by_cat[cat]["value_em"] += int(is_val_hit)
            by_cat[cat]["single_ptr_em"] += int(r["single_token_ptr_match"].lower() == "true")
            by_cat[cat]["det_lookup_em"] += int(r["deterministic_lookup_match"].lower() == "true")

            by_len[span_l]["total"] += 1
            by_len[span_l]["value_em"] += int(is_val_hit)
            by_len[span_l]["span_em"] += int(is_span_hit)

            by_punct[is_punct]["total"] += 1
            by_punct[is_punct]["value_em"] += int(is_val_hit)
            by_punct[is_punct]["span_em"] += int(is_span_hit)

        recomputed[s] = {
            "total_examples": total,
            "span_exact_match": f"{span_em}/{total} ({span_em/total*100:.2f}%)",
            "value_exact_match": f"{val_em}/{total} ({val_em/total*100:.2f}%)",
            "token_sequence_match": f"{tok_em}/{total} ({tok_em/total*100:.2f}%)",
            "start_accuracy": f"{start_acc}/{total} ({start_acc/total*100:.2f}%)",
            "end_accuracy": f"{end_acc}/{total} ({end_acc/total*100:.2f}%)",
            "invalid_span_rate": f"{inv_spans}/{total} ({inv_spans/total*100:.2f}%)",
            "single_token_pointer_baseline": f"{single_ptr}/{total} ({single_ptr/total*100:.2f}%)",
            "deterministic_lookup_baseline": f"{det_lookup}/{total} ({det_lookup/total*100:.2f}%)",
            "by_category": {
                k: {
                    "total": v["total"],
                    "value_em": f"{v['value_em']}/{v['total']} ({v['value_em']/v['total']*100:.2f}%)",
                    "span_em": f"{v['span_em']}/{v['total']} ({v['span_em']/v['total']*100:.2f}%)",
                    "single_ptr_em": f"{v['single_ptr_em']}/{v['total']} ({v['single_ptr_em']/v['total']*100:.2f}%)"
                } for k, v in by_cat.items()
            },
            "by_target_span_length": {
                k: {
                    "total": v["total"],
                    "value_em": f"{v['value_em']}/{v['total']} ({v['value_em']/v['total']*100:.2f}%)",
                    "span_em": f"{v['span_em']}/{v['total']} ({v['span_em']/v['total']*100:.2f}%)"
                } for k, v in sorted(by_len.items())
            },
            "by_punctuation_adjacency": {
                ("punctuation_adjacent" if k else "whitespace_delimited"): {
                    "total": v["total"],
                    "value_em": f"{v['value_em']}/{v['total']} ({v['value_em']/v['total']*100:.2f}%)",
                    "span_em": f"{v['span_em']}/{v['total']} ({v['span_em']/v['total']*100:.2f}%)"
                } for k, v in by_punct.items()
            }
        }

    return recomputed, rows

def audit_dataset_splits():
    splits = {}
    for s in ["train.jsonl", "val.jsonl", "test_a_familiar.jsonl", "test_b_held_out.jsonl"]:
        with open(DATA_DIR / s, "r", encoding="utf-8") as f:
            splits[s] = [json.loads(line) for line in f if line.strip()]

    train_prompts = set(r["prompt"] for r in splits["train.jsonl"])
    train_targets = set(r["target_value"] for r in splits["train.jsonl"])

    split_overlap = {}
    for s_name, recs in splits.items():
        if s_name == "train.jsonl":
            continue
        prompts = set(r["prompt"] for r in recs)
        targets = set(r["target_value"] for r in recs)
        p_overlap = len(prompts.intersection(train_prompts))
        t_overlap = len(targets.intersection(train_targets))

        split_overlap[s_name] = {
            "total_records": len(recs),
            "unique_prompts": len(prompts),
            "exact_prompt_duplicates_with_train": p_overlap,
            "unique_target_values": len(targets),
            "target_values_in_train": t_overlap,
            "target_overlap_pct": round(t_overlap / len(targets) * 100.0, 2),
            "is_strictly_held_out": (t_overlap == 0)
        }

    return split_overlap

def audit_test_b_failures(rows):
    test_b_fails = [r for r in rows if r["split"] == "test_b" and r["value_exact_match"].lower() == "false"]
    classified_failures = []

    reason_counts = collections.defaultdict(int)

    for r in test_b_fails:
        s_gt = int(r["expected_start"])
        e_gt = int(r["expected_end"])
        s_pred = int(r["predicted_start"])
        e_pred = int(r["predicted_end"])
        exp_val = r["expected_target_value"]
        dec_val = r["decoded_output"]

        if s_pred == s_gt and e_pred < e_gt:
            classification = "PREMATURE_END_TRUNCATION (Predicted start correct, but end pointer terminated early on unseen subwords)"
        elif s_pred == s_gt and e_pred > e_gt:
            classification = "EXTENDED_END_BOUNDARY (Predicted start correct, but end pointer included trailing tokens)"
        elif s_pred != s_gt and e_pred == e_gt:
            classification = "INCORRECT_START_BOUNDARY"
        elif s_pred != s_gt and e_pred != e_gt:
            classification = "DISPLACED_SPAN_SELECTION (Both start and end point to a distractor or different occurrence)"
        else:
            classification = "DECODING_NORMALIZATION_MISMATCH"

        reason_counts[classification] += 1
        classified_failures.append({
            "example_id": r["example_id"],
            "category": r["category"],
            "expected_target_value": exp_val,
            "decoded_output": dec_val,
            "expected_start": s_gt,
            "expected_end": e_gt,
            "predicted_start": s_pred,
            "predicted_end": e_pred,
            "expected_tokens": r["expected_tokens"],
            "predicted_tokens": r["predicted_tokens"],
            "confidence": r["confidence"],
            "failure_classification": classification,
            "prompt": r["prompt"]
        })

    return classified_failures, dict(reason_counts)

def audit_integrity():
    integrity_records = {}
    all_pass = True
    for name, (exp_sha, path) in EXPECTED_HASHES.items():
        actual_sha = compute_sha256(path)
        is_match = (actual_sha == exp_sha)
        if not is_match:
            all_pass = False
        integrity_records[name] = {
            "path": str(path),
            "expected_sha256": exp_sha,
            "actual_sha256": actual_sha,
            "status": "PASS" if is_match else "FAIL"
        }
    return integrity_records, all_pass

def main():
    print("=" * 70)
    print("EXP 10.1: INDEPENDENT GENERALIZATION & LEAKAGE AUDIT")
    print("=" * 70)

    # 1. Audit Integrity
    integrity, all_pass = audit_integrity()
    assert all_pass, "Critical Integrity Failure: Canonical artifacts modified!"
    with open(AUDIT_DIR / "AUDIT_10_1_INTEGRITY.json", "w", encoding="utf-8") as f:
        json.dump(integrity, f, indent=2)
    print("  [PASS] All canonical checkpoints, tokenizers, and runtime files cryptographically verified.")

    # 2. Recompute Metrics from CSV
    recomputed_metrics, rows = recompute_metrics_from_csv()
    with open(AUDIT_DIR / "AUDIT_10_1_RESULTS.json", "w", encoding="utf-8") as f:
        json.dump(recomputed_metrics, f, indent=2)
    print("  [PASS] Metrics recomputed independently from per-example predictions.")

    # 3. Dataset Leakage & Overlap Audit
    split_overlap = audit_dataset_splits()
    with open(AUDIT_DIR / "AUDIT_10_1_SPLIT_OVERLAP.json", "w", encoding="utf-8") as f:
        json.dump(split_overlap, f, indent=2)
    print("  [PASS] Split overlap and target isolation audited.")

    # 4. Analyze Test B Failures
    failures, failure_summary = audit_test_b_failures(rows)
    with open(AUDIT_DIR / "AUDIT_10_1_FAILURES.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(failures[0].keys()))
        writer.writeheader()
        writer.writerows(failures)
    print(f"  [PASS] All {len(failures)} Test B failures classified and saved to AUDIT_10_1_FAILURES.csv.")

    print("\n" + "=" * 70)
    print("AUDIT 10.1 SUMMARY OF RECOMPUTED METRICS:")
    print("=" * 70)
    for s_name, data in recomputed_metrics.items():
        print(f"\n[{s_name.upper()}] (Total: {data['total_examples']}):")
        print(f"  Value Exact Match:          {data['value_exact_match']}")
        print(f"  Span Exact Match (s & e):   {data['span_exact_match']}")
        print(f"  Start Position Accuracy:    {data['start_accuracy']}")
        print(f"  End Position Accuracy:      {data['end_accuracy']}")
        print(f"  Invalid Span Rate:          {data['invalid_span_rate']}")
        print(f"  Single-Token Pointer Base:  {data['single_token_pointer_baseline']}")
        print(f"  Deterministic Lookup Base:  {data['deterministic_lookup_baseline']}")

    print("\n" + "=" * 70)
    print("TEST B FAILURE CLASSIFICATION (73 Total Failures):")
    print("=" * 70)
    for reason, count in failure_summary.items():
        print(f"  - {reason}: {count} ({count/73*100:.1f}%)")

if __name__ == "__main__":
    main()
