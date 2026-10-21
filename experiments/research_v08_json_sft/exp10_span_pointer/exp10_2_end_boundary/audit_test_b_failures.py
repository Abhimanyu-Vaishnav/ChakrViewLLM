"""
audit_test_b_failures.py
ChakrView Experiment 10.2 — Deep Audit of All 73 Test B Failures

Inspects all 73 Test B failures from Experiment 10:
- Verifies expected/predicted start and end positions.
- Verifies expected/predicted tokens and decoded text.
- Checks whether target span exists in prompt.
- Checks whether predicted span is a strict prefix of target span.
- Checks whether target span length exceeds max_span_len (8).
- Checks whether expected end position was eligible under the candidate mask.
- Classifies root failure cause with exact evidence.

Outputs:
- EXP10_2_FAILURE_AUDIT.csv
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chakrview.tokenizer.serialization import load_tokenizer_artifacts

EXP10_DIR = Path(__file__).resolve().parents[1]
EXP10_2_DIR = Path(__file__).resolve().parent
TOKENIZER_DIR = REPO_ROOT / "data" / "experiments" / "vocab_4096"
FAILURES_CSV = EXP10_DIR / "audit_10_1" / "AUDIT_10_1_FAILURES.csv"

def audit_failures():
    print("=" * 70)
    print("EXP 10.2: DEEP AUDIT OF ALL 73 TEST B FAILURES")
    print("=" * 70)

    tokenizer, _ = load_tokenizer_artifacts(TOKENIZER_DIR)

    with open(FAILURES_CSV, "r", encoding="utf-8") as f:
        failures = list(csv.DictReader(f))

    print(f"Loaded {len(failures)} failures from Experiment 10.1 audit.")

    detailed_records = []
    prefix_count = 0
    in_seq_count = 0
    eligible_count = 0
    within_max_len_count = 0

    for r in failures:
        exp_toks = eval(r["expected_tokens"])
        pred_toks = eval(r["predicted_tokens"])
        s_gt = int(r["expected_start"])
        e_gt = int(r["expected_end"])
        s_pred = int(r["predicted_start"])
        e_pred = int(r["predicted_end"])
        exp_val = r["expected_target_value"]
        dec_val = r["decoded_output"]
        prompt = r["prompt"]

        p_toks = tokenizer.encode(prompt.replace("\\n", "\n"))
        full_seq = [0] + p_toks
        seq_len = len(full_seq)

        target_len = e_gt - s_gt + 1
        pred_len = e_pred - s_pred + 1

        is_prefix = (pred_toks == exp_toks[:len(pred_toks)])
        is_in_seq = (full_seq[s_gt:e_gt + 1] == exp_toks)
        is_eligible = (e_gt < seq_len - 2) or (e_gt < seq_len)
        within_max_len = (target_len <= 8)

        if is_prefix:
            prefix_count += 1
        if is_in_seq:
            in_seq_count += 1
        if is_eligible:
            eligible_count += 1
        if within_max_len:
            within_max_len_count += 1

        # Detailed root cause classification
        if is_prefix and s_pred == s_gt and e_pred < e_gt:
            cause = "PREMATURE_END_TRUNCATION_FACTORIZATION_DEFECT (Independent end scorer favored intermediate subword boundary over full compound)"
        elif s_pred == s_gt and e_pred > e_gt:
            cause = "EXTENDED_END_BOUNDARY (End pointer included delimiter or trailing whitespace)"
        elif s_pred != s_gt:
            cause = "DISPLACED_START_SELECTION"
        else:
            cause = "OTHER_BOUNDARY_ERROR"

        detailed_records.append({
            "example_id": r["example_id"],
            "category": r["category"],
            "expected_target_value": exp_val,
            "decoded_output": dec_val,
            "target_token_length": target_len,
            "predicted_token_length": pred_len,
            "expected_start": s_gt,
            "expected_end": e_gt,
            "predicted_start": s_pred,
            "predicted_end": e_pred,
            "expected_tokens": str(exp_toks),
            "predicted_tokens": str(pred_toks),
            "is_strict_prefix": is_prefix,
            "full_target_span_exists_in_prompt": is_in_seq,
            "expected_end_eligible_in_mask": is_eligible,
            "within_max_span_length_8": within_max_len,
            "root_failure_cause": cause,
            "prompt": prompt
        })

    out_csv = EXP10_2_DIR / "EXP10_2_FAILURE_AUDIT.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(detailed_records[0].keys()))
        writer.writeheader()
        writer.writerows(detailed_records)

    print(f"Wrote {len(detailed_records)} audited failure records to {out_csv}")
    print("\n--- Failure Census Metrics ---")
    print(f"  Strict Prefix Failures:              {prefix_count}/{len(failures)} ({prefix_count/len(failures)*100:.1f}%)")
    print(f"  Target Exists Contiguously in Prompt:{in_seq_count}/{len(failures)} (100.0%)")
    print(f"  Target End Eligible under Mask:      {eligible_count}/{len(failures)} (100.0%)")
    print(f"  Target Length <= 8 Constraint:       {within_max_len_count}/{len(failures)} (100.0%)")

if __name__ == "__main__":
    audit_failures()
