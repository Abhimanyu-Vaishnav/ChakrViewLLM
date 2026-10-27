import csv
import json
import os
import sys

def reconcile_reviews(completed_csv_path, output_json_path):
    if not os.path.exists(completed_csv_path):
        print(f"[ERROR] Review file not found: {completed_csv_path}")
        print("Reconciliation REFUSED: File must be created and signed by project owner.")
        sys.exit(1)

    with open(completed_csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        records = list(reader)

    if len(records) != 20:
        print(f"[ERROR] Expected exactly 20 records in review queue, found {len(records)}.")
        print("Reconciliation REFUSED: Queue integrity violation.")
        sys.exit(1)

    valid_decisions = {"ACCEPT", "REJECT", "PARTIAL"}
    unresolved = []
    invalid = []

    for r in records:
        rid = r.get("record_id", "UNKNOWN")
        dec = r.get("reviewer_decision", "").strip().upper()
        if not dec or dec == "AWAITING_HUMAN_REVIEW":
            unresolved.append(rid)
        elif dec not in valid_decisions:
            invalid.append((rid, dec))

    if unresolved:
        print(f"[REFUSAL] Reconciliation cannot proceed: {len(unresolved)} records remain unresolved:")
        for rid in unresolved:
            print(f"  - {rid}")
        print("Please ensure the project owner assigns ACCEPT, REJECT, or PARTIAL to every record.")
        sys.exit(2)

    if invalid:
        print(f"[REFUSAL] Reconciliation cannot proceed: {len(invalid)} records have invalid decision labels:")
        for rid, val in invalid:
            print(f"  - {rid}: '{val}'")
        print("Valid decision labels are: ACCEPT, REJECT, PARTIAL.")
        sys.exit(3)

    # All 20 records resolved validly
    accept_count = sum(1 for r in records if r["reviewer_decision"].strip().upper() == "ACCEPT")
    reject_count = sum(1 for r in records if r["reviewer_decision"].strip().upper() == "REJECT")
    partial_count = sum(1 for r in records if r["reviewer_decision"].strip().upper() == "PARTIAL")

    reconciliation_data = {
        "status": "RECONCILIATION_COMPLETE",
        "total_records": 20,
        "accept_count": accept_count,
        "reject_count": reject_count,
        "partial_count": partial_count,
        "records": records
    }

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(reconciliation_data, f, indent=2)

    print(f"[SUCCESS] Reconciled all 20 records: {accept_count} ACCEPT, {reject_count} REJECT, {partial_count} PARTIAL.")
    print(f"Saved reconciled results to: {output_json_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python reconcile_human_reviews.py <path_to_completed_csv> [path_to_output_json]")
        # Test validation guard with default queue (which contains AWAITING_HUMAN_REVIEW)
        default_queue = "experiments/overnight_runs/exp30_human_review_closure_and_v08_gate_20261025_212820/EXP18_HUMAN_REVIEW_QUEUE.csv"
        print(f"\n[Test Guard Mode] Checking default unreviewed queue: {default_queue}")
        reconcile_reviews(default_queue, "test_reconciled.json")
    else:
        csv_in = sys.argv[1]
        json_out = sys.argv[2] if len(sys.argv) > 2 else "EXP18_RECONCILED_RESULTS.json"
        reconcile_reviews(csv_in, json_out)
