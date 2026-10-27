import csv
import json
import os

queue_path = "experiments/overnight_runs/exp30_human_review_closure_and_v08_gate_20261025_212820/EXP18_HUMAN_REVIEW_QUEUE.csv"
rc_dir = "experiments/overnight_runs/v08_release_candidate_readiness_20261025_213337"

def build_dossier():
    with open(queue_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"Total records in queue: {len(rows)}")

    lines = [
        "# Exp18 Short Explanation Human Review Dossier (v0.8 Release Gate)",
        "",
        "**Notice**: This dossier contains all 20 quarantined explanation records awaiting human review by the project owner. In accordance with ChakrView fail-closed data governance, **AI agents are prohibited from fabricating reviewer judgments**.",
        "",
        "## Scoring Rubric",
        "- **`ACCEPT`**: Output provides a coherent, factually correct explanation answering the prompt.",
        "- **`REJECT`**: Output exhibits repetitive token loops, factually false claims, or severe incoherence.",
        "- **`PARTIAL`**: Output contains correct explanatory components but suffers from truncated formatting or minor repetition.",
        "",
        "---",
        ""
    ]

    for i, r in enumerate(rows, 1):
        rid = r["record_id"]
        prompt = r["prompt"]
        exp = r["expected_answer"]
        act = r["actual_output"]
        stop = r["stop_reason"]
        rule = r["scoring_rule"]
        dec = r["reviewer_decision"]

        lines.append(f"### Record {i:02d}: `{rid}`")
        lines.append(f"- **Prompt**: {prompt}")
        lines.append(f"- **Expected Explanation**: *{exp}*")
        lines.append(f"- **Actual Model Output**: `{act}`")
        lines.append(f"- **Termination Reason**: `{stop}`")
        lines.append(f"- **Scoring Rule**: {rule}")
        lines.append(f"- **Current Status**: **`{dec}`**")
        lines.append("")

    out_doc = os.path.join(rc_dir, "EXP18_HUMAN_REVIEW_DOSSIER.md")
    with open(out_doc, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Successfully generated {out_doc}")

if __name__ == "__main__":
    build_dossier()
