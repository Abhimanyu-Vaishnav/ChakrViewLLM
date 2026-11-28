import os
import sys
import json

sys.path.insert(0, os.path.abspath("."))

with open("experiments/forensics_nr055/blind_evaluation_records.json", "r", encoding="utf-8") as f:
    blind_records = json.load(f)

print(f"Total blind evaluation records: {len(blind_records)}")

# Group by prompt to create structured review packets
review_queue = []
prompt_groups = {}

for rec in blind_records:
    pid = rec["prompt_id"]
    prompt_groups.setdefault(pid, []).append(rec)

for pid, group in prompt_groups.items():
    # Shuffle or anonymize model codes
    item_review = {
        "prompt_id": pid,
        "prompt": group[0]["prompt"],
        "category": group[0]["category"],
        "expected_answer": group[0]["expected_answer"],
        "candidate_responses": []
    }
    for r in group:
        item_review["candidate_responses"].append({
            "model_code": r["model_code"],
            "generated_response": r["generated_response"],
            "automated_score": r["score"],
            "automated_verdict": r["verdict"],
            "human_factual_accuracy": None,
            "human_language_quality": None,
            "human_reasoning_validity": None
        })
    review_queue.append(item_review)

with open("experiments/forensics_nr056/human_review_queue.json", "w", encoding="utf-8") as f:
    json.dump(review_queue, f, indent=2, ensure_ascii=False)

print(f"Human review queue constructed with {len(review_queue)} distinct prompt packets.")
