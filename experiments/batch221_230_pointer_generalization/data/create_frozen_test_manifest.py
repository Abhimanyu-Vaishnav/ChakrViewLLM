"""
Script to create the frozen 100-record test manifest for Batches 221-230.
Location: experiments/batch221_230_pointer_generalization/data/create_frozen_test_manifest.py
"""
import hashlib
import json
from pathlib import Path
import random


def main():
    random.seed(42)
    test_path = Path("experiments/batch211_220_relational_learning/data/relational_corpus_test.json")
    all_test = json.loads(test_path.read_text(encoding="utf-8"))

    sample_100 = random.sample(all_test, 100)

    frozen_records = []
    for idx, r in enumerate(sample_100):
        gold = r["target_value"].strip()
        norm_gold = gold.lower().rstrip(".,;:?!")
        rid = r.get("record_id", str(idx))
        record = {
            "example_id": f"ex_{idx:03d}_{rid}",
            "record_id": rid,
            "domain": r.get("domain", "general"),
            "language": r.get("language", "en"),
            "entity": r["entity"],
            "relation": r["relation"],
            "context": r["context"],
            "query": r["query"],
            "gold_value": gold,
            "normalized_gold": norm_gold,
            "distractor_entity": r.get("distractor_entity", ""),
            "distractor_value": r.get("distractor_value", ""),
            "is_held_out": r.get("is_held_out", True),
            "diagnostic_tags": [
                r.get("evaluation_slice", "held_out"),
                "multilingual" if r.get("language") == "hi" or " (" in r["entity"] else "monolingual_en",
                "has_digits" if any(c.isdigit() for c in gold) else "no_digits",
                "multi_token" if len(gold.split()) > 1 else "single_token"
            ]
        }
        frozen_records.append(record)

    out_path = Path("experiments/batch221_230_pointer_generalization/data/frozen_test_manifest_100.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(frozen_records, indent=2, ensure_ascii=False), encoding="utf-8")
    h = hashlib.sha256(out_path.read_bytes()).hexdigest()
    print(f"Saved frozen test manifest: {len(frozen_records)} records to {out_path}")
    print(f"SHA-256: {h}")


if __name__ == "__main__":
    main()
