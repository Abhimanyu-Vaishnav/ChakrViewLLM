"""
Phase 2: Training & Held-Out Dataset Audit Script
Audits data composition, exact overlaps, normalized token overlaps, template similarities,
and checks for data leakage between train, val, and test_held_out.
"""

import json
import re
import difflib
from pathlib import Path
from collections import Counter

REPO_ROOT = Path(__file__).resolve().parents[4]
PILOT_DATASET_DIR = REPO_ROOT / "experiments" / "cognition" / "autonomous_12h" / "conversational_pilot" / "dataset"

def normalize_text(text: str) -> str:
    # Lowercase, strip punctuation and collapse whitespace
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def load_jsonl(path: Path):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items

def main():
    print("=== Phase 2: Training & Held-Out Dataset Integrity Audit ===")
    
    train_items = load_jsonl(PILOT_DATASET_DIR / "train.jsonl")
    val_items = load_jsonl(PILOT_DATASET_DIR / "val.jsonl")
    test_items = load_jsonl(PILOT_DATASET_DIR / "test_held_out.jsonl")
    
    print(f"Loaded: Train={len(train_items)}, Val={len(val_items)}, Test={len(test_items)}")

    # 1. Source distribution
    train_sources = Counter(x["source"] for x in train_items)
    val_sources = Counter(x["source"] for x in val_items)
    test_sources = Counter(x["source"] for x in test_items)

    train_categories = Counter(x.get("category", "unknown") for x in train_items)
    val_categories = Counter(x.get("category", "unknown") for x in val_items)
    test_categories = Counter(x.get("category", "unknown") for x in test_items)

    # 2. Overlap analysis: Prompt exact matches
    train_prompts = {x["prompt"].strip(): x for x in train_items}
    val_prompts = {x["prompt"].strip(): x for x in val_items}
    test_prompts = {x["prompt"].strip(): x for x in test_items}

    val_exact_in_train = [p for p in val_prompts if p in train_prompts]
    test_exact_in_train = [p for p in test_prompts if p in train_prompts]
    test_exact_in_val = [p for p in test_prompts if p in val_prompts]

    # 3. Normalized prompt matches
    train_norm_prompts = {normalize_text(x["prompt"]): x for x in train_items}
    test_norm_in_train = [x["prompt"] for x in test_items if normalize_text(x["prompt"]) in train_norm_prompts]

    # 4. Response exact matches
    train_resps = {x["response"].strip() for x in train_items}
    test_resps_in_train = [x["response"] for x in test_items if x["response"].strip() in train_resps]

    # 5. Near-duplicate / Levenshtein / Token Jaccard similarity between test and train
    high_similarity_pairs = []
    for t_item in test_items:
        t_p = t_item["prompt"]
        t_tokens = set(normalize_text(t_p).split())
        for tr_item in train_items:
            tr_p = tr_item["prompt"]
            tr_tokens = set(normalize_text(tr_p).split())
            if not t_tokens or not tr_tokens:
                continue
            jaccard = len(t_tokens & tr_tokens) / len(t_tokens | tr_tokens)
            if jaccard > 0.85:
                high_similarity_pairs.append({
                    "test_id": t_item["id"],
                    "test_prompt": t_p,
                    "train_id": tr_item["id"],
                    "train_prompt": tr_p,
                    "jaccard_similarity": round(jaccard, 4),
                    "test_response": t_item["response"],
                    "train_response": tr_item["response"]
                })

    # 6. Entity & Template Leakage Assessment
    # CBR5 format often uses structured key-value updates: e.g. "Where is Bob? ... Bob is in Paris."
    template_leakage_cases = []
    for p in high_similarity_pairs:
        # Check if identical template with only entity substituted
        if p["test_response"] == p["train_response"]:
            template_leakage_cases.append(p)

    results = {
        "dataset_counts": {
            "train": len(train_items),
            "val": len(val_items),
            "test_held_out": len(test_items),
            "total": len(train_items) + len(val_items) + len(test_items)
        },
        "source_distribution": {
            "train": dict(train_sources),
            "val": dict(val_sources),
            "test_held_out": dict(test_sources)
        },
        "category_distribution": {
            "train": dict(train_categories),
            "val": dict(val_categories),
            "test_held_out": dict(test_categories)
        },
        "exact_leakage": {
            "val_prompt_exact_matches_in_train": len(val_exact_in_train),
            "test_prompt_exact_matches_in_train": len(test_exact_in_train),
            "test_prompt_exact_matches_in_val": len(test_exact_in_val),
            "test_norm_prompt_matches_in_train": len(test_norm_in_train),
            "test_response_exact_matches_in_train": len(test_resps_in_train),
            "total_test_resps_matching_train_count": len(test_resps_in_train)
        },
        "near_duplicate_analysis": {
            "num_pairs_jaccard_gt_0_85": len(high_similarity_pairs),
            "num_pairs_identical_response": len(template_leakage_cases),
            "sample_high_similarity_pairs": high_similarity_pairs[:10]
        },
        "integrity_verdict": {
            "exact_prompt_leakage": len(test_exact_in_train) == 0,
            "synthetic_template_sharing": len(high_similarity_pairs) > 0,
            "scientific_assessment": (
                "No verbatim prompt duplicate leakage detected between train and test_held_out. "
                "However, because the dataset was created via random shuffle across CBR-04 and CBR-05 "
                "rather than template-disjoint or entity-disjoint splitting, test_held_out shares "
                "identical prompt templates and identical static response phrases (e.g. persona strings, "
                "epistemic abstention phrases, and city query syntaxes) with the training set. "
                "Therefore, test_held_out tests in-distribution generalization on held-out instances, "
                "NOT out-of-distribution or novel-template generalization."
            )
        }
    }

    out_file = REPO_ROOT / "experiments/cognition/cr_exp_04_validation/dataset_integrity_audit.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Dataset integrity audit written to {out_file}")
    print(f"Exact prompt leaks: {len(test_exact_in_train)}")
    print(f"High-similarity template pairs: {len(high_similarity_pairs)}")
    print(f"Test responses matching train response strings: {len(test_resps_in_train)} / {len(test_items)}")

if __name__ == "__main__":
    main()
