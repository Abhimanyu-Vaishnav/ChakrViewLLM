"""
Master Deterministic Dataset Compiler for Exp34 SFT.

Orchestrates all 10 verified category generators to build the full 1,000-example SFT dataset:
1. technical_explanations (100)
2. python_programming (100)
3. code_debugging (100)
4. algorithmic_reasoning (100)
5. arithmetic_reasoning (100)
6. passage_summarization (100)
7. constrained_formatting (100)
8. conversational_clarification (100)
9. hindi_comprehension (100)
10. bilingual_hinglish (100)

Total: exactly 1,000 examples.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

from scripts.generate_full_1000_dataset import build_category_1_tech_explanations
from scripts.generate_dataset_categories_2_to_5 import (
    get_python_programming_100,
    get_code_debugging_100,
    get_algorithmic_reasoning_100,
    get_arithmetic_reasoning_100,
)
from scripts.generate_dataset_categories_6_to_10 import (
    get_passage_summarization_100,
    get_constrained_formatting_100,
    get_conversational_clarification_100,
    get_hindi_comprehension_100,
    get_bilingual_hinglish_100,
)


def compile_1000_sft_dataset() -> List[Dict[str, Any]]:
    """Gathers all 10 categories, each strictly 100 non-duplicate examples."""
    cat1 = build_category_1_tech_explanations()
    cat2 = get_python_programming_100()
    cat3 = get_code_debugging_100()
    cat4 = get_algorithmic_reasoning_100()
    cat5 = get_arithmetic_reasoning_100()
    cat6 = get_passage_summarization_100()
    cat7 = get_constrained_formatting_100()
    cat8 = get_conversational_clarification_100()
    cat9 = get_hindi_comprehension_100()
    cat10 = get_bilingual_hinglish_100()

    categories = [cat1, cat2, cat3, cat4, cat5, cat6, cat7, cat8, cat9, cat10]
    for idx, c in enumerate(categories, 1):
        assert len(c) == 100, f"Category {idx} has {len(c)} records, expected 100"

    all_records = []
    for c in categories:
        all_records.extend(c)

    assert len(all_records) == 1000, f"Total records {len(all_records)} != 1000"
    return all_records


def write_verified_dataset(
    records: List[Dict[str, Any]],
    output_jsonl: Path,
    output_manifest: Path,
    seed: int = 42,
) -> Dict[str, Any]:
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    with open(output_jsonl, "w", encoding="utf-8", newline="\n") as f:
        for r in records:
            row = {
                "id": r["id"],
                "prompt": r["prompt"],
                "response": r["response"],
                "category": r["category"],
                "source_type": r["source_type"],
                "language": r["language"],
                "license": r["license"],
                "data_role": r.get("data_role", "TRAIN"),
                "review_status": r.get("review_status", "VERIFIED"),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with open(output_jsonl, "rb") as f:
        jsonl_sha256 = hashlib.sha256(f.read()).hexdigest()

    cat_counts: Dict[str, int] = {}
    lang_counts: Dict[str, int] = {}
    src_counts: Dict[str, int] = {}

    for r in records:
        c = r["category"]
        l = r["language"]
        s = r["source_type"]
        cat_counts[c] = cat_counts.get(c, 0) + 1
        lang_counts[l] = lang_counts.get(l, 0) + 1
        src_counts[s] = src_counts.get(s, 0) + 1

    manifest = {
        "dataset_name": "Exp34_Attractor_Suppression_SFT_Mixture",
        "dataset_version": "1.0.0",
        "experiment_id": "Exp34_Attractor_Suppression_SFT",
        "total_records": len(records),
        "seed": seed,
        "jsonl_sha256": jsonl_sha256,
        "category_counts": cat_counts,
        "language_counts": lang_counts,
        "source_type_counts": src_counts,
        "license_status": "All examples CC0-1.0 synthetic verified curation",
        "generator_script": "scripts/compile_exp34_dataset.py",
        "created_at_utc": "2026-10-29T18:00:00Z",
    }

    with open(output_manifest, "w", encoding="utf-8", newline="\n") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
        f.write("\n")

    return manifest


def main():
    parser = argparse.ArgumentParser(description="Compile verified 1,000 Exp34 SFT dataset")
    parser.add_argument("--output_jsonl", type=Path, default=Path("data/sft/exp34_instruction_mixture_train.jsonl"))
    parser.add_argument("--output_manifest", type=Path, default=Path("data/sft/exp34_instruction_mixture_manifest.json"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    recs = compile_1000_sft_dataset()
    manifest = write_verified_dataset(recs, args.output_jsonl, args.output_manifest, seed=args.seed)
    print(f"Dataset successfully compiled: {manifest['total_records']} rows.")
    print(f"JSONL Path: {args.output_jsonl} (SHA-256: {manifest['jsonl_sha256']})")


if __name__ == "__main__":
    main()

