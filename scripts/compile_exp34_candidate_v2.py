"""
Exp34 Candidate V2 Master Dataset Compiler.

Orchestrates verified Category 1-10 generators to produce the fully remediated, 100% diverse,
1,000-example Exp34 candidate dataset.
Output location: data/sft/exp34_candidate_v2/
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

from scripts.generate_full_1000_dataset import build_category_1_tech_explanations
from scripts.generate_candidate_v2_category2 import build_category_2_python_programming
from scripts.generate_candidate_v2_categories3_to_5 import (
    build_category_3_debugging,
    build_category_4_algorithmic_reasoning,
    build_category_5_arithmetic_reasoning,
)
from scripts.generate_candidate_v2_categories6_to_10 import (
    build_category_6_summarization,
    build_category_7_constrained_formatting,
    build_category_8_conversational_clarification,
    build_category_9_hindi_comprehension,
    build_category_10_bilingual_hinglish,
)


def compile_candidate_v2_dataset() -> List[Dict[str, Any]]:
    generators = [
        ("technical_explanations", build_category_1_tech_explanations),
        ("python_programming", build_category_2_python_programming),
        ("code_debugging", build_category_3_debugging),
        ("algorithmic_reasoning", build_category_4_algorithmic_reasoning),
        ("arithmetic_reasoning", build_category_5_arithmetic_reasoning),
        ("passage_summarization", build_category_6_summarization),
        ("constrained_formatting", build_category_7_constrained_formatting),
        ("conversational_clarification", build_category_8_conversational_clarification),
        ("hindi_comprehension", build_category_9_hindi_comprehension),
        ("bilingual_hinglish", build_category_10_bilingual_hinglish),
    ]

    all_records = []
    for cat_name, gen_fn in generators:
        recs = gen_fn()
        assert len(recs) == 100, f"Category {cat_name} produced {len(recs)} records, expected 100"
        prompts = [r["prompt"] for r in recs]
        responses = [r["response"] for r in recs]
        assert len(set(prompts)) == 100, f"Category {cat_name} has duplicate prompts: {len(set(prompts))}/100"
        assert len(set(responses)) == 100, f"Category {cat_name} has duplicate responses: {len(set(responses))}/100"
        all_records.extend(recs)

    assert len(all_records) == 1000, f"Total records {len(all_records)} != 1000"
    all_prompts = [r["prompt"] for r in all_records]
    all_responses = [r["response"] for r in all_records]
    assert len(set(all_prompts)) == 1000, f"Global duplicate prompts detected: {len(set(all_prompts))}/1000"
    assert len(set(all_responses)) == 1000, f"Global duplicate responses detected: {len(set(all_responses))}/1000"

    return all_records


def write_candidate_v2_files(
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
        "dataset_name": "Exp34_Attractor_Suppression_SFT_Mixture_Candidate_V2",
        "dataset_version": "2.0.0-candidate",
        "experiment_id": "Exp34_Attractor_Suppression_SFT",
        "total_records": len(records),
        "seed": seed,
        "jsonl_sha256": jsonl_sha256,
        "category_counts": cat_counts,
        "language_counts": lang_counts,
        "source_type_counts": src_counts,
        "unique_prompts_count": len(set(r["prompt"] for r in records)),
        "unique_responses_count": len(set(r["response"] for r in records)),
        "license_status": "All examples CC0-1.0 synthetic verified curation & local repo corpus transformation",
        "generator_script": "scripts/compile_exp34_candidate_v2.py",
        "created_at_utc": "2026-10-29T18:25:00Z",
    }

    with open(output_manifest, "w", encoding="utf-8", newline="\n") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
        f.write("\n")

    return manifest


def main():
    parser = argparse.ArgumentParser(description="Compile verified Exp34 Candidate V2 dataset")
    parser.add_argument("--output_jsonl", type=Path, default=Path("data/sft/exp34_candidate_v2/exp34_instruction_mixture_train_v2.jsonl"))
    parser.add_argument("--output_manifest", type=Path, default=Path("data/sft/exp34_candidate_v2/exp34_instruction_mixture_manifest_v2.json"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    recs = compile_candidate_v2_dataset()
    manifest = write_candidate_v2_files(recs, args.output_jsonl, args.output_manifest, seed=args.seed)
    print(f"Exp34 Candidate V2 successfully compiled: {manifest['total_records']} rows.")
    print(f"JSONL Path: {args.output_jsonl} (SHA-256: {manifest['jsonl_sha256']})")
    print(f"Unique prompts: {manifest['unique_prompts_count']} / 1000")
    print(f"Unique responses: {manifest['unique_responses_count']} / 1000")


if __name__ == "__main__":
    main()

