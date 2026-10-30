"""
Exp34 Candidate V3 Master Dataset Compiler.

Orchestrates 10 independently verified Category 1-10 generators to produce the
fully remediated, truly diverse, 1,000-example Exp34 Candidate V3 dataset.
Output location: data/sft/exp34_candidate_v3/
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

from scripts.generate_full_1000_dataset import build_category_1_tech_explanations
from scripts.generate_candidate_v2_category2 import build_category_2_python_programming
from scripts.generate_candidate_v3_category3_dbug import build_category_3_debugging
from scripts.generate_candidate_v3_category4_algo import build_category_4_algorithmic_reasoning
from scripts.generate_candidate_v2_categories3_to_5 import build_category_5_arithmetic_reasoning
from scripts.generate_candidate_v3_category6_summ import build_category_6_summarization
from scripts.generate_candidate_v3_category7_fmt import build_category_7_constrained_formatting
from scripts.generate_candidate_v3_category8_conv import build_category_8_conversational_clarification
from scripts.generate_candidate_v3_category9_hind import build_category_9_hindi_comprehension
from scripts.generate_candidate_v3_category10_bili import build_category_10_bilingual_hinglish


def compile_candidate_v3_dataset() -> List[Dict[str, Any]]:
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

        # Verify uniqueness within category
        prompts = [r["prompt"] for r in recs]
        responses = [r["response"] for r in recs]
        assert len(set(prompts)) == 100, f"Category {cat_name} has duplicate prompts!"
        assert len(set(responses)) == 100, f"Category {cat_name} has duplicate responses!"

        all_records.extend(recs)

    assert len(all_records) == 1000, f"Total records = {len(all_records)}, expected 1000"

    # Global uniqueness assertions
    all_prompts = [r["prompt"] for r in all_records]
    all_responses = [r["response"] for r in all_records]
    assert len(set(all_prompts)) == 1000, "Dataset has duplicate prompts globally!"
    assert len(set(all_responses)) == 1000, "Dataset has duplicate responses globally!"

    return all_records


def write_candidate_v3_files(output_dir: Path | str = "data/sft/exp34_candidate_v3") -> Dict[str, Any]:
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    records = compile_candidate_v3_dataset()

    jsonl_file = out_path / "exp34_instruction_mixture_train_v3.jsonl"
    with open(jsonl_file, "w", encoding="utf-8", newline="\n") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Compute exact SHA-256
    with open(jsonl_file, "rb") as f:
        jsonl_sha256 = hashlib.sha256(f.read()).hexdigest()

    manifest_file = out_path / "exp34_instruction_mixture_manifest_v3.json"
    manifest_data = {
        "dataset_name": "Exp34_Attractor_Suppression_SFT_Mixture_Candidate_V3",
        "dataset_version": "3.0.0-candidate",
        "experiment_id": "Exp34_Attractor_Suppression_SFT",
        "total_records": len(records),
        "seed": 42,
        "jsonl_sha256": jsonl_sha256,
        "category_counts": {
            "technical_explanations": 100,
            "python_programming": 100,
            "code_debugging": 100,
            "algorithmic_reasoning": 100,
            "arithmetic_reasoning": 100,
            "passage_summarization": 100,
            "constrained_formatting": 100,
            "conversational_clarification": 100,
            "hindi_comprehension": 100,
            "bilingual_hinglish": 100,
        },
        "language_counts": {
            "en": 800,
            "hi": 100,
            "hi-en": 100,
        },
        "source_type_counts": {
            "synthetic_factual_curation": 100,
            "synthetic_code_curation": 200,
            "synthetic_algorithmic_curation": 100,
            "synthetic_arithmetic_verified": 100,
            "local_corpus_transformation": 300,
            "synthetic_constrained_formatting": 100,
            "synthetic_conversational_curation": 100,
        },
        "unique_prompts_count": 1000,
        "unique_responses_count": 1000,
        "license_status": "All examples CC0-1.0 synthetic verified curation & local repo corpus transformation",
        "generator_script": "scripts/compile_exp34_candidate_v3.py",
        "created_at_utc": "2026-10-29T19:30:00Z",
    }

    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2, ensure_ascii=False)

    print(f"Candidate V3 compiled successfully:")
    print(f"  JSONL:    {jsonl_file} (SHA256: {jsonl_sha256})")
    print(f"  Manifest: {manifest_file}")
    return {"jsonl": str(jsonl_file), "sha256": jsonl_sha256, "manifest": str(manifest_file)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compile Exp34 Candidate V3 Dataset")
    parser.add_argument("--output_dir", default="data/sft/exp34_candidate_v3")
    args = parser.parse_args()
    write_candidate_v3_files(args.output_dir)

