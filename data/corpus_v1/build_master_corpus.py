"""
data/corpus_v1/build_master_corpus.py
Constructs the ChakrView 1.0 Master Conversational Corpus (V1) across all 22 capability areas:
- Merges vetted canonical Stage-D and UCIP-14R datasets (provenanced, deduplicated).
- Filters exact duplicates and prevents test contamination against sealed benchmarks.
- Produces clean, standardized JSONL splits (train, validation).
- Computes SHA-256 manifests and comprehensive domain/language distribution statistics.
"""

import os
import sys
import json
import hashlib
from collections import Counter
from pathlib import Path

def file_sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def normalize_text(text):
    return " ".join(text.strip().split())

def build_v1_corpus():
    print("=== BUILDING CHAKRVIEW 1.0 MASTER CONVERSATIONAL CORPUS (V1) ===")

    # 1. Load sealed benchmark to ensure 0 test leakage
    sealed_bmk_path = Path("data/stage_j_020/fresh_matched_benchmark_1000.json")
    benchmark_prompts = set()
    if sealed_bmk_path.exists():
        with open(sealed_bmk_path, "r", encoding="utf-8") as f:
            bmk_data = json.load(f)
            for item in bmk_data:
                benchmark_prompts.add(normalize_text(item["prompt"]))
    print(f"Loaded {len(benchmark_prompts)} sealed benchmark prompts for contamination gating.")

    sources = [
        {"path": "data/sft/ucip14r/sft_train_12000.jsonl", "split": "train", "weight": 1.0},
        {"path": "data/sft/ucip14r/sft_validation_1500.jsonl", "split": "val", "weight": 1.0},
        {"path": "data/stage_d/splits/stage_d_train.jsonl", "split": "train", "weight": 1.0},
        {"path": "data/stage_d/splits/stage_d_val.jsonl", "split": "val", "weight": 1.0},
        {"path": "data/sft/exp34_instruction_mixture_train.jsonl", "split": "train", "weight": 1.0}
    ]

    seen_prompts = set()
    train_records = []
    val_records = []
    skipped_duplicates = 0
    skipped_contamination = 0

    lang_counter = Counter()
    cat_counter = Counter()

    for src in sources:
        p = Path(src["path"])
        if not p.exists():
            print(f"Warning: source {p} not found, skipping.")
            continue

        print(f"Processing source: {p}...")
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                d = json.loads(line)
                
                # Extract prompt and response
                prompt = d.get("prompt", "")
                response = d.get("response", "")
                if not prompt or not response:
                    continue

                norm_p = normalize_text(prompt)
                
                # Contamination check
                if norm_p in benchmark_prompts:
                    skipped_contamination += 1
                    continue

                # Deduplication check
                if norm_p in seen_prompts:
                    skipped_duplicates += 1
                    continue
                seen_prompts.add(norm_p)

                lang = d.get("language", "en")
                if lang in ["en", "ENG"]: lang = "en"
                elif lang in ["hi", "HIN", "hindi"]: lang = "hi"
                elif lang in ["hi-en", "hinglish", "HING"]: lang = "hinglish"
                elif lang in ["sa", "SAN", "sanskrit"]: lang = "sa"

                cat = d.get("category", "GENERAL")
                record = {
                    "id": d.get("id", f"CBR10-{len(seen_prompts):06d}"),
                    "prompt": prompt,
                    "response": response,
                    "language": lang,
                    "category": cat,
                    "source": str(p)
                }

                lang_counter[lang] += 1
                cat_counter[cat] += 1

                if src["split"] == "val":
                    val_records.append(record)
                else:
                    train_records.append(record)

    out_dir = Path("data/corpus_v1")
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "master_corpus_train.jsonl"
    val_path = out_dir / "master_corpus_val.jsonl"

    with open(train_path, "w", encoding="utf-8") as f:
        for r in train_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(val_path, "w", encoding="utf-8") as f:
        for r in val_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    manifest = {
        "corpus_name": "ChakrView 1.0 Master Conversational Corpus V1",
        "total_unique_records": len(train_records) + len(val_records),
        "train_records": len(train_records),
        "val_records": len(val_records),
        "skipped_duplicates": skipped_duplicates,
        "skipped_contamination": skipped_contamination,
        "languages": dict(lang_counter),
        "categories": dict(cat_counter.most_common(25)),
        "files": {
            "train": {
                "path": str(train_path),
                "sha256": file_sha256(train_path),
                "size_bytes": os.path.getsize(train_path)
            },
            "val": {
                "path": str(val_path),
                "sha256": file_sha256(val_path),
                "size_bytes": os.path.getsize(val_path)
            }
        }
    }

    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"\nCorpus V1 Build Complete!")
    print(f"Total Unique Samples: {manifest['total_unique_records']} (Train: {len(train_records)}, Val: {len(val_records)})")
    print(f"Skipped Duplicates: {skipped_duplicates} | Contamination Filtered: {skipped_contamination}")
    print(f"Languages: {dict(lang_counter)}")
    print(f"Manifest saved to: {manifest_path}")

if __name__ == "__main__":
    build_v1_corpus()
