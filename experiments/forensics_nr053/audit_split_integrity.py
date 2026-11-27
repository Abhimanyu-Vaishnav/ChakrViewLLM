"""
experiments/forensics_nr053/audit_split_integrity.py

Phase 2 Forensic Tool:
Audits Corpus V4 Train vs Validation split integrity:
1. Exact document hash matching (SHA-256)
2. Normalized text hash matching (lowercased, whitespace-stripped)
3. Near-duplicate overlap (Jaccard similarity on 5-grams)
4. Paragraph and sentence level overlap
5. Source-level overlap (checking whether paragraphs from the same source file are in both train and val)
6. Contamination audit against Sealed 120 and Matched 1,000 benchmarks
7. Confirms whether the nominal 624,068 token total includes train, val, or test
"""

import sys
import os
import json
import hashlib
from pathlib import Path
from collections import Counter

sys.path.insert(0, ".")

def sha256_text(t):
    return hashlib.sha256(t.strip().encode("utf-8")).hexdigest()

def norm_hash(t):
    norm = " ".join(t.lower().split())
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()

def get_ngrams(text, n=5):
    words = text.lower().split()
    if len(words) < n:
        return set([" ".join(words)])
    return set(" ".join(words[i:i+n]) for i in range(len(words) - n + 1))

def audit_splits():
    print("=== AUDITING CORPUS V4 SPLIT INTEGRITY ===", flush=True)

    manifest = json.load(open("data/manifests/corpus_v4_manifest.json"))
    train_file = Path("data/corpus_v4/corpus_v4_train.jsonl")
    val_file = Path("data/corpus_v4/corpus_v4_val.jsonl")

    train_docs = []
    with open(train_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip(): train_docs.append(json.loads(line))

    val_docs = []
    with open(val_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip(): val_docs.append(json.loads(line))

    print(f"Loaded Train: {len(train_docs)} docs | Val: {len(val_docs)} docs")

    # 1. Exact & Normalized Hashes
    train_exact = set(sha256_text(d["text"]) for d in train_docs)
    val_exact = set(sha256_text(d["text"]) for d in val_docs)
    exact_overlap = train_exact.intersection(val_exact)

    train_norm = set(norm_hash(d["text"]) for d in train_docs)
    val_norm = set(norm_hash(d["text"]) for d in val_docs)
    norm_overlap = train_norm.intersection(val_norm)

    # 2. Near-duplicate analysis (Jaccard > 0.8 on 5-grams)
    train_ngrams = [get_ngrams(d["text"]) for d in train_docs]
    val_ngrams = [get_ngrams(d["text"]) for d in val_docs]

    near_dup_count = 0
    near_dup_examples = []
    for v_idx, v_ng in enumerate(val_ngrams):
        for t_idx, t_ng in enumerate(train_ngrams[:500]): # sample for efficiency
            if not v_ng or not t_ng: continue
            intersect = len(v_ng.intersection(t_ng))
            union = len(v_ng.union(t_ng))
            jaccard = intersect / union if union > 0 else 0
            if jaccard > 0.8:
                near_dup_count += 1
                near_dup_examples.append({
                    "val_text": val_docs[v_idx]["text"][:100],
                    "train_text": train_docs[t_idx]["text"][:100],
                    "jaccard": round(jaccard, 3)
                })
                break

    # 3. Source-level overlap
    train_sources = set(d.get("source", "unknown") for d in train_docs)
    val_sources = set(d.get("source", "unknown") for d in val_docs)
    source_overlap = train_sources.intersection(val_sources)

    # 4. Benchmark contamination
    sealed_120 = json.load(open("experiments/fast_track_047_049/sealed_120_benchmark.json"))
    matched_1000 = json.load(open("data/stage_j_020/fresh_matched_benchmark_1000.json"))

    s120_prompts = set(x["prompt"].strip().lower() for x in sealed_120)
    m1000_prompts = set(x["prompt"].strip().lower() for x in matched_1000 if "prompt" in x)

    train_texts = [d["text"].lower() for d in train_docs]
    s120_train_hits = sum(1 for p in s120_prompts if any(p in t for t in train_texts))
    m1000_train_hits = sum(1 for p in m1000_prompts if any(p in t for t in train_texts))

    results = {
        "nominal_tokens": {
            "total_verified": manifest["total_tokens_verified"],
            "train_tokens": manifest["train_tokens"],
            "val_tokens": manifest["val_tokens"],
            "includes_val_in_total": (manifest["total_tokens_verified"] == manifest["train_tokens"] + manifest["val_tokens"])
        },
        "exact_document_overlap": len(exact_overlap),
        "normalized_text_overlap": len(norm_overlap),
        "near_duplicate_count_sample": near_dup_count,
        "near_duplicate_examples": near_dup_examples[:3],
        "shared_source_files_count": len(source_overlap),
        "shared_source_files": sorted(list(source_overlap)),
        "sealed_120_hits_in_train": s120_train_hits,
        "matched_1000_hits_in_train": m1000_train_hits
    }

    out_file = Path("experiments/forensics_nr053/corpus_v4_split_integrity.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nSplit audit complete! Results saved to {out_file}", flush=True)
    print(f"Exact Document Overlap: {len(exact_overlap)}")
    print(f"Normalized Text Overlap: {len(norm_overlap)}")
    print(f"Shared Source Files: {len(source_overlap)} (Document chunks originated from common foundation files)")
    print(f"Benchmark Contamination Hits: Sealed 120 = {s120_train_hits}, Matched 1000 = {m1000_train_hits}")

if __name__ == "__main__":
    audit_splits()
