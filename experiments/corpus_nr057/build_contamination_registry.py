"""
Full Benchmark Contamination Auditor for NR-057.
Loads all sealed benchmark evaluations:
- data/evaluation/benchmark_v3/benchmark_v3_300.json
- data/evaluation/benchmark_v2/benchmark_v2_100.json
- experiments/forensics_nr053/fresh_heldout_benchmark_50.json
- evaluation/benchmark_prompts_v1.0.json
- evaluation/benchmark_prompts_v2.0.json

Produces contamination_exclusion_registry.json.
"""

import os
import sys
import json
import hashlib
from typing import List, Dict, Set, Tuple

sys.path.insert(0, os.path.abspath("."))
from chakrview.tokenizer.serialization import load_tokenizer_artifacts
from experiments.corpus_nr057.pipeline_core import normalize_text, extract_ngrams

def build_benchmark_registry():
    print("[BenchmarkAuditor] Loading tokenizer...")
    tok, _ = load_tokenizer_artifacts("data/experiments/vocab_4096")
    
    benchmark_paths = [
        "data/evaluation/benchmark_v3/benchmark_v3_300.json",
        "data/evaluation/benchmark_v2/benchmark_v2_100.json",
        "experiments/forensics_nr053/fresh_heldout_benchmark_50.json",
        "evaluation/benchmark_prompts_v1.0.json",
        "evaluation/benchmark_prompts_v2.0.json"
    ]
    
    registry = {
        "mission": "NR-057",
        "total_benchmark_sources": 0,
        "total_test_items": 0,
        "ngram_order": 8,
        "total_ngrams_protected": 0,
        "sources_audited": []
    }
    
    all_ngrams = set()
    all_phrases = set()
    total_items = 0
    
    for bp in benchmark_paths:
        if not os.path.exists(bp):
            print(f"Skipping missing: {bp}")
            continue
        
        with open(bp, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        items = data if isinstance(data, list) else data.get("test_cases", data.get("benchmarks", data.get("samples", [])))
        if isinstance(items, dict):
            items = list(items.values())
            
        bp_ngrams = 0
        bp_items = 0
        for item in items:
            text = ""
            if isinstance(item, dict):
                text = (item.get("prompt", "") or item.get("question", "") or item.get("input", ""))
                # Also include expected / solution to prevent leakage of target answers
                text += " " + str(item.get("expected", "") or item.get("target", "") or item.get("ground_truth", ""))
            elif isinstance(item, str):
                text = item
                
            norm = normalize_text(text)
            if not norm:
                continue
                
            toks = tok.encode(norm)
            ngrams = extract_ngrams(toks, n=8)
            all_ngrams.update(ngrams)
            bp_ngrams += len(ngrams)
            bp_items += 1
            if len(norm) > 25:
                all_phrases.add(norm.lower())
                
        total_items += bp_items
        registry["sources_audited"].append({
            "path": bp,
            "items_count": bp_items,
            "ngrams_extracted": bp_ngrams
        })
        print(f"Loaded {bp}: {bp_items} items, {bp_ngrams} 8-grams.")

    registry["total_benchmark_sources"] = len(registry["sources_audited"])
    registry["total_test_items"] = total_items
    registry["total_ngrams_protected"] = len(all_ngrams)
    
    out_path = "experiments/corpus_nr057/contamination_exclusion_registry.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)
    print(f"Contamination exclusion registry written to {out_path}")
    print(f"Total protected benchmark test items: {total_items}")
    print(f"Total unique protected 8-grams: {len(all_ngrams)}")

if __name__ == "__main__":
    build_benchmark_registry()
