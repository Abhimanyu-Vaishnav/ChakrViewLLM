"""
experiments/nr052_workspace/audit_all_data.py

Phase 1 audit script:
- Scans all text, json, jsonl files in the repository.
- Extracts records, determines document counts, character counts, and exact token counts using canonical 4096 tokenizer.
- Analyzes duplicates, language distributions, and categorizes suitability (Pretraining vs SFT vs Evaluation).
- Outputs reports/NR052_DATA_INVENTORY.json and summary statistics.
"""

import sys
import os
import json
import hashlib
from pathlib import Path
from collections import Counter, defaultdict

sys.path.insert(0, ".")
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def detect_language(text):
    # Fast heuristic language detector
    devanagari_count = sum(1 for c in text if '\u0900' <= c <= '\u097F')
    latin_count = sum(1 for c in text if 'a' <= c.lower() <= 'z')
    total = devanagari_count + latin_count
    if total == 0:
        return "other"
    dev_ratio = devanagari_count / total
    if dev_ratio > 0.4:
        return "hi"
    elif dev_ratio > 0.05:
        return "hi-en"  # Hinglish / mixed
    else:
        # Check if code
        code_markers = ["def ", "class ", "import ", "return ", "function(", "var ", "const ", "=>", "for(", "while("]
        if any(m in text for m in code_markers):
            return "code"
        return "en"

def main():
    print("=== PHASE 1: COMPLETE DATA INVENTORY AUDIT ===")
    tokenizer, _ = load_tokenizer_artifacts("data/experiments/vocab_4096", verify_checksums=False)
    
    # Target data sources
    source_specs = [
        ("Raw Stage-B Foundation", "data/raw", ["*.txt"]),
        ("Processed Stage-B Splits", "data/processed/stage_b", ["*.txt"]),
        ("Corpus V1", "data/corpus_v1", ["*.jsonl"]),
        ("Corpus V2", "data/corpus_v2", ["*.jsonl"]),
        ("Corpus V3", "data/corpus_v3", ["*.jsonl"]),
        ("Pretraining NR-051", "data/pretraining_nr051", ["*.jsonl"]),
        ("Stage-D QA Dataset", "data/stage_d/splits", ["*.jsonl"]),
        ("Stage-E Corpus 013", "data/stage_e_corpus_013", ["*.jsonl"]),
        ("Stage-F Corpus 014", "data/stage_f_corpus_014", ["*.jsonl"]),
        ("SFT UCIP14R Master", "data/sft/ucip14r", ["sft_master_15000.jsonl"]),
        ("SFT UCIP17 Master", "data/sft/ucip17", ["sft_master_15000.jsonl"]),
        ("SFT UCIP20 Bind", "data/sft/ucip20_bind", ["*.jsonl"]),
        ("SFT CBR & EXP Mixtures", "data/sft/exp34_candidate_v3", ["*.jsonl"]),
        ("SFT M3 Conversational", "data/sft/m3_conversational", ["*.jsonl"]),
        ("Sealed 120 Benchmark (Eval Only)", "experiments/fast_track_047_049", ["sealed_120_benchmark.json"]),
        ("Matched 1000 Benchmark (Eval Only)", "data/stage_j_020", ["fresh_matched_benchmark_1000.json"])
    ]
    
    inventory = []
    global_doc_hashes = set()
    total_unique_pretrain_candidates = 0
    total_unique_sft_candidates = 0
    
    for label, base_dir, patterns in source_specs:
        base_path = Path(base_dir)
        if not base_path.exists():
            continue
            
        matched_files = []
        for pat in patterns:
            matched_files.extend(list(base_path.rglob(pat)))
            
        for fpath in matched_files:
            file_size = fpath.stat().st_size
            doc_count = 0
            char_count = 0
            token_count = 0
            unique_hashes = set()
            lang_dist = Counter()
            suitability = "UNKNOWN"
            
            # Determine suitability
            path_str = str(fpath).lower()
            if "benchmark" in path_str or "sealed" in path_str:
                suitability = "EVALUATION_SEALED"
            elif "sft" in path_str or "mixture" in path_str or "bind" in path_str:
                suitability = "SFT_CONVERSATIONAL"
            elif "pretrain" in path_str or "stage_b" in path_str or "raw" in path_str:
                suitability = "PRETRAINING_NATURAL"
            elif "corpus_v" in path_str:
                suitability = "SFT_CONVERSATIONAL"
            else:
                suitability = "GENERAL"
                
            # Process content
            if fpath.suffix == ".txt":
                try:
                    text = fpath.read_text(encoding="utf-8", errors="ignore")
                    paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 30]
                    doc_count = len(paragraphs)
                    char_count = len(text)
                    tokens = tokenizer.encode(text)
                    token_count = len(tokens)
                    for p in paragraphs:
                        h = sha256_text(p)
                        unique_hashes.add(h)
                        lang_dist[detect_language(p)] += 1
                except Exception as e:
                    print(f"Error reading {fpath}: {e}")
            elif fpath.suffix in [".jsonl", ".json"]:
                try:
                    if fpath.suffix == ".jsonl":
                        with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                            for line in f:
                                if not line.strip(): continue
                                obj = json.loads(line)
                                doc_count += 1
                                # Extract text
                                if "text" in obj:
                                    t = obj["text"]
                                elif "prompt" in obj and "response" in obj:
                                    t = obj["prompt"] + "\n" + obj["response"]
                                elif "messages" in obj:
                                    t = "\n".join(m.get("content", "") for m in obj["messages"])
                                else:
                                    t = json.dumps(obj)
                                    
                                char_count += len(t)
                                h = sha256_text(t)
                                unique_hashes.add(h)
                                lang_dist[detect_language(t)] += 1
                                # Sample token count for speed if very large
                                if doc_count <= 2000:
                                    token_count += len(tokenizer.encode(t))
                        if doc_count > 2000:
                            token_count = int(token_count * (doc_count / 2000))
                    else: # .json
                        with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                            data = json.load(f)
                            if isinstance(data, list):
                                doc_count = len(data)
                                for item in data:
                                    t = json.dumps(item) if not isinstance(item, str) else item
                                    char_count += len(t)
                                    unique_hashes.add(sha256_text(t))
                                    lang_dist[detect_language(t)] += 1
                                    if len(unique_hashes) <= 1000:
                                        token_count += len(tokenizer.encode(t))
                                if doc_count > 1000:
                                    token_count = int(token_count * (doc_count / min(doc_count, 1000)))
                            elif isinstance(data, dict):
                                doc_count = 1
                                t = json.dumps(data)
                                char_count = len(t)
                                token_count = len(tokenizer.encode(t[:20000]))
                                unique_hashes.add(sha256_text(t))
                except Exception as e:
                    print(f"Error reading {fpath}: {e}")
                    
            dup_rate = 1.0 - (len(unique_hashes) / max(1, doc_count))
            
            entry = {
                "source_label": label,
                "file_path": str(fpath).replace("\\", "/"),
                "file_size_bytes": file_size,
                "doc_count": doc_count,
                "unique_doc_count": len(unique_hashes),
                "char_count": char_count,
                "token_count": token_count,
                "duplicate_rate": round(dup_rate, 4),
                "suitability": suitability,
                "language_distribution": dict(lang_dist)
            }
            inventory.append(entry)
            print(f"Audited: {fpath.name:35} | Docs: {doc_count:6} | Unique: {len(unique_hashes):6} | Toks: {token_count:9,} | Type: {suitability}")
            
    # Save inventory json
    out_json = Path("reports/NR052_DATA_INVENTORY.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(inventory, f, indent=2)
    print(f"\nInventory saved to {out_json}")

if __name__ == "__main__":
    main()
