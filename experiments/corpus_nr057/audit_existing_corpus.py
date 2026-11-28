import os
import sys
import json
import hashlib
import unicodedata
import re

sys.path.insert(0, os.path.abspath("."))
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

tokenizer, _ = load_tokenizer_artifacts("data/experiments/vocab_4096")

def normalize_text(text):
    if not text: return ""
    text = unicodedata.normalize("NFKC", str(text)).lower().strip()
    return " ".join(text.split())

corpora_paths = {
    "Corpus_V1_Train": "data/corpus_v1/master_corpus_train.jsonl",
    "Corpus_V1_Val": "data/corpus_v1/master_corpus_val.jsonl",
    "Corpus_V2_Train": "data/corpus_v2/corpus_v2_train.jsonl",
    "Corpus_V2_Val": "data/corpus_v2/corpus_v2_val.jsonl",
    "Corpus_V3_Train": "data/corpus_v3/corpus_v3_train.jsonl",
    "Corpus_V3_Val": "data/corpus_v3/corpus_v3_val.jsonl",
    "Corpus_V4_Train": "data/corpus_v4/corpus_v4_train.jsonl",
    "Corpus_V4_Val": "data/corpus_v4/corpus_v4_val.jsonl",
    "NR051_Pretrain_Train": "data/pretraining_nr051/pretrain_train.jsonl",
    "NR051_Pretrain_Val": "data/pretraining_nr051/pretrain_val.jsonl"
}

inventory = {}
all_hashes_seen = set()
unique_corpus_tokens = 0

for name, path in corpora_paths.items():
    if not os.path.exists(path):
        continue
        
    doc_count = 0
    raw_bytes = os.path.getsize(path)
    norm_bytes = 0
    total_tokens = 0
    doc_lens = []
    unique_hashes = set()
    exact_duplicates = 0
    
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            doc_count += 1
            d = json.loads(line)
            txt = d.get("text", "")
            if not txt and "prompt" in d and "response" in d:
                txt = f"{d['prompt']} {d['response']}"
            
            norm = normalize_text(txt)
            norm_bytes += len(norm.encode("utf-8"))
            h = hashlib.sha256(norm.encode("utf-8")).hexdigest()
            if h in unique_hashes:
                exact_duplicates += 1
            else:
                unique_hashes.add(h)
                
            toks = tokenizer.encode(txt)
            tok_len = len(toks)
            total_tokens += tok_len
            doc_lens.append(tok_len)
            
    doc_lens.sort()
    avg_len = sum(doc_lens) / len(doc_lens) if doc_lens else 0
    median_len = doc_lens[len(doc_lens) // 2] if doc_lens else 0
    
    inventory[name] = {
        "file_path": path,
        "doc_count": doc_count,
        "unique_normalized_docs": len(unique_hashes),
        "exact_duplicates": exact_duplicates,
        "duplicate_rate_pct": round(exact_duplicates / doc_count * 100, 2) if doc_count else 0,
        "raw_bytes": raw_bytes,
        "normalized_bytes": norm_bytes,
        "total_tokens": total_tokens,
        "avg_tokens_per_doc": round(avg_len, 1),
        "median_tokens_per_doc": median_len
    }
    print(f"[{name}] docs={doc_count:,} unique={len(unique_hashes):,} tokens={total_tokens:,} dup_rate={inventory[name]['duplicate_rate_pct']}%")

with open("experiments/corpus_nr057/existing_corpus_inventory.json", "w") as f:
    json.dump(inventory, f, indent=2)

print("\nInventory complete and written to existing_corpus_inventory.json")
