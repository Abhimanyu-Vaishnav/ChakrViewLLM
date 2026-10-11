"""
Corpus and Shard Quality Audit Script for Milestone 1.
"""
import hashlib
import json
from pathlib import Path
from collections import Counter
import numpy as np

def run_audit():
    repo_root = Path(__file__).resolve().parent.parent
    stage_c = repo_root / "data" / "tokenized" / "stage_c"
    
    report = {
        "splits": {},
        "integrity_errors": [],
        "token_statistics": {},
        "top_transitions": [],
        "n_grams_overlap": {}
    }
    
    all_tokens = {}
    
    for split in ["train", "validation", "test"]:
        split_dir = stage_c / split
        meta_file = split_dir / "metadata.json"
        with open(meta_file, "r", encoding="utf-8") as f:
            meta = json.load(f)
            
        vocab_size = meta["vocab_size"]
        split_tokens_list = []
        shard_audits = []
        
        for shard_info in meta["shards"]:
            sfile = split_dir / shard_info["filename"]
            if not sfile.exists():
                report["integrity_errors"].append(f"Missing file: {sfile}")
                continue
            content = sfile.read_bytes()
            computed_sha = hashlib.sha256(content).hexdigest()
            if computed_sha != shard_info["sha256"]:
                report["integrity_errors"].append(f"SHA mismatch for {sfile.name}: expected {shard_info['sha256']}, got {computed_sha}")
                
            tokens = np.frombuffer(content, dtype=np.uint16)
            if len(tokens) != shard_info["token_count"]:
                report["integrity_errors"].append(f"Token count mismatch for {sfile.name}: expected {shard_info['token_count']}, got {len(tokens)}")
                
            if len(tokens) > 0:
                min_id = int(tokens.min())
                max_id = int(tokens.max())
                if min_id < 0 or max_id >= vocab_size:
                    report["integrity_errors"].append(f"Vocab range error in {sfile.name}: range [{min_id}, {max_id}], vocab_size {vocab_size}")
            
            shard_audits.append({
                "filename": shard_info["filename"],
                "token_count": len(tokens),
                "sha256_match": computed_sha == shard_info["sha256"],
                "min_token": int(tokens.min()) if len(tokens) > 0 else None,
                "max_token": int(tokens.max()) if len(tokens) > 0 else None,
            })
            split_tokens_list.append(tokens)
            
        full_split_tokens = np.concatenate(split_tokens_list) if split_tokens_list else np.array([], dtype=np.uint16)
        all_tokens[split] = full_split_tokens
        
        report["splits"][split] = {
            "metadata_total_tokens": meta["total_tokens"],
            "actual_total_tokens": len(full_split_tokens),
            "match": len(full_split_tokens) == meta["total_tokens"],
            "shards_count": len(shard_audits),
            "tokenizer_checksum": meta.get("tokenizer_checksum"),
            "source_manifest_hash": meta.get("source_manifest_hash"),
        }
    
    # Vocabulary & distribution statistics on train split
    train_tokens = all_tokens["train"]
    counts = Counter(train_tokens.tolist())
    report["token_statistics"] = {
        "train_unique_tokens": len(counts),
        "train_total_tokens": len(train_tokens),
        "special_tokens_counts": {
            "token_0": counts.get(0, 0),
            "token_1": counts.get(1, 0),
            "token_2 (PAD)": counts.get(2, 0),
            "token_3": counts.get(3, 0),
        },
        "most_common_tokens": counts.most_common(10)
    }
    
    # Transition distribution (first 200,000 pairs of train)
    sample_pairs = list(zip(train_tokens[:200000], train_tokens[1:200001]))
    pair_counts = Counter(sample_pairs)
    report["top_transitions"] = [
        {"from": int(k[0]), "to": int(k[1]), "count": v}
        for k, v in pair_counts.most_common(10)
    ]
    
    # Overlap / Leakage audit across splits using 16-token sliding windows on first 50,000 tokens
    val_16 = set(tuple(all_tokens["validation"][i:i+16].tolist()) for i in range(0, min(len(all_tokens["validation"]), 50000) - 16, 16))
    train_16 = set(tuple(train_tokens[i:i+16].tolist()) for i in range(0, min(len(train_tokens), 100000) - 16, 16))
    overlap = len(val_16.intersection(train_16))
    report["n_grams_overlap"] = {
        "val_16grams_sampled": len(val_16),
        "train_16grams_sampled": len(train_16),
        "overlapping_16grams": overlap
    }
    
    out_path = repo_root / "artifacts" / "corpus_audit_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        
    print("Corpus Audit complete. Errors:", len(report["integrity_errors"]))
    print("Report written to:", out_path)

if __name__ == "__main__":
    run_audit()
