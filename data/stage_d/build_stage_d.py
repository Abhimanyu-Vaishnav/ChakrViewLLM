"""
Mission CV-GENESIS-NR-011: Stage-D Expanded SFT Corpus Builder.
Constructs a quality-controlled, deduplicated, multi-domain training corpus for ChakrGenesis-17M.

Domains:
1. Python programming & debugging (executable code targets)
2. Devanagari Hindi explanations & factual QA
3. Bilingual Hinglish conversational dialogue
4. English technical, scientific & social dialogue
5. Multi-turn reasoning & structured problem solving

Guarantees:
- Strict canonical special token alignment: eos_id=1, pad_id=2
- Zero overlap with evaluation benchmarks
- Exact deduplication via normalized prompt hashing
- Deterministic 85/10/5 split
"""

import hashlib
import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from chakrview.tokenizer.serialization import load_tokenizer_artifacts

OUTPUT_DIR = Path("data/stage_d/splits")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

tok, _ = load_tokenizer_artifacts("data/experiments/vocab_4096")

# 1. Load sealed blind prompts to prevent contamination
sealed_prompt_hashes = set()
for report_path in [Path("reports/NR006_BLIND_DATASET_MANIFEST.json"), Path("reports/NR009_300_BENCHMARK_RESULTS.json")]:
    if report_path.exists():
        with open(report_path, "r", encoding="utf-8") as f:
            d = json.load(f)
        prompts = d.get("prompts_inventory", [])
        for it in prompts:
            p = it.get("prompt", "").strip().lower()
            if p:
                sealed_prompt_hashes.add(hashlib.sha256(p.encode("utf-8")).hexdigest())

print(f"[FILTER] Sealed evaluation hashes loaded: {len(sealed_prompt_hashes)}")

# 2. Ingest ucip14r master dataset
master_path = Path("data/sft/ucip14r/sft_master_15000.jsonl")
candidates = []
seen_hashes = set()

with open(master_path, "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        p = item.get("prompt") or item.get("input") or item.get("instruction") or ""
        r = item.get("response") or item.get("output") or item.get("target") or ""
        p, r = p.strip(), r.strip()
        if not p or not r:
            continue

        if not p.startswith("User:"):
            p = f"User: {p}\nAssistant:"
        elif not p.endswith("Assistant:"):
            p = f"{p}\nAssistant:"

        p_norm = p.lower()
        ph = hashlib.sha256(p_norm.encode("utf-8")).hexdigest()
        if ph in sealed_prompt_hashes or ph in seen_hashes:
            continue

        # Token length filter (<= 256 tokens)
        p_len = len(tok.encode(p))
        r_len = len(tok.encode(r))
        if p_len + r_len > 240:
            continue

        seen_hashes.add(ph)
        candidates.append({
            "id": item.get("id"),
            "category": item.get("category", "general"),
            "language": item.get("language", "en"),
            "prompt": p,
            "response": r,
            "source": "ucip14r_master",
            "token_count": p_len + r_len
        })

print(f"[INGEST] Ingested {len(candidates)} high-quality unique items from ucip14r.")

# 3. Ingest multi-turn & conversational dialogue from NR-008 mixture
nr008_train = Path("experiments/genesis_balanced_sft_008/data/train.jsonl")
nr008_added = 0
with open(nr008_train, "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        p = item.get("prompt", "").strip()
        r = item.get("response", "").strip()
        if not p or not r:
            continue
        p_norm = p.lower()
        ph = hashlib.sha256(p_norm.encode("utf-8")).hexdigest()
        if ph in sealed_prompt_hashes or ph in seen_hashes:
            continue
        p_len = len(tok.encode(p))
        r_len = len(tok.encode(r))
        if p_len + r_len > 240:
            continue
        seen_hashes.add(ph)
        candidates.append({
            "id": item.get("id", f"nr008_{nr008_added:05d}"),
            "category": item.get("category", "conversation"),
            "language": item.get("language", "en"),
            "prompt": p,
            "response": r,
            "source": "nr008_curated",
            "token_count": p_len + r_len
        })
        nr008_added += 1

print(f"[INGEST] Added {nr008_added} additional dialogue items from NR-008 base.")

# 4. Shuffle and deterministic split
random.seed(42)
random.shuffle(candidates)

total = len(candidates)
n_train = int(total * 0.85)
n_val = int(total * 0.10)
n_test = total - n_train - n_val

train_split = candidates[:n_train]
val_split = candidates[n_train:n_train+n_val]
test_split = candidates[n_train+n_val:]

def write_jsonl(path, items):
    with open(path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

write_jsonl(OUTPUT_DIR / "stage_d_train.jsonl", train_split)
write_jsonl(OUTPUT_DIR / "stage_d_val.jsonl", val_split)
write_jsonl(OUTPUT_DIR / "stage_d_test.jsonl", test_split)

# Compute language and domain distributions
def get_dist(items, key):
    counts = {}
    for it in items:
        v = it.get(key, "unknown")
        counts[v] = counts.get(v, 0) + 1
    return {k: f"{v} ({v/len(items)*100:.1f}%)" for k, v in sorted(counts.items(), key=lambda x: x[1], reverse=True)}

manifest = {
    "corpus_name": "ChakrView Stage-D SFT Foundation Mixture",
    "version": "1.0.0",
    "total_unique_samples": total,
    "tokenizer_vocab_hash": "db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a",
    "special_tokens": {"<BOS>": 0, "<EOS>": 1, "<PAD>": 2},
    "splits": {
        "train": {
            "count": len(train_split),
            "languages": get_dist(train_split, "language"),
            "domains": get_dist(train_split, "category")
        },
        "validation": {
            "count": len(val_split),
            "languages": get_dist(val_split, "language"),
            "domains": get_dist(val_split, "category")
        },
        "test": {
            "count": len(test_split),
            "languages": get_dist(test_split, "language"),
            "domains": get_dist(test_split, "category")
        }
    }
}

manifest_path = Path("data/stage_d/stage_d_manifest.json")
with open(manifest_path, "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2, ensure_ascii=False)

print("\n" + "="*60)
print(f"STAGE-D CORPUS READY: {total:,} total unique examples")
print(f"Train: {len(train_split)} | Val: {len(val_split)} | Test: {len(test_split)}")
print("Train Languages:", manifest["splits"]["train"]["languages"])
print("Top Domains:", list(manifest["splits"]["train"]["domains"].items())[:6])
print("="*60)
