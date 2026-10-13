"""
Batch 131–134: Relational Rule Induction (RRI-1) Dataset Generator
Tests conditional associative routing:
Given binding context `CONTEXT: {K1}={V1}, {K2}={V2}\nQUERY: {K_query}\nANSWER: ` -> emits `{V_target}`.

Partitions:
1. Partition 1 (Familiar pairs, familiar query): memorization check (N=10)
2. Partition 2 (Unseen combinations of familiar entities): compositional routing check (N=15)
3. Partition 3 (Held-out unseen entities): abstract zero-shot transfer check (N=10)
4. Partition 4 (Negative controls / unanswerable queries): fail-closed refusal check (N=10)
"""
import json
import random
import hashlib
from pathlib import Path

DATA_DIR = Path("experiments/batch131_134_mission_alignment_audit/data")

# Distinct uppercase letters A-Z (all strictly 1-token in ChakrView tokenizer)
TRAIN_KEYS = ["A", "B", "C", "D", "E", "F", "G", "H"]
TRAIN_VALS = ["M", "N", "P", "Q", "R", "S", "T", "U"]

HELDOUT_KEYS = ["W", "X"]
HELDOUT_VALS = ["Y", "Z"]

TEMPLATES = [
    "CONTEXT: {k1}={v1}, {k2}={v2}\nQUERY: {q}\nANSWER: ",
    "BINDINGS:\n- {k1} maps to {v1}\n- {k2} maps to {v2}\nLOOKUP: {q}\nRESULT: ",
    "Given pairs ({k1}:{v1}) and ({k2}:{v2}). What is {q}? Answer: ",
]

NEGATIVE_PROMPTS = [
    "CONTEXT: A=M, B=N\nQUERY: Z\nANSWER: ",
    "CONTEXT: C=P, D=Q\nQUERY: X\nANSWER: ",
    "BINDINGS:\n- E maps to R\nLOOKUP: missing\nRESULT: ",
    "Given pairs (G:T). What is unknown? Answer: ",
    "CONTEXT: H=U\nQUERY: none\nANSWER: ",
    "What is the capital of France?",
    "Explain quantum entanglement.",
    "Calculate 12 + 15.",
    "Write a poem about trees.",
    "Perform lookup on empty dictionary.",
]

def format_record(ex_id: str, prompt: str, target: str, is_routable: bool, category: str, meta: dict) -> dict:
    return {
        "id": ex_id,
        "category": category,
        "is_routable": is_routable,
        "prompt": prompt,
        "target": target,
        "full_text": prompt + target,
        "meta": meta,
    }

def generate_datasets(seed: int = 42):
    rng = random.Random(seed)
    
    # Generate all pairwise combinations from train pool: (k1, v1) and (k2, v2) where k1 != k2
    train_pairs = []
    for k in TRAIN_KEYS:
        for v in TRAIN_VALS:
            train_pairs.append((k, v))
            
    rng.shuffle(train_pairs)
    
    # We construct 80 training records of two-binding contexts
    train_records = []
    seen_combos = set()
    for i in range(80):
        # Pick 2 distinct keys
        k1, k2 = rng.sample(TRAIN_KEYS, 2)
        v1, v2 = rng.sample(TRAIN_VALS, 2)
        q = k1 if rng.random() < 0.5 else k2
        ans = v1 if q == k1 else v2
        tmpl = TEMPLATES[i % len(TEMPLATES)]
        prompt = tmpl.format(k1=k1, v1=v1, k2=k2, v2=v2, q=q)
        combo_key = (k1, v1, k2, v2, q)
        seen_combos.add(combo_key)
        rec = format_record(
            f"TRAIN-ROUTE-{i:03d}", prompt, ans, True, "train_routing",
            {"k1": k1, "v1": v1, "k2": k2, "v2": v2, "q": q, "ans": ans}
        )
        train_records.append(rec)
        
    # Add 20 train negative controls
    for j in range(20):
        prompt = NEGATIVE_PROMPTS[j % len(NEGATIVE_PROMPTS)]
        rec = format_record(
            f"TRAIN-NEG-{j:03d}", prompt, "UNKNOWN", False, "train_negative_control",
            {"q": "unanswerable", "ans": "UNKNOWN"}
        )
        train_records.append(rec)
        
    rng.shuffle(train_records)
    
    # Build Test Set (45 records)
    test_records = []
    
    # Partition 1: Familiar combinations (memorization check, N=10)
    familiar_sample = list(seen_combos)[:10]
    for i, (k1, v1, k2, v2, q) in enumerate(familiar_sample):
        ans = v1 if q == k1 else v2
        tmpl = TEMPLATES[0]
        prompt = tmpl.format(k1=k1, v1=v1, k2=k2, v2=v2, q=q)
        test_records.append(format_record(
            f"TEST-FAMILIAR-{i:02d}", prompt, ans, True, "familiar_combinations",
            {"k1": k1, "v1": v1, "k2": k2, "v2": v2, "q": q, "ans": ans}
        ))
        
    # Partition 2: Unseen combinations of familiar entities (compositional routing, N=15)
    p2_count = 0
    attempts = 0
    while p2_count < 15 and attempts < 1000:
        attempts += 1
        k1, k2 = rng.sample(TRAIN_KEYS, 2)
        v1, v2 = rng.sample(TRAIN_VALS, 2)
        q = k1 if rng.random() < 0.5 else k2
        ans = v1 if q == k1 else v2
        combo_key = (k1, v1, k2, v2, q)
        if combo_key not in seen_combos:
            tmpl = TEMPLATES[1]  # Frame in distinct template
            prompt = tmpl.format(k1=k1, v1=v1, k2=k2, v2=v2, q=q)
            test_records.append(format_record(
                f"TEST-UNSEEN-COMBO-{p2_count:02d}", prompt, ans, True, "unseen_combinations",
                {"k1": k1, "v1": v1, "k2": k2, "v2": v2, "q": q, "ans": ans}
            ))
            p2_count += 1
            
    # Partition 3: Held-out unseen entities (zero-shot transfer, N=10)
    for i in range(10):
        # uses heldout keys/vals
        k1 = HELDOUT_KEYS[0]
        v1 = HELDOUT_VALS[0]
        k2 = HELDOUT_KEYS[1]
        v2 = HELDOUT_VALS[1]
        q = k1 if (i % 2 == 0) else k2
        ans = v1 if q == k1 else v2
        tmpl = TEMPLATES[i % len(TEMPLATES)]
        prompt = tmpl.format(k1=k1, v1=v1, k2=k2, v2=v2, q=q)
        test_records.append(format_record(
            f"TEST-HELDOUT-ENTITIES-{i:02d}", prompt, ans, True, "heldout_entities",
            {"k1": k1, "v1": v1, "k2": k2, "v2": v2, "q": q, "ans": ans}
        ))
        
    # Partition 4: Negative controls (clean refusal, N=10)
    for i, neg_p in enumerate(NEGATIVE_PROMPTS):
        test_records.append(format_record(
            f"TEST-NEG-{i:02d}", neg_p, "UNKNOWN", False, "test_negative_control",
            {"q": "unanswerable", "ans": "UNKNOWN"}
        ))
        
    return train_records, test_records

def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    train, test = generate_datasets(seed=42)
    
    files = {
        "train.jsonl": train,
        "test.jsonl": test,
    }
    
    manifest = {
        "benchmark": "Batch 131-134 Relational Rule Induction (RRI-1)",
        "seed": 42,
        "counts": {k: len(v) for k, v in files.items()},
        "train_keys": TRAIN_KEYS,
        "train_vals": TRAIN_VALS,
        "heldout_keys": HELDOUT_KEYS,
        "heldout_vals": HELDOUT_VALS,
        "file_hashes": {},
    }
    
    for name, recs in files.items():
        p = DATA_DIR / name
        with open(p, "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        manifest["file_hashes"][name] = hashlib.sha256(p.read_bytes()).hexdigest()
        print(f"Saved {name}: {len(recs)} records (SHA-256: {manifest['file_hashes'][name]})")
        
    m_p = DATA_DIR / "manifest.json"
    with open(m_p, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest written to {m_p}")

if __name__ == "__main__":
    main()
