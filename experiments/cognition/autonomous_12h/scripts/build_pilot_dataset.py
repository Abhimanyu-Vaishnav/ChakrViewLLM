"""
Stage 4 Dataset Builder: Creates clean, audited, deduplicated dataset
for the autonomous 12-hour conversational pilot.
Combines:
- 144 samples of CBR4 foundation (persona, greetings, grounding)
- 360 samples of CBR5 (key-value updates, multi-turn dialogues, city locations)
Splits deterministically into:
- Train: 400 samples
- Validation: 50 samples
- Test (Held-out): 54 samples
Total: 504 samples
"""

import json
import hashlib
import random
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
CBR4_TRAIN = REPO_ROOT / "data" / "sft" / "cbr_sft_04" / "train.jsonl"
CBR5_TRAIN = REPO_ROOT / "data" / "sft" / "cbr_sft_05" / "train.jsonl"
OUT_DIR = REPO_ROOT / "experiments" / "cognition" / "autonomous_12h" / "conversational_pilot" / "dataset"

def hash_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(8192 * 1024):
            h.update(chunk)
    return h.hexdigest()

def main():
    print("=== BUILDING CONVERSATIONAL PILOT DATASET ===")
    all_samples = []
    seen_prompts = set()
    
    # Load CBR4
    with open(CBR4_TRAIN, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            prompt = d["prompt"].strip()
            if prompt not in seen_prompts:
                seen_prompts.add(prompt)
                all_samples.append({
                    "id": d["id"],
                    "source": "cbr_sft_04",
                    "category": d.get("category", "foundation"),
                    "prompt": d["prompt"],
                    "response": d["response"]
                })
                
    # Load CBR5
    with open(CBR5_TRAIN, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            prompt = d["prompt"].strip()
            if prompt not in seen_prompts:
                seen_prompts.add(prompt)
                all_samples.append({
                    "id": d["id"],
                    "source": "cbr_sft_05",
                    "category": d.get("category", "binding_multiturn"),
                    "prompt": d["prompt"],
                    "response": d["response"]
                })
                
    print(f"Total deduplicated candidate samples: {len(all_samples)}")
    
    # Shuffle deterministically with seed 101
    rng = random.Random(101)
    rng.shuffle(all_samples)
    
    # Split: Train 400, Val 50, Test 54
    train_samples = all_samples[:400]
    val_samples = all_samples[400:450]
    test_samples = all_samples[450:504]
    
    print(f"Split counts -> Train: {len(train_samples)}, Val: {len(val_samples)}, Test: {len(test_samples)}")
    
    train_path = OUT_DIR / "train.jsonl"
    val_path = OUT_DIR / "val.jsonl"
    test_path = OUT_DIR / "test_held_out.jsonl"
    
    for path, data in [(train_path, train_samples), (val_path, val_samples), (test_path, test_samples)]:
        with open(path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                
    manifest = {
        "dataset_name": "Autonomous-12H-Conversational-Pilot-Dataset",
        "seed": 101,
        "counts": {
            "total": len(all_samples),
            "train": len(train_samples),
            "val": len(val_samples),
            "test_held_out": len(test_samples)
        },
        "hashes": {
            "train_sha256": hash_file(train_path),
            "val_sha256": hash_file(val_path),
            "test_sha256": hash_file(test_path)
        }
    }
    
    manifest_path = OUT_DIR / "dataset_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        
    print(f"Saved manifest to: {manifest_path}")
    print("Dataset generation completed successfully.")

if __name__ == "__main__":
    main()
