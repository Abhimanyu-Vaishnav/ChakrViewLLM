"""
audit_context_benchmark.py
ChakrView Experiment 14: Context Grounding Benchmark Audit.

Verifies:
1. Exact prompt string overlap between Train, Val, and Test splits (must be 0).
2. Entity identifier leakage between splits (train entity names must not appear in test prompts).
3. Token sequence length distribution under 512 context limit.
4. Lossless BPE roundtrip encoding.
5. Verification of all 10 distinct task categories.
"""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Dict, List, Set
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

EXP_DIR = Path(__file__).resolve().parent
DATA_DIR = EXP_DIR / "data"

from chakrview.tokenizer.serialization import load_tokenizer_artifacts

def load_jsonl(path: Path) -> List[Dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records

def main():
    print("=" * 70)
    print("CHAKRVIEW: CONTEXT BENCHMARK INTEGRITY & LEAKAGE AUDIT")
    print("=" * 70)
    
    train_data = load_jsonl(DATA_DIR / "benchmark_train.jsonl")
    val_data = load_jsonl(DATA_DIR / "benchmark_val.jsonl")
    test_data = load_jsonl(DATA_DIR / "benchmark_test.jsonl")
    
    print(f"Loaded: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
    
    # 1. Exact prompt overlap check
    train_prompts = {d["prompt"] for d in train_data}
    val_prompts = {d["prompt"] for d in val_data}
    test_prompts = {d["prompt"] for d in test_data}
    
    t_v = train_prompts.intersection(val_prompts)
    t_t = train_prompts.intersection(test_prompts)
    v_t = val_prompts.intersection(test_prompts)
    
    print(f"\n1. Split Prompt Leakage Check:")
    print(f"   Train-Val Overlap:  {len(t_v)} examples")
    print(f"   Train-Test Overlap: {len(t_t)} examples")
    print(f"   Val-Test Overlap:   {len(v_t)} examples")
    assert len(t_v) == 0, "Train-Val prompt overlap detected!"
    assert len(t_t) == 0, "Train-Test prompt overlap detected!"
    assert len(v_t) == 0, "Val-Test prompt overlap detected!"
    print("   [PASS] Zero exact prompt leakage between splits.")
    
    # 2. Entity identifier leakage check
    print(f"\n2. Entity Token Leakage Check:")
    train_entities = set()
    for d in train_data:
        m = d.get("metadata", {})
        for k in ["target_entity", "function_name", "new_target", "cancelled_target", "target_pod", "target_server"]:
            if k in m:
                train_entities.add(str(m[k]))
                
    test_entity_leakage = 0
    for d in test_data:
        m = d.get("metadata", {})
        for k in ["target_entity", "function_name", "new_target", "cancelled_target", "target_pod", "target_server"]:
            if k in m and str(m[k]) in train_entities:
                test_entity_leakage += 1
                
    print(f"   Train Entities Identified: {len(train_entities)}")
    print(f"   Test Examples Leaking Train Entities: {test_entity_leakage}")
    assert test_entity_leakage == 0, f"Found {test_entity_leakage} test entities leaking from train!"
    print("   [PASS] Held-out test entities are strictly disjoint from training entities.")
    
    # 3. Tokenizer compatibility
    tok_dir = Path("data/experiments/vocab_4096")
    tokenizer, _ = load_tokenizer_artifacts(tok_dir, verify_checksums=False)
    
    max_len = 0
    for split_name, ds in [("train", train_data), ("val", val_data), ("test", test_data)]:
        for item in ds:
            p_toks = tokenizer.encode(item["prompt"])
            r_toks = tokenizer.encode(item["target_response"])
            tot_len = len(p_toks) + len(r_toks)
            if tot_len > max_len:
                max_len = tot_len
            assert tokenizer.decode(p_toks) == item["prompt"], "Lossy prompt BPE encoding"
            assert tokenizer.decode(r_toks) == item["target_response"], "Lossy target BPE encoding"
            
    print(f"\n3. Tokenization & Sequence Length:")
    print(f"   Max sequence length: {max_len} tokens (Context budget: 512)")
    assert max_len <= 512, "Sequence exceeds context budget!"
    print("   [PASS] All benchmark sequences fit within context window and decode losslessly.")
    
    # 4. Category balance
    print(f"\n4. Category Balance:")
    for split_name, ds in [("Train", train_data), ("Val", val_data), ("Test", test_data)]:
        counts = {}
        for item in ds:
            c = item["category"]
            counts[c] = counts.get(c, 0) + 1
        print(f"   {split_name}: {len(counts)} categories, {min(counts.values())} to {max(counts.values())} samples per category.")
        assert len(counts) == 10, f"Missing categories in {split_name}"
        
    print("\n[SUCCESS] Benchmark passed all leakage, entity isolation, and integrity checks.")

if __name__ == "__main__":
    main()
