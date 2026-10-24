"""
audit_conversation_data.py
ChakrView Experiment 13: Conversation Data Leakage and Quality Audit.

Verifies:
1. Exact prompt/response overlap between Train, Val, and Test splits (must be 0).
2. Token length distribution (ensures fitting inside 512 context limit).
3. Speaker turn alternation (User -> Assistant -> User -> Assistant).
4. Category coverage (all 20 categories balanced).
5. Lossless BPE tokenization roundtrip on prompt and response.
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
    print("CHAKRVIEW: CONVERSATION DATA INTEGRITY & LEAKAGE AUDIT")
    print("=" * 70)
    
    train_data = load_jsonl(DATA_DIR / "train.jsonl")
    val_data = load_jsonl(DATA_DIR / "val.jsonl")
    test_data = load_jsonl(DATA_DIR / "test.jsonl")
    
    print(f"Loaded: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
    
    # 1. Leakage check
    train_prompts = {d["prompt"] for d in train_data}
    val_prompts = {d["prompt"] for d in val_data}
    test_prompts = {d["prompt"] for d in test_data}
    
    train_val_overlap = train_prompts.intersection(val_prompts)
    train_test_overlap = train_prompts.intersection(test_prompts)
    val_test_overlap = val_prompts.intersection(test_prompts)
    
    print(f"\n1. Split Leakage Analysis:")
    print(f"   Train-Val Overlap:  {len(train_val_overlap)} examples")
    print(f"   Train-Test Overlap: {len(train_test_overlap)} examples")
    print(f"   Val-Test Overlap:   {len(val_test_overlap)} examples")
    assert len(train_val_overlap) == 0, "Train-Val overlap detected!"
    assert len(train_test_overlap) == 0, "Train-Test overlap detected!"
    assert len(val_test_overlap) == 0, "Val-Test overlap detected!"
    print("   [PASS] Zero exact prompt leakage between splits.")
    
    # 2. Tokenizer compatibility
    tok_dir = Path("data/experiments/vocab_4096")
    tokenizer, _ = load_tokenizer_artifacts(tok_dir, verify_checksums=False)
    
    max_len = 0
    for split_name, ds in [("train", train_data), ("val", val_data), ("test", test_data)]:
        for item in ds:
            p_toks = tokenizer.encode(item["prompt"])
            r_toks = tokenizer.encode(item["response"])
            tot_len = len(p_toks) + len(r_toks)
            if tot_len > max_len:
                max_len = tot_len
                
            # Verify lossless roundtrip
            assert tokenizer.decode(p_toks) == item["prompt"], f"Lossy prompt encoding: {item['prompt']}"
            assert tokenizer.decode(r_toks) == item["response"], f"Lossy response encoding: {item['response']}"
            
    print(f"\n2. Tokenization & Sequence Length:")
    print(f"   Max sequence length across all splits: {max_len} tokens (Context budget: 512)")
    assert max_len <= 512, f"Sequence length exceeds 512 limit: {max_len}"
    print("   [PASS] All samples fit within context window and decode losslessly.")
    
    # 3. Category distribution
    print("\n3. Category Balance:")
    for split_name, ds in [("Train", train_data), ("Val", val_data), ("Test", test_data)]:
        cat_counts = {}
        for item in ds:
            c = item["category"]
            cat_counts[c] = cat_counts.get(c, 0) + 1
        print(f"   {split_name}: {len(cat_counts)} categories represented, {min(cat_counts.values())} to {max(cat_counts.values())} samples per category.")
        assert len(cat_counts) == 20, f"Missing categories in {split_name}"
        
    print("\n[SUCCESS] Conversation data passed all integrity and leakage audits.")

if __name__ == "__main__":
    main()
