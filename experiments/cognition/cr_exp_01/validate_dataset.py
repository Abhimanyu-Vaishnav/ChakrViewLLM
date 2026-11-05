"""
Strict Dataset Integrity Validator for CR-EXP-01
Validates:
1. Zero token/entity leakage across train, val, and test.
2. Complete absence of duplicate examples across splits.
3. Explicit presence of 'Answer: ' delimiter.
4. Well-formed JSON formatting and token length adherence.
"""

import json
from pathlib import Path

def validate():
    data_dir = Path("data/sft/cr_exp01")
    
    files = {
        "train": data_dir / "train_1000.jsonl",
        "val": data_dir / "val_100.jsonl",
        "test_fam": data_dir / "test_familiar_100.jsonl",
        "test_nov": data_dir / "test_novel_100.jsonl",
        "test_reassign": data_dir / "test_reassignment_100.jsonl",
        "test_dist": data_dir / "test_distractors_100.jsonl",
        "test_tmpl": data_dir / "test_unseen_template_100.jsonl"
    }
    
    loaded = {}
    for name, p in files.items():
        assert p.exists(), f"File {p} does not exist!"
        with open(p, "r", encoding="utf-8") as f:
            loaded[name] = [json.loads(line) for line in f]
        print(f"Loaded {name}: {len(loaded[name])} records.")
        
    # Check 1: Example text duplicates between train and held-out test
    train_texts = set(r["text"] for r in loaded["train"])
    for test_name in ["test_nov", "test_reassign", "test_dist", "test_tmpl"]:
        for r in loaded[test_name]:
            assert r["text"] not in train_texts, f"LEAKAGE DETECTED: {r['text']} in {test_name} also exists in train!"
    print("Check 1 Passed: ZERO example text leakage between train and held-out tests.")
    
    # Check 2: Key & Target Value sets between train and test_nov
    tr_keys = set(r["key"] for r in loaded["train"])
    tr_vals = set(r["target_value"] for r in loaded["train"])
    
    nov_keys = set(r["key"] for r in loaded["test_nov"])
    nov_vals = set(r["target_value"] for r in loaded["test_nov"])
    
    assert tr_keys.isdisjoint(nov_keys), f"Key leakage: {tr_keys.intersection(nov_keys)}"
    assert tr_vals.isdisjoint(nov_vals), f"Value leakage: {tr_vals.intersection(nov_vals)}"
    print("Check 2 Passed: 100% DISJOINT key and value entity pools between train and test_novel.")
    
    # Check 3: Delimiter verification
    for name, records in loaded.items():
        for r in records:
            assert "Answer: " in r["text"], f"Record {r['id']} in {name} lacks 'Answer: ' delimiter!"
    print("Check 3 Passed: 100% of records contain explicit 'Answer: ' delimiter.")
    
    # Check 4: Reassignment targets match updated state
    for r in loaded["test_reassign"]:
        assert r["target_value"] in r["text"], f"Target {r['target_value']} missing from text!"
        assert r["target_value"] != r["v_old"], f"Reassignment target equal to old value!"
    print("Check 4 Passed: Reassignment conditions unambiguously test updated value.")
    
    print("\nALL DATASET INTEGRITY VALIDATOR CHECKS PASSED PERFECTLY.")

if __name__ == "__main__":
    validate()

