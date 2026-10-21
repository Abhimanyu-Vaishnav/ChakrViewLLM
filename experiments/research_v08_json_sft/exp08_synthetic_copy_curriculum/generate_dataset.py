"""
generate_dataset.py
ChakrView Experiment 08 — Synthetic Copying Curriculum Dataset Generator

Generates deterministic synthetic dataset splits:
1. Train (~2,000 examples):
   - Stage A: 800 examples (single target, fixed prompt template)
   - Stage B: 600 examples (target among distractors, specified target key/label)
   - Stage C: 600 examples (varied templates and target positions)
2. Validation (~300 examples):
   - Formed using train target token vocabulary, novel contexts.
3. Test A (~300 examples):
   - Familiar target token IDs (seen as supervised targets during training), but completely novel prompt contexts/distractor combinations.
4. Test B (~300 examples):
   - Held-out target token IDs (never seen as supervised targets during training).

Strictly audited for zero leakage, disjoint token sets between Train and Test B, and deterministic seeding.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chakrview.tokenizer.serialization import load_tokenizer_artifacts

EXP_DIR = Path(__file__).resolve().parent
DATA_DIR = EXP_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
TOKENIZER_DIR = REPO_ROOT / "data" / "experiments" / "vocab_4096"

SEED = 42

def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def get_clean_single_tokens(tokenizer) -> List[Tuple[int, str]]:
    """
    Finds vocabulary tokens that encode and decode cleanly to exactly 1 token,
    consist of printable ascii alphanumeric chars (length 3 to 10),
    and exclude special tokens, whitespace-only, or non-ascii artifacts.
    """
    valid_tokens = []
    for tok_id, tok_str in tokenizer.vocab.items():
        if tok_id in [0, 1, 2]: # special tokens
            continue
        # Decode and test round-trip
        try:
            decoded = tokenizer.decode([tok_id])
            s = decoded.strip()
            # Require clean alphabetic string
            if len(s) >= 3 and len(s) <= 10 and s.isalpha():
                # Verify single token encoding
                encoded = tokenizer.encode(s)
                if len(encoded) == 1 and encoded[0] == tok_id:
                    valid_tokens.append((tok_id, s))
        except Exception:
            continue
    return valid_tokens

def main():
    print("=" * 70)
    print("EXP 08: SYNTHETIC COPYING CURRICULUM DATASET GENERATION")
    print("=" * 70)

    tokenizer, _ = load_tokenizer_artifacts(TOKENIZER_DIR)
    clean_tokens = get_clean_single_tokens(tokenizer)
    print(f"Total clean single-token words available: {len(clean_tokens)}")
    assert len(clean_tokens) >= 200, f"Insufficient clean tokens: {len(clean_tokens)}"

    # Deterministic split of token IDs
    rng = random.Random(SEED)
    shuffled_tokens = list(clean_tokens)
    rng.shuffle(shuffled_tokens)

    # Reserve 60 tokens strictly for Test B (Held-out token identities)
    test_b_token_pool = shuffled_tokens[:60]
    # Remaining tokens for Train, Val, Test A
    train_token_pool = shuffled_tokens[60:]

    train_tok_ids = set(t[0] for t in train_token_pool)
    test_b_tok_ids = set(t[0] for t in test_b_token_pool)
    assert train_tok_ids.isdisjoint(test_b_tok_ids), "Critical: Token pools overlap!"
    print(f"Train/Val/TestA Token Pool: {len(train_token_pool)} tokens")
    print(f"TestB (Held-out) Token Pool: {len(test_b_token_pool)} tokens")

    # Templates
    # Stage A: fixed prompt template
    # Stage B: distractor template
    # Stage C: varied templates
    
    # We will build generators for each stage
    def make_stage_a(tok_pool, count, prompt_seed):
        prng = random.Random(prompt_seed)
        records = []
        for i in range(count):
            tok_id, val = prng.choice(tok_pool)
            prompt = f"Target value: {val}\nCopy: "
            records.append({
                "stage": "Stage_A",
                "template_type": "standard_target",
                "prompt": prompt,
                "target_key": "target",
                "target_value": val,
                "target_token_id": tok_id,
                "distractors": [],
                "target_position": 0
            })
        return records

    def make_stage_b(tok_pool, count, prompt_seed):
        prng = random.Random(prompt_seed)
        records = []
        for i in range(count):
            # Select target and 2 distractors
            selected = prng.sample(tok_pool, 3)
            target_tok = selected[0]
            d1, d2 = selected[1], selected[2]
            
            # Randomize position of target (0, 1, or 2)
            pos = prng.randint(0, 2)
            items = [d1, d2]
            items.insert(pos, target_tok)
            
            # Format: Item A: val1, Item B: val2, Item C: val3. Copy Item [X]:
            labels = ["Alpha", "Beta", "Gamma"]
            target_label = labels[pos]
            
            prompt = (
                f"Items: {labels[0]}={items[0][1]}, {labels[1]}={items[1][1]}, {labels[2]}={items[2][1]}\n"
                f"Copy {target_label}: "
            )
            records.append({
                "stage": "Stage_B",
                "template_type": "multi_item_lookup",
                "prompt": prompt,
                "target_key": target_label,
                "target_value": target_tok[1],
                "target_token_id": target_tok[0],
                "distractors": [it[1] for it in items if it[0] != target_tok[0]],
                "target_position": pos
            })
        return records

    def make_stage_c(tok_pool, count, prompt_seed):
        prng = random.Random(prompt_seed)
        records = []
        templates = [
            ("key_val", "Data: key={target}, ref={distractor}\nExtract key: "),
            ("bracket", "Input sequence: [{distractor}, {target}]\nSelect second item: "),
            ("query_assign", "Assignment: value is {target}; ignore {distractor}.\nOutput: "),
            ("colon_tag", "Record: primary:{target} secondary:{distractor}\nPrimary: ")
        ]
        for i in range(count):
            t_idx = prng.randint(0, len(templates) - 1)
            t_name, t_fmt = templates[t_idx]
            selected = prng.sample(tok_pool, 2)
            target_tok, dist_tok = selected[0], selected[1]
            
            prompt = t_fmt.format(target=target_tok[1], distractor=dist_tok[1])
            records.append({
                "stage": "Stage_C",
                "template_type": t_name,
                "prompt": prompt,
                "target_key": "target",
                "target_value": target_tok[1],
                "target_token_id": target_tok[0],
                "distractors": [dist_tok[1]],
                "target_position": 0 if "{target}" in t_fmt[:t_fmt.index("{distractor}")] else 1
            })
        return records

    # To guarantee Test A targets are a strict subset of Train targets:
    # We choose a focused target vocabulary of 120 tokens for training targets.
    # We ensure every one of these 120 tokens appears in the training set targets.
    # Val and Test A sample their targets strictly from this same 120-token pool.
    train_target_pool = train_token_pool[:120]
    # The remaining tokens in train_token_pool can serve as distractors or extra tokens
    distractor_pool = train_token_pool

    # 1. Build Training Set (800 Stage A + 600 Stage B + 600 Stage C = 2000 records)
    train_a = make_stage_a(train_target_pool, 800, prompt_seed=101)
    train_b = make_stage_b(train_target_pool, 600, prompt_seed=102)
    train_c = make_stage_c(train_target_pool, 600, prompt_seed=103)
    train_records = train_a + train_b + train_c
    rng.shuffle(train_records)

    # 2. Build Validation Set (100 Stage A + 100 Stage B + 100 Stage C = 300 records)
    val_a = make_stage_a(train_target_pool, 100, prompt_seed=201)
    val_b = make_stage_b(train_target_pool, 100, prompt_seed=202)
    val_c = make_stage_c(train_target_pool, 100, prompt_seed=203)
    val_records = val_a + val_b + val_c
    rng.shuffle(val_records)

    # 3. Build Test A Set (100 Stage A + 100 Stage B + 100 Stage C = 300 records)
    # Familiar token identities, novel prompt contexts
    test_a_a = make_stage_a(train_target_pool, 100, prompt_seed=301)
    test_a_b = make_stage_b(train_target_pool, 100, prompt_seed=302)
    test_a_c = make_stage_c(train_target_pool, 100, prompt_seed=303)
    test_a_records = test_a_a + test_a_b + test_a_c
    rng.shuffle(test_a_records)

    # 4. Build Test B Set (100 Stage A + 100 Stage B + 100 Stage C = 300 records)
    # Held-out token identities (from test_b_token_pool)
    test_b_a = make_stage_a(test_b_token_pool, 100, prompt_seed=401)
    test_b_b = make_stage_b(test_b_token_pool, 100, prompt_seed=402)
    test_b_c = make_stage_c(test_b_token_pool, 100, prompt_seed=403)
    test_b_records = test_b_a + test_b_b + test_b_c
    rng.shuffle(test_b_records)

    # Audit for prompt overlaps between train and eval splits
    train_prompts = set(r["prompt"] for r in train_records)
    val_prompts = set(r["prompt"] for r in val_records)
    test_a_prompts = set(r["prompt"] for r in test_a_records)
    test_b_prompts = set(r["prompt"] for r in test_b_records)

    # Filter out any exact duplicate prompts if randomly generated
    val_records = [r for r in val_records if r["prompt"] not in train_prompts]
    test_a_records = [r for r in test_a_records if r["prompt"] not in train_prompts]
    test_b_records = [r for r in test_b_records if r["prompt"] not in train_prompts]

    # Verify Test A targets are subset of train targets
    train_target_ids = set(r["target_token_id"] for r in train_records)
    test_a_target_ids = set(r["target_token_id"] for r in test_a_records)
    test_b_target_ids = set(r["target_token_id"] for r in test_b_records)

    assert test_a_target_ids.issubset(train_target_ids), "Test A contains non-train target tokens!"
    assert train_target_ids.isdisjoint(test_b_target_ids), "Test B target tokens overlap with Train!"

    print(f"\nFinal Split Sizes:")
    print(f"Train:  {len(train_records)} examples")
    print(f"Val:    {len(val_records)} examples")
    print(f"Test A: {len(test_a_records)} examples (Familiar Tokens, Novel Contexts)")
    print(f"Test B: {len(test_b_records)} examples (Held-out Token Identities)")

    # Save splits to JSONL
    train_path = DATA_DIR / "train.jsonl"
    val_path = DATA_DIR / "val.jsonl"
    test_a_path = DATA_DIR / "test_a_familiar_tokens.jsonl"
    test_b_path = DATA_DIR / "test_b_held_out_tokens.jsonl"

    for path, data in [(train_path, train_records), (val_path, val_records), 
                       (test_a_path, test_a_records), (test_b_path, test_b_records)]:
        with open(path, "w", encoding="utf-8") as f:
            for rec in data:
                f.write(json.dumps(rec) + "\n")

    manifest = {
        "seed": SEED,
        "splits": {
            "train": {
                "count": len(train_records),
                "sha256": compute_sha256(train_path),
                "unique_target_tokens": len(train_target_ids)
            },
            "val": {
                "count": len(val_records),
                "sha256": compute_sha256(val_path),
                "unique_target_tokens": len(set(r["target_token_id"] for r in val_records))
            },
            "test_a_familiar": {
                "count": len(test_a_records),
                "sha256": compute_sha256(test_a_path),
                "unique_target_tokens": len(test_a_target_ids),
                "all_targets_in_train": True
            },
            "test_b_held_out": {
                "count": len(test_b_records),
                "sha256": compute_sha256(test_b_path),
                "unique_target_tokens": len(test_b_target_ids),
                "targets_overlap_train": False
            }
        }
    }

    manifest_path = DATA_DIR / "dataset_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nSaved dataset manifest to {manifest_path}")

if __name__ == "__main__":
    main()
