"""
prepare_exp09_dataset.py
ChakrView Experiment 09 — Step 3: Freeze a Fair Comparison Dataset

Builds a deterministic dataset with:
- 707 verified clean single-token words.
- Partition:
  - Held-out token pool (Test B): 80 tokens (never in Train targets).
  - Broad training target pool: 450 tokens (expanded from 120 in Exp 08).
  - Train split (2,500 examples):
    * Stage A (800 ex): direct copy
    * Stage B (900 ex): 3-distractor lookup with Alpha, Beta, Gamma keys
    * Stage C (800 ex): 4 syntactic variations
  - Validation split (300 examples): for checkpoint selection only.
  - Test A (300 examples): Familiar tokens in novel contexts / distractor combinations.
  - Test B (300 examples): Held-out tokens in novel contexts / distractor combinations.

Each record includes:
- prompt
- target_value
- target_token_id
- target_position (item index: 0, 1, or 2)
- source_token_index (exact token position in prompt sequence for pointer supervision)
- candidate_token_indices (all candidate item positions in the prompt)
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chakrview.tokenizer.serialization import load_tokenizer_artifacts

EXP09_DIR = Path(__file__).resolve().parent
DATA_DIR = EXP09_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
TOKENIZER_DIR = REPO_ROOT / "data" / "experiments" / "vocab_4096"

SEED = 999

def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def get_clean_single_tokens(tokenizer) -> List[Tuple[int, str]]:
    valid_tokens = []
    for tok_id, tok_str in tokenizer.vocab.items():
        if tok_id in [0, 1, 2]:
            continue
        try:
            decoded = tokenizer.decode([tok_id])
            s = decoded.strip()
            if len(s) >= 3 and len(s) <= 10 and s.isalpha():
                encoded = tokenizer.encode(s)
                if len(encoded) == 1 and encoded[0] == tok_id:
                    valid_tokens.append((tok_id, s))
        except Exception:
            continue
    return valid_tokens

def locate_token_in_sequence(full_prompt_toks: List[int], target_tok_id: int) -> int:
    """Finds the position in [0] + full_prompt_toks corresponding to target_tok_id."""
    seq = [0] + full_prompt_toks
    matches = [i for i, t in enumerate(seq) if t == target_tok_id]
    if matches:
        return matches[-1]
    # Fallback to last token of prompt
    return len(seq) - 1

def build_splits():
    print("=" * 70)
    print("EXP 09 STEP 3: DATASET GENERATION & LEAKAGE AUDIT")
    print("=" * 70)

    tokenizer, _ = load_tokenizer_artifacts(TOKENIZER_DIR)
    all_clean = get_clean_single_tokens(tokenizer)
    print(f"Total clean single-tokens: {len(all_clean)}")

    rng = random.Random(SEED)
    shuffled = list(all_clean)
    rng.shuffle(shuffled)

    # 80 held-out tokens strictly for Test B
    test_b_pool = shuffled[:80]
    # 450 tokens for training targets (broad-vocabulary arm)
    train_target_pool = shuffled[80:530]
    # All non-test-b tokens for distractors
    distractor_pool = shuffled[80:]

    print(f"Train/Val/TestA Target Pool: {len(train_target_pool)} tokens")
    print(f"Test B Held-Out Target Pool:  {len(test_b_pool)} tokens")

    def make_dataset(tok_pool, dist_pool, n_a, n_b, n_c, base_seed):
        prng = random.Random(base_seed)
        recs = []

        # Stage A: Single target
        for _ in range(n_a):
            target_tok = prng.choice(tok_pool)
            prompt = f"Target value: {target_tok[1]}\nCopy: "
            p_toks = tokenizer.encode(prompt)
            src_idx = locate_token_in_sequence(p_toks, target_tok[0])
            recs.append({
                "stage": "Stage_A",
                "template_type": "standard_target",
                "prompt": prompt,
                "target_key": "target",
                "target_value": target_tok[1],
                "target_token_id": target_tok[0],
                "distractors": [],
                "target_position": 0,
                "source_token_index": src_idx,
                "candidate_token_indices": [src_idx]
            })

        # Stage B: Multi-item lookup with distractors
        for _ in range(n_b):
            target_tok = prng.choice(tok_pool)
            # Pick 2 distinct distractors from dist_pool
            cands = [d for d in dist_pool if d[0] != target_tok[0]]
            d1, d2 = prng.sample(cands, 2)
            pos = prng.randint(0, 2)
            items = [d1, d2]
            items.insert(pos, target_tok)
            labels = ["Alpha", "Beta", "Gamma"]
            target_label = labels[pos]

            prompt = (
                f"Items: {labels[0]}={items[0][1]}, {labels[1]}={items[1][1]}, {labels[2]}={items[2][1]}\n"
                f"Copy {target_label}: "
            )
            p_toks = tokenizer.encode(prompt)
            item_tok_ids = [it[0] for it in items]
            seq = [0] + p_toks
            cand_indices = []
            for it_id in item_tok_ids:
                for idx, t in enumerate(seq):
                    if t == it_id and idx not in cand_indices:
                        cand_indices.append(idx)
                        break

            src_idx = cand_indices[pos] if len(cand_indices) == 3 else locate_token_in_sequence(p_toks, target_tok[0])
            recs.append({
                "stage": "Stage_B",
                "template_type": "multi_item_lookup",
                "prompt": prompt,
                "target_key": target_label,
                "target_value": target_tok[1],
                "target_token_id": target_tok[0],
                "distractors": [it[1] for it in items if it[0] != target_tok[0]],
                "target_position": pos,
                "source_token_index": src_idx,
                "candidate_token_indices": cand_indices
            })

        # Stage C: Varied syntactic templates
        templates = [
            ("key_val", "Data: key={target}, ref={distractor}\nExtract key: "),
            ("bracket", "Input sequence: [{distractor}, {target}]\nSelect second item: "),
            ("query_assign", "Assignment: value is {target}; ignore {distractor}.\nOutput: "),
            ("colon_tag", "Record: primary:{target} secondary:{distractor}\nPrimary: ")
        ]
        for _ in range(n_c):
            t_name, t_fmt = prng.choice(templates)
            target_tok = prng.choice(tok_pool)
            cands = [d for d in dist_pool if d[0] != target_tok[0]]
            dist_tok = prng.choice(cands)
            prompt = t_fmt.format(target=target_tok[1], distractor=dist_tok[1])
            p_toks = tokenizer.encode(prompt)
            target_pos = 0 if "{target}" in t_fmt[:t_fmt.index("{distractor}")] else 1

            seq = [0] + p_toks
            cand_indices = []
            for it_id in [target_tok[0], dist_tok[0]]:
                for idx, t in enumerate(seq):
                    if t == it_id and idx not in cand_indices:
                        cand_indices.append(idx)
                        break

            src_idx = cand_indices[0] if cand_indices else locate_token_in_sequence(p_toks, target_tok[0])
            recs.append({
                "stage": "Stage_C",
                "template_type": t_name,
                "prompt": prompt,
                "target_key": "target",
                "target_value": target_tok[1],
                "target_token_id": target_tok[0],
                "distractors": [dist_tok[1]],
                "target_position": target_pos,
                "source_token_index": src_idx,
                "candidate_token_indices": cand_indices
            })

        prng.shuffle(recs)
        return recs

    # Generate splits
    # Train: 800 A + 900 B + 800 C = 2,500 examples
    train_recs = make_dataset(train_target_pool, distractor_pool, 800, 900, 800, base_seed=1001)
    val_recs = make_dataset(train_target_pool, distractor_pool, 100, 100, 100, base_seed=2001)
    test_a_recs = make_dataset(train_target_pool, distractor_pool, 100, 100, 100, base_seed=3001)
    test_b_recs = make_dataset(test_b_pool, test_b_pool, 100, 100, 100, base_seed=4001)

    # Filter any exact prompt overlap
    train_prompts = set(r["prompt"] for r in train_recs)
    val_recs = [r for r in val_recs if r["prompt"] not in train_prompts]
    test_a_recs = [r for r in test_a_recs if r["prompt"] not in train_prompts]
    test_b_recs = [r for r in test_b_recs if r["prompt"] not in train_prompts]

    # Verify token partitions
    train_target_ids = set(r["target_token_id"] for r in train_recs)
    test_a_target_ids = set(r["target_token_id"] for r in test_a_recs)
    test_b_target_ids = set(r["target_token_id"] for r in test_b_recs)

    assert test_a_target_ids.issubset(train_target_ids), "Test A targets not in train!"
    assert train_target_ids.isdisjoint(test_b_target_ids), "Test B targets overlap train!"
    print("  [PASS] Zero exact prompt overlap and strict token partition verified.")

    # Save splits
    train_path = DATA_DIR / "train.jsonl"
    val_path = DATA_DIR / "val.jsonl"
    test_a_path = DATA_DIR / "test_a_familiar.jsonl"
    test_b_path = DATA_DIR / "test_b_held_out.jsonl"

    for p, data in [(train_path, train_recs), (val_path, val_recs), (test_a_path, test_a_recs), (test_b_path, test_b_recs)]:
        with open(p, "w", encoding="utf-8") as f:
            for r in data:
                f.write(json.dumps(r) + "\n")

    manifest = {
        "seed": SEED,
        "clean_tokens_total": len(all_clean),
        "held_out_token_pool_size": len(test_b_pool),
        "train_target_pool_size": len(train_target_pool),
        "splits": {
            "train": {"count": len(train_recs), "unique_targets": len(train_target_ids), "sha256": compute_sha256(train_path)},
            "val": {"count": len(val_recs), "unique_targets": len(set(r["target_token_id"] for r in val_recs)), "sha256": compute_sha256(val_path)},
            "test_a": {"count": len(test_a_recs), "unique_targets": len(test_a_target_ids), "sha256": compute_sha256(test_a_path)},
            "test_b": {"count": len(test_b_recs), "unique_targets": len(test_b_target_ids), "sha256": compute_sha256(test_b_path)}
        }
    }
    with open(DATA_DIR / "dataset_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nFinal Split Sizes:")
    print(f"  Train:  {len(train_recs)} (Targets: {len(train_target_ids)})")
    print(f"  Val:    {len(val_recs)}")
    print(f"  Test A: {len(test_a_recs)} (Familiar)")
    print(f"  Test B: {len(test_b_recs)} (Held-out)")
    print(f"Manifest written to {DATA_DIR / 'dataset_manifest.json'}")

if __name__ == "__main__":
    build_splits()
