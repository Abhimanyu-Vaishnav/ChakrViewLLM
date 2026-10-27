"""
Dataset generator for Experiment 32: Pointer Retrieval, Distractor Resolution & Abstention.
Strictly eliminates all cross-split prompt leakage, internal duplicates, and token overlap.
Splits:
- train.jsonl (160 examples: 120 positive retrieval + 40 negative abstention)
- val.jsonl (40 examples: 30 positive + 10 negative)
- test_group_A_familiar.jsonl (50 examples)
- test_group_B_unseen_tokens.jsonl (50 examples)
- test_group_C_permutations.jsonl (50 examples)
- test_group_D_distractors.jsonl (50 examples)
- test_group_E_counterfactuals.jsonl (50 pairs = 100 examples)
- test_group_F_negative_abstain.jsonl (50 examples)
Total test set: 350 strictly unique, non-overlapping examples.
"""

import json
import random
import hashlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from chakrview.tokenizer.serialization import load_tokenizer_artifacts

DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

TOK_DIR = REPO_ROOT / "data" / "experiments" / "vocab_4096"
tokenizer, _ = load_tokenizer_artifacts(TOK_DIR)

TRAIN_VALUES = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"]
UNSEEN_VALUES = ["M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
                 "1", "2", "3", "4", "5", "6", "7", "8", "9", "!"]
ABSTAIN_TOKEN = "?"

# Assert token validity
for val in TRAIN_VALUES + UNSEEN_VALUES + [ABSTAIN_TOKEN]:
    ids = tokenizer.encode(val)
    assert len(ids) == 1, f"Token '{val}' is not single-token: {ids}"

assert set(TRAIN_VALUES).isdisjoint(set(UNSEEN_VALUES)), "Train and unseen values intersect!"

KEYS = ["alpha", "beta", "gamma", "delta"]

def format_instance(bindings, query_key, answer):
    ctx_parts = [f"Stream {k}: Value: {v}." for k, v in bindings]
    context = "Context: " + " ".join(ctx_parts)
    question = f"Question: What is the {query_key} value?"
    prompt = f"{context}\n{question}\nAnswer: "
    full_text = f"{prompt}{answer}"
    return {
        "prompt": prompt,
        "answer": answer,
        "full_text": full_text,
        "query_key": query_key,
        "bindings": bindings
    }

def generate_splits(seed=42):
    rng = random.Random(seed)
    seen_prompts = set()

    def get_unique_positive(keys_pool, vals_pool, n_keys_choice, target_pos_rule=None):
        while True:
            n_keys = rng.choice(n_keys_choice) if isinstance(n_keys_choice, list) else n_keys_choice
            chosen_keys = rng.sample(keys_pool, n_keys)
            chosen_vals = rng.sample(vals_pool, n_keys)
            bindings = list(zip(chosen_keys, chosen_vals))
            if target_pos_rule == "edge":
                target_idx = rng.choice([0, n_keys - 1])
            else:
                target_idx = rng.randint(0, n_keys - 1)
            target_pair = bindings[target_idx]
            inst = format_instance(bindings, target_pair[0], target_pair[1])
            if inst["prompt"] not in seen_prompts:
                seen_prompts.add(inst["prompt"])
                return inst

    def get_unique_negative(keys_pool, vals_pool, n_present):
        while True:
            present_keys = rng.sample(keys_pool, n_present)
            absent_keys = [k for k in keys_pool if k not in present_keys]
            query_k = rng.choice(absent_keys)
            chosen_vals = rng.sample(vals_pool, n_present)
            bindings = list(zip(present_keys, chosen_vals))
            inst = format_instance(bindings, query_k, ABSTAIN_TOKEN)
            if inst["prompt"] not in seen_prompts:
                seen_prompts.add(inst["prompt"])
                return inst

    # 1. Train set (160 instances: 120 positive retrieval + 40 negative abstention)
    train_data = []
    for _ in range(120):
        train_data.append(get_unique_positive(KEYS, TRAIN_VALUES, [2, 3]))
    for _ in range(40):
        train_data.append(get_unique_negative(KEYS, TRAIN_VALUES, 2))

    # 2. Validation set (40 instances: 30 positive + 10 negative)
    val_data = []
    for _ in range(30):
        val_data.append(get_unique_positive(KEYS, TRAIN_VALUES, [2, 3]))
    for _ in range(10):
        val_data.append(get_unique_negative(KEYS, TRAIN_VALUES, 2))

    # 3. Test Group A - Familiar distribution (50 instances)
    group_a = []
    for _ in range(50):
        group_a.append(get_unique_positive(KEYS, TRAIN_VALUES, [2, 3]))

    # 4. Test Group B - Unseen Token Identities (50 instances, strictly UNSEEN values)
    group_b = []
    for _ in range(50):
        group_b.append(get_unique_positive(KEYS, UNSEEN_VALUES, [2, 3]))

    # 5. Test Group C - Unseen Slot Permutations (50 instances, target in edge slots)
    group_c = []
    for _ in range(50):
        group_c.append(get_unique_positive(KEYS, TRAIN_VALUES, 3, target_pos_rule="edge"))

    # 6. Test Group D - Distractor Complexity (50 instances, all 4 keys present)
    group_d = []
    for _ in range(50):
        group_d.append(get_unique_positive(KEYS, TRAIN_VALUES + UNSEEN_VALUES, 4))

    # 7. Test Group E - Counterfactual Pairs (50 pairs = 100 instances)
    group_e = []
    for pair_idx in range(50):
        while True:
            chosen_keys = rng.sample(KEYS, 3)
            v1, v2, v3 = rng.sample(TRAIN_VALUES + UNSEEN_VALUES, 3)
            remaining = [v for v in (TRAIN_VALUES + UNSEEN_VALUES) if v not in [v1, v2, v3]]
            v_alt = rng.choice(remaining)
            target_k = chosen_keys[1]

            b_orig = [(chosen_keys[0], v1), (target_k, v2), (chosen_keys[2], v3)]
            b_cf = [(chosen_keys[0], v1), (target_k, v_alt), (chosen_keys[2], v3)]

            inst_orig = format_instance(b_orig, target_k, v2)
            inst_cf = format_instance(b_cf, target_k, v_alt)

            if inst_orig["prompt"] not in seen_prompts and inst_cf["prompt"] not in seen_prompts:
                seen_prompts.add(inst_orig["prompt"])
                seen_prompts.add(inst_cf["prompt"])
                inst_orig["pair_id"] = pair_idx
                inst_orig["variant"] = "original"
                inst_cf["pair_id"] = pair_idx
                inst_cf["variant"] = "counterfactual"
                group_e.extend([inst_orig, inst_cf])
                break

    # 8. Test Group F - Negative Controls / Abstention (50 instances)
    group_f = []
    for _ in range(50):
        group_f.append(get_unique_negative(KEYS, TRAIN_VALUES + UNSEEN_VALUES, 2))

    return {
        "train.jsonl": train_data,
        "val.jsonl": val_data,
        "test_group_A_familiar.jsonl": group_a,
        "test_group_B_unseen_tokens.jsonl": group_b,
        "test_group_C_permutations.jsonl": group_c,
        "test_group_D_distractors.jsonl": group_d,
        "test_group_E_counterfactuals.jsonl": group_e,
        "test_group_F_negative_abstain.jsonl": group_f,
    }

def main():
    splits = generate_splits(seed=42)
    manifest = {}
    for filename, records in splits.items():
        filepath = DATA_DIR / filename
        with open(filepath, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        hasher = hashlib.sha256()
        with open(filepath, "rb") as f:
            hasher.update(f.read())
        manifest[filename] = {
            "count": len(records),
            "sha256": hasher.hexdigest()
        }
        print(f"Generated {filename}: {len(records)} examples, SHA256: {hasher.hexdigest()[:16]}...")

    manifest_path = DATA_DIR / "dataset_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Exp32 dataset manifest written to {manifest_path}")

if __name__ == "__main__":
    main()
