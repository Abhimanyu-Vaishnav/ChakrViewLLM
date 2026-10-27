"""
Data generator for Experiment 31: Capability Generalization & Leakage-Resistant Evaluation.
Produces disjoint splits:
- train.jsonl (120 examples)
- val.jsonl (40 examples)
- test_group_A_familiar.jsonl (50 examples)
- test_group_B_unseen_tokens.jsonl (50 examples)
- test_group_C_permutations.jsonl (50 examples)
- test_group_D_distractors.jsonl (50 examples)
- test_group_E_counterfactuals.jsonl (50 pairs = 100 examples)
- test_group_F_negative_abstain.jsonl (50 examples)
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

# Tokenizer verification
TOK_DIR = REPO_ROOT / "data" / "experiments" / "vocab_4096"
tokenizer, _ = load_tokenizer_artifacts(TOK_DIR)

# Define token pools
TRAIN_VALUES = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"]
UNSEEN_VALUES = ["M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
                 "1", "2", "3", "4", "5", "6", "7", "8", "9", "!"]
ABSTAIN_TOKEN = "?"

# Verify all values encode to single token
for val in TRAIN_VALUES + UNSEEN_VALUES + [ABSTAIN_TOKEN]:
    ids = tokenizer.encode(val)
    assert len(ids) == 1, f"Token '{val}' encodes to multiple ids: {ids}"

# Assert disjointness
assert set(TRAIN_VALUES).isdisjoint(set(UNSEEN_VALUES)), "Train and unseen values intersect!"

KEYS = ["alpha", "beta", "gamma", "delta"]

def format_instance(bindings, query_key, answer):
    # bindings: list of (key, value)
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

    # 1. Train set (120 instances, familiar tokens only, 2-3 bindings)
    train_data = []
    for _ in range(120):
        n_keys = rng.choice([2, 3])
        chosen_keys = rng.sample(KEYS, n_keys)
        chosen_vals = rng.sample(TRAIN_VALUES, n_keys)
        bindings = list(zip(chosen_keys, chosen_vals))
        target_pair = rng.choice(bindings)
        train_data.append(format_instance(bindings, target_pair[0], target_pair[1]))

    # 2. Validation set (40 instances, familiar tokens, disjoint combinations)
    val_data = []
    for _ in range(40):
        n_keys = rng.choice([2, 3])
        chosen_keys = rng.sample(KEYS, n_keys)
        chosen_vals = rng.sample(TRAIN_VALUES, n_keys)
        bindings = list(zip(chosen_keys, chosen_vals))
        target_pair = rng.choice(bindings)
        val_data.append(format_instance(bindings, target_pair[0], target_pair[1]))

    # 3. Test Group A - Familiar distribution (50 instances)
    group_a = []
    for _ in range(50):
        n_keys = rng.choice([2, 3])
        chosen_keys = rng.sample(KEYS, n_keys)
        chosen_vals = rng.sample(TRAIN_VALUES, n_keys)
        bindings = list(zip(chosen_keys, chosen_vals))
        target_pair = rng.choice(bindings)
        group_a.append(format_instance(bindings, target_pair[0], target_pair[1]))

    # 4. Test Group B - Unseen Token Identities (50 instances)
    group_b = []
    for _ in range(50):
        n_keys = rng.choice([2, 3])
        chosen_keys = rng.sample(KEYS, n_keys)
        # Target value MUST be strictly from UNSEEN_VALUES
        chosen_vals = rng.sample(UNSEEN_VALUES, n_keys)
        bindings = list(zip(chosen_keys, chosen_vals))
        target_pair = rng.choice(bindings)
        group_b.append(format_instance(bindings, target_pair[0], target_pair[1]))

    # 5. Test Group C - Unseen Slot Permutations (50 instances)
    group_c = []
    for _ in range(50):
        # 3 keys, but slot order specifically varied so query target is strictly in first or last position
        chosen_keys = rng.sample(KEYS, 3)
        chosen_vals = rng.sample(TRAIN_VALUES, 3)
        bindings = list(zip(chosen_keys, chosen_vals))
        target_idx = rng.choice([0, 2]) # strictly edge slots, testing positional bias
        target_pair = bindings[target_idx]
        group_c.append(format_instance(bindings, target_pair[0], target_pair[1]))

    # 6. Test Group D - Distractor Complexity (50 instances)
    group_d = []
    for _ in range(50):
        # 4 keys present (maximum distraction density)
        chosen_keys = list(KEYS) # all 4 keys
        rng.shuffle(chosen_keys)
        chosen_vals = rng.sample(TRAIN_VALUES + UNSEEN_VALUES, 4)
        bindings = list(zip(chosen_keys, chosen_vals))
        target_pair = rng.choice(bindings)
        group_d.append(format_instance(bindings, target_pair[0], target_pair[1]))

    # 7. Test Group E - Counterfactual Pairs (50 pairs = 100 instances)
    group_e = []
    for pair_idx in range(50):
        chosen_keys = rng.sample(KEYS, 3)
        v1, v2, v3 = rng.sample(TRAIN_VALUES + UNSEEN_VALUES, 3)
        # Alternate value for target
        remaining = [v for v in (TRAIN_VALUES + UNSEEN_VALUES) if v not in [v1, v2, v3]]
        v_alt = rng.choice(remaining)
        target_k = chosen_keys[1] # middle key

        b_orig = [(chosen_keys[0], v1), (target_k, v2), (chosen_keys[2], v3)]
        b_cf = [(chosen_keys[0], v1), (target_k, v_alt), (chosen_keys[2], v3)]

        inst_orig = format_instance(b_orig, target_k, v2)
        inst_orig["pair_id"] = pair_idx
        inst_orig["variant"] = "original"

        inst_cf = format_instance(b_cf, target_k, v_alt)
        inst_cf["pair_id"] = pair_idx
        inst_cf["variant"] = "counterfactual"

        group_e.extend([inst_orig, inst_cf])

    # 8. Test Group F - Negative Controls / Abstention (50 instances)
    group_f = []
    for _ in range(50):
        # Context contains only 2 keys, query asks for a 3rd absent key
        present_keys = rng.sample(KEYS, 2)
        absent_keys = [k for k in KEYS if k not in present_keys]
        query_k = rng.choice(absent_keys)
        chosen_vals = rng.sample(TRAIN_VALUES, 2)
        bindings = list(zip(present_keys, chosen_vals))
        # Correct answer is '?'
        group_f.append(format_instance(bindings, query_k, ABSTAIN_TOKEN))

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
        print(f"Generated {filename}: {len(records)} examples, SHA256: {hasher.hexdigest()[:12]}...")

    manifest_path = DATA_DIR / "dataset_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Dataset manifest written to {manifest_path}")

if __name__ == "__main__":
    main()
