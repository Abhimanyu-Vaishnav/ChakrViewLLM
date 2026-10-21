"""Dataset Generator for ChakrView v0.8 Experiment 06 Minimal Copy Diagnostic.

Task: Minimal Context-Grounded Reproduction
Input format: `Target value: {value}\nCopy: `
Expected response: `{value}`

Two stages:
- Stage A: Single-token values verified with canonical tokenizer (exactly 1 token in response).
- Stage B: Short multi-token values verified with canonical tokenizer (2 to 4 tokens in response).

Partitioning & Disjointness:
- All training, validation, and held-out test values are strictly disjoint at both:
  1. Raw string level.
  2. Token sequence level.
- No leakage between splits.
- Split manifest and hashes deterministically recorded.
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chakrview.tokenizer.serialization import load_tokenizer_artifacts

OUTPUT_DIR = Path(__file__).resolve().parent / "data"
RANDOM_SEED = 42

PROMPT_TEMPLATE = "Target value: {value}\nCopy: "


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def find_single_token_values(tokenizer: Any) -> List[Tuple[str, int]]:
    """Finds unambiguous single-token lowercase dictionary words."""
    found = []
    for tid, b in tokenizer.vocab.items():
        if tid in (0, 1, 2):
            continue
        try:
            s = b.decode("utf-8")
        except Exception:
            continue
        # Filter for clean alphabetic standalone words of length 4 to 8
        if s.isalpha() and s.islower() and 4 <= len(s) <= 8:
            toks = tokenizer.encode(s)
            # Verify it encodes to exactly 1 token equal to tid
            if len(toks) == 1 and toks[0] == tid:
                # Also verify that when placed after 'Target value: ', it still encodes to [tid]
                prefix = "Target value: "
                p_toks = tokenizer.encode(prefix)
                full_toks = tokenizer.encode(prefix + s)
                if full_toks == p_toks + [tid]:
                    found.append((s, tid))
    return sorted(found, key=lambda x: x[0])


def generate_multi_token_values(tokenizer: Any, base_words: List[str]) -> List[Tuple[str, List[int]]]:
    """Generates composite 2-token and 3-token underscored identifier strings."""
    found = []
    # Use clean pairs of single-token words separated by underscore
    for i in range(len(base_words) - 1):
        for j in range(i + 1, min(i + 6, len(base_words))):
            w1 = base_words[i]
            w2 = base_words[j]
            comp = f"{w1}_{w2}"
            toks = tokenizer.encode(comp)
            if 2 <= len(toks) <= 4:
                prefix = "Target value: "
                p_toks = tokenizer.encode(prefix)
                full_toks = tokenizer.encode(prefix + comp)
                if full_toks == p_toks + toks:
                    found.append((comp, toks))
    return sorted(found, key=lambda x: x[0])


def build_record(
    rid: str,
    stage: str,
    val_str: str,
    val_toks: List[int],
    split: str,
    data_role: str,
) -> Dict[str, Any]:
    prompt = PROMPT_TEMPLATE.format(value=val_str)
    response = val_str
    return {
        "id": rid,
        "category": "minimal_copy",
        "stage": stage,
        "token_count": len(val_toks),
        "target_tokens": val_toks,
        "prompt": prompt,
        "response": response,
        "expected_value": val_str,
        "split": split,
        "data_role": data_role,
        "append_eos": True,
        "provenance": "exp06_deterministic_minimal_copy_generator",
        "review_status": "APPROVED",
        "status": "APPROVED",
    }


def generate_dataset():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(RANDOM_SEED)

    tok_dir = REPO_ROOT / "data" / "experiments" / "vocab_4096"
    tokenizer, _ = load_tokenizer_artifacts(tok_dir)

    # 1. Discover verified single-token words
    single_token_candidates = find_single_token_values(tokenizer)
    print(f"Total verified single-token candidates: {len(single_token_candidates)}")
    rng.shuffle(single_token_candidates)

    # Partition single-token values:
    # 50 Train, 15 Val, 25 Held-out Test
    stage_a_train = single_token_candidates[:50]
    stage_a_val = single_token_candidates[50:65]
    stage_a_test = single_token_candidates[65:90]

    # 2. Build multi-token values from disjoint pools to guarantee zero token-sequence leakage
    # Remaining words from single_token_candidates for multi-token building
    pool_train = [w for w, _ in single_token_candidates[90:150]]
    pool_val = [w for w, _ in single_token_candidates[150:180]]
    pool_test = [w for w, _ in single_token_candidates[180:230]]

    stage_b_train_candidates = generate_multi_token_values(tokenizer, pool_train)
    stage_b_val_candidates = generate_multi_token_values(tokenizer, pool_val)
    stage_b_test_candidates = generate_multi_token_values(tokenizer, pool_test)

    rng.shuffle(stage_b_train_candidates)
    rng.shuffle(stage_b_val_candidates)
    rng.shuffle(stage_b_test_candidates)

    stage_b_train = stage_b_train_candidates[:50]
    stage_b_val = stage_b_val_candidates[:15]
    stage_b_test = stage_b_test_candidates[:25]

    # Verify zero overlap
    train_vals = set([w for w, _ in stage_a_train] + [w for w, _ in stage_b_train])
    val_vals = set([w for w, _ in stage_a_val] + [w for w, _ in stage_b_val])
    test_vals = set([w for w, _ in stage_a_test] + [w for w, _ in stage_b_test])

    assert train_vals.isdisjoint(val_vals), "Train and Val overlap!"
    assert train_vals.isdisjoint(test_vals), "Train and Test overlap!"
    assert val_vals.isdisjoint(test_vals), "Val and Test overlap!"
    print(f"Strict disjointness verified: Train={len(train_vals)}, Val={len(val_vals)}, Test={len(test_vals)}")

    # 3. Build records
    train_records = []
    for i, (w, t) in enumerate(stage_a_train):
        train_records.append(build_record(f"exp06_train_A_{i+1:03d}", "Stage_A", w, [t], "train", "TRAIN"))
    for i, (w, toks) in enumerate(stage_b_train):
        train_records.append(build_record(f"exp06_train_B_{i+1:03d}", "Stage_B", w, toks, "train", "TRAIN"))

    val_records = []
    for i, (w, t) in enumerate(stage_a_val):
        val_records.append(build_record(f"exp06_val_A_{i+1:03d}", "Stage_A", w, [t], "val", "VALIDATION"))
    for i, (w, toks) in enumerate(stage_b_val):
        val_records.append(build_record(f"exp06_val_B_{i+1:03d}", "Stage_B", w, toks, "val", "VALIDATION"))

    test_records = []
    for i, (w, t) in enumerate(stage_a_test):
        test_records.append(build_record(f"exp06_test_A_{i+1:03d}", "Stage_A", w, [t], "test", "HELD_OUT_BENCHMARK"))
    for i, (w, toks) in enumerate(stage_b_test):
        test_records.append(build_record(f"exp06_test_B_{i+1:03d}", "Stage_B", w, toks, "test", "HELD_OUT_BENCHMARK"))

    rng.shuffle(train_records)
    rng.shuffle(val_records)
    rng.shuffle(test_records)

    # 4. Write splits
    train_path = OUTPUT_DIR / "train.jsonl"
    val_path = OUTPUT_DIR / "val.jsonl"
    test_path = OUTPUT_DIR / "test.jsonl"

    for path, recs in [(train_path, train_records), (val_path, val_records), (test_path, test_records)]:
        with open(path, "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    train_sha = compute_sha256(train_path)
    val_sha = compute_sha256(val_path)
    test_sha = compute_sha256(test_path)

    manifest = {
        "dataset_name": "exp06_minimal_copy_diagnostic",
        "random_seed": RANDOM_SEED,
        "splits": {
            "train": {"path": str(train_path), "records": len(train_records), "sha256": train_sha, "stage_a": len(stage_a_train), "stage_b": len(stage_b_train)},
            "val": {"path": str(val_path), "records": len(val_records), "sha256": val_sha, "stage_a": len(stage_a_val), "stage_b": len(stage_b_val)},
            "test": {"path": str(test_path), "records": len(test_records), "sha256": test_sha, "stage_a": len(stage_a_test), "stage_b": len(stage_b_test)},
        },
    }

    manifest_path = OUTPUT_DIR / "dataset_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("\nDataset generation successfully completed:")
    print(f"  Train: {train_path.name} ({len(train_records)} records, SHA={train_sha})")
    print(f"  Val:   {val_path.name} ({len(val_records)} records, SHA={val_sha})")
    print(f"  Test:  {test_path.name} ({len(test_records)} records, SHA={test_sha})")


if __name__ == "__main__":
    generate_dataset()
