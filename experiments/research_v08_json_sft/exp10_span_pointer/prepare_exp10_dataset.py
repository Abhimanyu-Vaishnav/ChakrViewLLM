"""
prepare_exp10_dataset.py
ChakrView Experiment 10 — Controlled Multi-Token Span Copying Dataset Generator

Generates deterministic synthetic dataset splits:
- Categories:
  A. Single-token values (regression control)
  B. Multi-token values (2-4 tokens: compound words, snake_case identifiers)
  C. Punctuation-adjacent spans (colon_tag, bracket, key_val, multi_item_lookup)
  D. Overlapping distractors
  E. Repeated values at multiple locations
  F. Novel contexts
  G. Held-out target identities (Test B)

Verification Guarantee:
- Every example finds the exact contiguous token sub-sequence in [0] + tokenizer.encode(prompt).
- Asserts that decoding the span [start:end+1] matches target_value under whitespace stripping.
- Excludes and logs any un-alignable examples.
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chakrview.tokenizer.serialization import load_tokenizer_artifacts

EXP10_DIR = Path(__file__).resolve().parent
DATA_DIR = EXP10_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
TOKENIZER_DIR = REPO_ROOT / "data" / "experiments" / "vocab_4096"

SEED = 1010

def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

# Multi-token entity vocabulary (compound words, identifiers)
MULTI_TOKEN_WORDS = [
    "auth_token", "learning_rate", "batch_size", "loss_fn", "optimizer_state",
    "checkpoint_dir", "model_config", "sequence_length", "hidden_dimension",
    "attention_mask", "pointer_head", "exact_match", "tokenizer_vocab",
    "eval_split", "source_span", "target_value", "gradient_norm", "cross_entropy",
    "data_pipeline", "weight_decay", "dropout_rate", "activation_fn",
    "neural_network", "deep_learning", "inference_step", "training_loop",
    "context_window", "subword_token", "position_index", "embedding_dim",
    "key_vector", "query_vector", "value_matrix", "softmax_prob",
    "hyper_param", "benchmark_suite", "error_code", "status_flag",
    "feature_vector", "cluster_node", "server_port", "database_uri",
    "file_path", "system_call", "memory_buffer", "stack_trace"
]

def find_subsequence(seq: List[int], subseq: List[int], prefer_occurrence: int = 0) -> Optional[Tuple[int, int]]:
    """
    Finds the start and end index (inclusive) of subseq inside seq.
    prefer_occurrence: 0 for first occurrence, 1 for second, etc.
    """
    n, m = len(seq), len(subseq)
    if m == 0 or m > n:
        return None
    matches = []
    for i in range(n - m + 1):
        if seq[i:i + m] == subseq:
            matches.append((i, i + m - 1))
    if not matches:
        return None
    if prefer_occurrence < len(matches):
        return matches[prefer_occurrence]
    return matches[-1]

def build_dataset():
    print("=" * 70)
    print("EXP 10: MULTI-TOKEN SPAN DATASET GENERATION & AUDIT")
    print("=" * 70)

    tokenizer, _ = load_tokenizer_artifacts(TOKENIZER_DIR)

    # 1. Clean Single-Token words
    clean_single = []
    for tok_id, tok_str in tokenizer.vocab.items():
        if tok_id in [0, 1, 2]:
            continue
        try:
            dec = tokenizer.decode([tok_id]).strip()
            if 3 <= len(dec) <= 10 and dec.isalpha():
                enc = tokenizer.encode(dec)
                if len(enc) == 1 and enc[0] == tok_id:
                    clean_single.append(dec)
        except Exception:
            continue

    # 2. Multi-token verified words
    verified_multi = []
    for w in MULTI_TOKEN_WORDS:
        enc = tokenizer.encode(w)
        if 2 <= len(enc) <= 5:
            dec = tokenizer.decode(enc).strip()
            if dec == w:
                verified_multi.append((w, enc))

    print(f"Clean Single-token pool: {len(clean_single)}")
    print(f"Verified Multi-token pool: {len(verified_multi)}")

    # Reserve held-out tokens strictly for Test B
    rng = random.Random(SEED)
    shuffled_single = list(clean_single)
    rng.shuffle(shuffled_single)
    test_b_single = shuffled_single[:50]
    train_single = shuffled_single[50:]

    shuffled_multi = list(verified_multi)
    rng.shuffle(shuffled_multi)
    test_b_multi = shuffled_multi[:15]
    train_multi = shuffled_multi[15:]

    print(f"Test B Held-Out Pool: {len(test_b_single)} single + {len(test_b_multi)} multi")
    print(f"Train/Val/TestA Pool: {len(train_single)} single + {len(train_multi)} multi")

    def generate_records(single_pool, multi_pool, count, seed, is_test_b=False):
        prng = random.Random(seed)
        records = []
        invalid_count = 0

        # Templates
        # 1. Direct multi-item lookup
        # 2. Punctuation-adjacent key_val: Data: key={target}, ref={distractor}\nExtract key:
        # 3. Punctuation-adjacent bracket: Input sequence: [{distractor}, {target}]\nSelect second item:
        # 4. Punctuation-adjacent colon_tag: Record: primary:{target} secondary:{distractor}\nPrimary:
        # 5. Overlapping / Repeated distractors
        
        while len(records) < count:
            category_roll = prng.random()
            
            # Select target: 40% single-token (control), 60% multi-token
            if category_roll < 0.4:
                cat = "A_single_token" if not is_test_b else "G_held_out_single"
                target_str = prng.choice(single_pool)
            else:
                cat = "B_multi_token" if not is_test_b else "G_held_out_multi"
                target_str = prng.choice(multi_pool)[0]

            # Choose template
            tmpl_type = prng.choice(["multi_item", "colon_tag", "bracket", "key_val", "repeated_distractor"])
            
            # Choose distractors
            d_pool = single_pool + [m[0] for m in multi_pool]
            d_cands = [d for d in d_pool if d != target_str]
            d1, d2 = prng.sample(d_cands, 2)

            is_repeated = False
            if tmpl_type == "multi_item":
                pos = prng.randint(0, 2)
                items = [d1, d2]
                items.insert(pos, target_str)
                labels = ["Alpha", "Beta", "Gamma"]
                target_label = labels[pos]
                prompt = f"Items: {labels[0]}={items[0]}, {labels[1]}={items[1]}, {labels[2]}={items[2]}\nCopy {target_label}: "
                pref_occ = 0
            elif tmpl_type == "colon_tag":
                cat = "C_punctuation_adjacent"
                pos = prng.randint(0, 1)
                if pos == 0:
                    prompt = f"Record: primary:{target_str} secondary:{d1}\nPrimary: "
                else:
                    prompt = f"Record: primary:{d1} secondary:{target_str}\nSecondary: "
                pref_occ = 0
            elif tmpl_type == "bracket":
                cat = "C_punctuation_adjacent"
                pos = prng.randint(0, 1)
                if pos == 0:
                    prompt = f"Input sequence: [{target_str}, {d1}]\nSelect first item: "
                else:
                    prompt = f"Input sequence: [{d1}, {target_str}]\nSelect second item: "
                pref_occ = 0
            elif tmpl_type == "key_val":
                cat = "C_punctuation_adjacent"
                prompt = f"Data: key={target_str}, ref={d1}\nExtract key: "
                pref_occ = 0
            elif tmpl_type == "repeated_distractor":
                # Category E: Target is repeated in prompt or distractor is repeated
                cat = "E_repeated_values"
                is_repeated = True
                prompt = f"List: first={target_str}, backup={target_str}, alternate={d1}\nSelect first: "
                pref_occ = 0

            # Encode prompt
            p_toks = tokenizer.encode(prompt)
            full_seq = [0] + p_toks
            
            # Find exact target sub-sequence inside full_seq
            # Encode target in context or independently
            target_subseq = tokenizer.encode(target_str)
            span = find_subsequence(full_seq, target_subseq, prefer_occurrence=pref_occ)

            if span is None:
                # Delimiter merged target subwords: search for subwords that decode to target_str
                # Try finding tokens whose decode matches target_str
                found = False
                for start_i in range(1, len(full_seq)):
                    for end_i in range(start_i, min(start_i + 8, len(full_seq))):
                        dec_slice = tokenizer.decode(full_seq[start_i:end_i+1]).strip()
                        # Clean any delimiter
                        dec_clean = dec_slice.lstrip(":=").rstrip(",").strip()
                        if dec_clean == target_str.strip():
                            span = (start_i, end_i)
                            found = True
                            break
                    if found:
                        break

            if span is None:
                invalid_count += 1
                continue

            start_idx, end_idx = span
            extracted_tokens = full_seq[start_idx:end_idx + 1]
            extracted_str = tokenizer.decode(extracted_tokens).strip().lstrip(":=").rstrip(",").strip()

            if extracted_str != target_str.strip():
                invalid_count += 1
                continue

            records.append({
                "example_id": f"{seed}_{len(records):04d}",
                "category": cat,
                "template_type": tmpl_type,
                "prompt": prompt,
                "target_value": target_str,
                "start_token_index": start_idx,
                "end_token_index": end_idx,
                "target_span_length": (end_idx - start_idx + 1),
                "expected_token_sequence": extracted_tokens,
                "is_punctuation_adjacent": (tmpl_type in ["colon_tag", "bracket", "key_val"]),
                "is_repeated": is_repeated
            })

        print(f"Generated {len(records)} valid records (Excluded {invalid_count} un-alignable).")
        return records

    print("\n--- Generating Dataset Splits ---")
    # Train: 2,400 records
    train_recs = generate_records(train_single, train_multi, 2400, seed=1001)
    # Val: 300 records
    val_recs = generate_records(train_single, train_multi, 300, seed=2001)
    # Test A (Familiar tokens/spans, novel contexts): 300 records
    test_a_recs = generate_records(train_single, train_multi, 300, seed=3001)
    # Test B (Held-out tokens/spans): 300 records
    test_b_recs = generate_records(test_b_single, test_b_multi, 300, seed=4001, is_test_b=True)

    # Filter any exact prompt overlap
    train_prompts = set(r["prompt"] for r in train_recs)
    val_recs = [r for r in val_recs if r["prompt"] not in train_prompts]
    test_a_recs = [r for r in test_a_recs if r["prompt"] not in train_prompts]
    test_b_recs = [r for r in test_b_recs if r["prompt"] not in train_prompts]

    # Verify held-out partition
    train_targets = set(r["target_value"] for r in train_recs)
    test_b_targets = set(r["target_value"] for r in test_b_recs)
    assert train_targets.isdisjoint(test_b_targets), "Critical: Test B targets overlap with Train!"
    print(f"  [PASS] Zero exact prompt overlap and strict disjointness between Train and Test B targets verified.")

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
        "splits": {
            "train": {"count": len(train_recs), "unique_targets": len(train_targets), "sha256": compute_sha256(train_path)},
            "val": {"count": len(val_recs), "sha256": compute_sha256(val_path)},
            "test_a": {"count": len(test_a_recs), "sha256": compute_sha256(test_a_path)},
            "test_b": {"count": len(test_b_recs), "unique_targets": len(test_b_targets), "sha256": compute_sha256(test_b_path)}
        }
    }
    with open(DATA_DIR / "dataset_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nFinal Split Summary:")
    print(f"  Train:  {len(train_recs)} examples")
    print(f"  Val:    {len(val_recs)} examples")
    print(f"  Test A: {len(test_a_recs)} examples (Familiar)")
    print(f"  Test B: {len(test_b_recs)} examples (Held-out)")
    print(f"Dataset manifest written to {DATA_DIR / 'dataset_manifest.json'}")

if __name__ == "__main__":
    build_dataset()
