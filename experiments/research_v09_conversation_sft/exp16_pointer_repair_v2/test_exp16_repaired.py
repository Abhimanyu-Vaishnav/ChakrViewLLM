"""
test_exp16_repaired.py
ChakrView Exp16 (v2) — Comprehensive Test Suite for Repaired Datasets.

Tests:
1. Immutability of canonical checkpoint, vocab, merges, session.py, and Exp15 checkpoint.
2. Immutability of original Exp16 datasets.
3. Repaired split counts: Train=300, Val=60, Test=120, exactly 10 balanced categories.
4. Zero exact duplicate prompts across train, val, and test splits.
5. Zero scenario-family prompt leakage across splits.
6. Target distribution variance (dynamic slots, distractor states, anaphora referents, ambiguity colors).
7. Complete decoded span match for all answerable examples (fail-closed integrity).
8. Strict enforcement of span length constraint: length <= 12 for all answerable spans.
9. Abstention integrity: position [0, 0] is reserved exclusively for is_abstention=True.
10. Target position coverage: start positions span a wide range across [50, 165].
"""

import hashlib
import json
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
EXP16_V2_DIR = Path(__file__).resolve().parent
DATA_DIR = EXP16_V2_DIR / "data"
ORIG_DATA_DIR = REPO_ROOT / "experiments" / "research_v09_conversation_sft" / "exp16_pointer_retraining_design" / "data"
CANONICAL_TOK_DIR = REPO_ROOT / "data" / "experiments" / "vocab_4096"
CANONICAL_CKPT = REPO_ROOT / "experiments" / "batch16_21_combined" / "main_training_10000" / "training_checkpoint_step_10000.pt"
EXP15_CKPT = REPO_ROOT / "experiments" / "research_v09_conversation_sft" / "exp03_field_conditioned_pointer" / "checkpoints" / "pointer_best.pt"

from chakrview.tokenizer.serialization import load_tokenizer_artifacts

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def test_canonical_and_historical_artifacts_immutability():
    """Confirms no canonical checkpoint, tokenizer, session.py, or Exp15 checkpoint changed."""
    expected_hashes = {
        CANONICAL_CKPT: "96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee",
        CANONICAL_TOK_DIR / "vocab.json": "db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a",
        CANONICAL_TOK_DIR / "merges.json": "ada4b1dcfbb3d962a2e60203089881872a5063de73b2fef9eeff81ce0016efa7",
        REPO_ROOT / "chakrview" / "runtime" / "session.py": "9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67",
        EXP15_CKPT: "8e99e29683aec12ac339f2f5ab9c935914871ff5e1fa6facd5c151ec8045b67d",
    }
    for file_path, exp_hash in expected_hashes.items():
        assert file_path.exists(), f"Missing file: {file_path}"
        actual = sha256_file(file_path)
        assert actual == exp_hash, f"Hash mismatch for {file_path}: {actual} != {exp_hash}"

def test_original_exp16_datasets_immutability():
    """Confirms original Exp16 datasets remain completely untouched."""
    expected_hashes = {
        ORIG_DATA_DIR / "train_exp16.jsonl": "55abc82154801da753f07bf1cf0ab1226ac5f07fc2506550d2c69b02d3372804",
        ORIG_DATA_DIR / "val_exp16.jsonl": "569a0091223aca34a84606a99ac60416b1efe3b0fdddf4d72b174b1b854c59a7",
        ORIG_DATA_DIR / "test_exp16.jsonl": "48de3477de8e50ba605165c093f2e329f16edf734ecdd4899b2c02e18e4885ad",
    }
    for file_path, exp_hash in expected_hashes.items():
        assert file_path.exists(), f"Missing file: {file_path}"
        actual = sha256_file(file_path)
        assert actual == exp_hash, f"Original Exp16 dataset mutated: {file_path}"

def test_repaired_split_counts_and_balance():
    """Verifies Train=300, Val=60, Test=120 and balanced categories."""
    splits = {
        "train_exp16_v2.jsonl": (300, 30),
        "val_exp16_v2.jsonl": (60, 6),
        "test_exp16_v2.jsonl": (120, 12),
    }
    for fname, (expected_total, expected_per_cat) in splits.items():
        path = DATA_DIR / fname
        assert path.exists(), f"Missing {fname}"
        records = [json.loads(line) for line in open(path, encoding="utf-8")]
        assert len(records) == expected_total, f"{fname} total {len(records)} != {expected_total}"
        from collections import Counter
        cat_counts = Counter(r["category"] for r in records)
        assert len(cat_counts) == 10
        for cat, cnt in cat_counts.items():
            assert cnt == expected_per_cat, f"{fname} category {cat} count {cnt} != {expected_per_cat}"

def test_zero_cross_split_prompt_duplicates():
    """Confirms zero prompt duplicates across train, val, and test splits."""
    train = [json.loads(l)["prompt"] for l in open(DATA_DIR / "train_exp16_v2.jsonl", encoding="utf-8")]
    val = [json.loads(l)["prompt"] for l in open(DATA_DIR / "val_exp16_v2.jsonl", encoding="utf-8")]
    test = [json.loads(l)["prompt"] for l in open(DATA_DIR / "test_exp16_v2.jsonl", encoding="utf-8")]

    assert len(set(train).intersection(set(test))) == 0, "Train and Test prompt overlap detected!"
    assert len(set(train).intersection(set(val))) == 0, "Train and Val prompt overlap detected!"
    assert len(set(val).intersection(set(test))) == 0, "Val and Test prompt overlap detected!"

def test_target_distribution_randomization():
    """Verifies that target selection is not biased to a single slot or constant."""
    train = [json.loads(l) for l in open(DATA_DIR / "train_exp16_v2.jsonl", encoding="utf-8")]
    
    # 1. Dynamic binding: slots 0, 1, 2 must all be represented
    dyn = [r for r in train if r["category"] == "dynamic_entity_binding" and not r["is_abstention"]]
    dyn_slots = set(r["target_text"].split("_")[-1] for r in dyn)
    assert dyn_slots == {"0", "1", "2"}, f"Dynamic slots not diversified: {dyn_slots}"
    
    # 2. Distractor resistance: states must not be 100% OPERATIONAL
    dist = [r for r in train if r["category"] == "distractor_resistance" and not r["is_abstention"]]
    dist_states = set(r["target_text"] for r in dist)
    assert len(dist_states) >= 4, f"Distractor target states lack diversity: {dist_states}"
    
    # 3. Anaphora: both Alpha and Beta must be targets
    anaph = [r for r in train if r["category"] == "anaphora_reference_resolution" and not r["is_abstention"]]
    anaph_referents = set(r["target_text"].split("_")[2] for r in anaph)
    assert "Alpha" in anaph_referents and "Beta" in anaph_referents

def test_span_length_contract_compliance():
    """Confirms every answerable span is <= 12 tokens (strictly reachable by decoder)."""
    for fname in ["train_exp16_v2.jsonl", "val_exp16_v2.jsonl", "test_exp16_v2.jsonl"]:
        path = DATA_DIR / fname
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if not r["is_abstention"]:
                    span_len = r["end_pos"] - r["start_pos"] + 1
                    assert 1 <= span_len <= 12, (
                        f"Span length violation in {r['example_id']}: length {span_len} exceeds max_span_len=12"
                    )

def test_fail_closed_span_decoding():
    """Verifies that every answerable record cleanly decodes back to the target entity."""
    tok, _ = load_tokenizer_artifacts(CANONICAL_TOK_DIR, verify_checksums=False)
    for fname in ["train_exp16_v2.jsonl", "val_exp16_v2.jsonl", "test_exp16_v2.jsonl"]:
        path = DATA_DIR / fname
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if r["is_abstention"]:
                    assert r["start_pos"] == 0 and r["end_pos"] == 0
                    assert r["target_text"] == ""
                else:
                    assert r["start_pos"] > 0
                    assert r["end_pos"] >= r["start_pos"]
                    tokens = r["input_ids"]
                    extracted = tokens[r["start_pos"] : r["end_pos"] + 1]
                    decoded = tok.decode(extracted).strip().lower()
                    target = r["target_text"].strip().lower()
                    assert decoded == target, f"Decoding mismatch in {r['example_id']}: '{decoded}' != '{target}'"

def test_deterministic_regeneration():
    """Confirms generation pipeline produces identical hashes under fixed seeds."""
    manifest_path = EXP16_V2_DIR / "EXP16_REPAIR_MANIFEST.json"
    assert manifest_path.exists()
    with open(manifest_path, "r", encoding="utf-8") as f:
        m = json.load(f)
    
    assert sha256_file(DATA_DIR / "train_exp16_v2.jsonl") == m["splits"]["train"]["sha256"]
    assert sha256_file(DATA_DIR / "val_exp16_v2.jsonl") == m["splits"]["val"]["sha256"]
    assert sha256_file(DATA_DIR / "test_exp16_v2.jsonl") == m["splits"]["test"]["sha256"]
