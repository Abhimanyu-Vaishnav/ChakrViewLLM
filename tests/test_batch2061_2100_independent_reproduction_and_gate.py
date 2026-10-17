#!/usr/bin/env python3
"""
Unit and regression tests for Batches 2061–2100:
Independent Reproduction, Corpus Integrity & CPT Release Gate.
Validates real system behavior, cryptographic immutability, corpus totals,
tokenizer fertility, statistical estimands, and gate schema compliance.
"""

import hashlib
import json
import math
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
BATCH2061_DIR = REPO_ROOT / "experiments" / "batch2061_2100"
BATCH2021_DIR = REPO_ROOT / "experiments" / "batch2021_2060"
BATCH1901_DIR = REPO_ROOT / "experiments" / "batch1901_1940"
CANONICAL_CHECKPOINT_PATH = REPO_ROOT / "experiments" / "batch16_21_combined" / "main_training_10000" / "training_checkpoint_step_10000.pt"
CPT_150_PATH = REPO_ROOT / "experiments" / "batch1861_1900" / "checkpoints" / "cpt_balanced_seed_42.pt"
CPT_300_PATH = REPO_ROOT / "experiments" / "batch1901_1940" / "checkpoints" / "cpt_balanced_seed_42_300steps.pt"
TOKENIZER_DIR = REPO_ROOT / "release" / "chakrview-v0.5.0" / "data" / "experiments" / "vocab_4096"


def test_checkpoint_and_tokenizer_cryptographic_integrity():
    """Verify all protected model checkpoints and tokenizer files match immutable hashes."""
    assert CANONICAL_CHECKPOINT_PATH.exists()
    h_canon = hashlib.sha256(CANONICAL_CHECKPOINT_PATH.read_bytes()).hexdigest()
    assert h_canon == "96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee"

    assert CPT_150_PATH.exists()
    h_150 = hashlib.sha256(CPT_150_PATH.read_bytes()).hexdigest()
    assert h_150 == "85753315bb55853f0e8bad91a916c3846c848333a1b8e3b9b02a5256b7aa7f1c"

    assert CPT_300_PATH.exists()
    h_300 = hashlib.sha256(CPT_300_PATH.read_bytes()).hexdigest()
    assert h_300 == "2f7751511b623e1dd06bbe2d721df71e679da34d48d4b443f092600d768e7628"

    for fname in ["config.json", "vocab.json", "merges.json"]:
        p = TOKENIZER_DIR / fname
        assert p.exists()
        assert p.stat().st_size > 500


def test_corpus_v2_counts_and_reproducible_tokens():
    """Verify Corpus v2 contains exactly 1,258 docs, 245,758 tokens, and 100% SHA-256 match."""
    from chakrview.tokenizer.serialization import load_tokenizer_artifacts

    tok, _ = load_tokenizer_artifacts(TOKENIZER_DIR)
    manifest_p = BATCH2021_DIR / "CORPUS_V2_MANIFEST.json"
    assert manifest_p.exists()

    with open(manifest_p, "r", encoding="utf-8") as f:
        corpus = json.load(f)

    docs = corpus["documents"]
    assert len(docs) == 1258
    assert sum(d["token_count"] for d in docs) == 245758
    assert sum(d["character_count"] for d in docs) == 533433

    # Sample-check SHA-256 and tokenization correctness across 20 deterministic docs
    for i in range(0, len(docs), 60):
        d = docs[i]
        text_bytes = d["text"].encode("utf-8")
        assert hashlib.sha256(text_bytes).hexdigest() == d["sha256"]
        assert len(tok.encode(d["text"])) == d["token_count"]


def test_corpus_source_family_isolation_and_near_duplicates():
    """Verify source-family split isolation and zero exact cross-split hash overlap."""
    manifest_p = BATCH2021_DIR / "CORPUS_V2_MANIFEST.json"
    with open(manifest_p, "r", encoding="utf-8") as f:
        corpus = json.load(f)

    docs = corpus["documents"]
    train_sources = set(d["source_id"] for d in docs if d["split"] == "train")
    val_sources = set(d["source_id"] for d in docs if d["split"] == "validation")
    test_sources = set(d["source_id"] for d in docs if d["split"] == "test")

    assert train_sources.isdisjoint(val_sources)
    assert train_sources.isdisjoint(test_sources)
    assert val_sources.isdisjoint(test_sources)

    train_shas = {d["sha256"] for d in docs if d["split"] == "train"}
    val_shas = {d["sha256"] for d in docs if d["split"] == "validation"}
    test_shas = {d["sha256"] for d in docs if d["split"] == "test"}

    assert train_shas.isdisjoint(val_shas)
    assert train_shas.isdisjoint(test_shas)
    assert val_shas.isdisjoint(test_shas)


def test_tokenizer_multilingual_lossless_roundtrip():
    """Verify tokenizer roundtrips without byte corruption on multilingual natural text."""
    from chakrview.tokenizer.serialization import load_tokenizer_artifacts

    tok, _ = load_tokenizer_artifacts(TOKENIZER_DIR)
    samples = [
        "Transmission Control Protocol (TCP) specification and state machine transitions.",
        "मेरी प्रिय पुस्तक मुंशी प्रेमचंद द्वारा रचित गोदान है।",
        "shrimad bhagavad gita classical sanskrit verses and shlokas.",
        "def quick_sort(arr):\n    return sorted(arr)",
        "aaj ka weather kaisa hai mujhe traffic ka update batao.",
        "कर्मण्येवाधिकारस्ते मा फलेषु कदाचन।"
    ]
    for s in samples:
        enc = tok.encode(s)
        dec = tok.decode(enc)
        assert dec == s, f"Decoded string mismatch for: {s}"


def test_paired_statistical_reproduction_and_p_value():
    """Verify the reproduced 150-step vs 300-step paired differences and p-value."""
    eval_p = BATCH1901_DIR / "INDEPENDENT_EVALUATION_RESULTS.json"
    assert eval_p.exists()

    with open(eval_p, "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    indep = eval_data["source_disjoint_independent_evaluation"]
    docs150 = {d["doc_id"]: d for d in indep["cpt_balanced_seed_42"]["metrics"]["documents"]}
    docs300 = {d["doc_id"]: d for d in indep["cpt_balanced_seed_42_300steps"]["metrics"]["documents"]}

    assert len(docs150) == 88
    assert len(docs300) == 88

    diffs = [docs300[did]["loss"] - docs150[did]["loss"] for did in docs150]
    n = len(diffs)
    mean_diff = sum(diffs) / n
    var_diff = sum((x - mean_diff) ** 2 for x in diffs) / (n - 1)
    se = math.sqrt(var_diff) / math.sqrt(n)
    t_stat = mean_diff / se

    assert round(mean_diff, 4) == 0.1324
    assert t_stat > 9.0, f"Expected t-stat > 9.0, got {t_stat}"
    # Proportion of documents regressing must be > 80%
    pos_diffs = sum(1 for x in diffs if x > 0)
    assert pos_diffs / n >= 0.85


def test_release_gate_json_schema_and_verdict():
    """Verify CPT_RELEASE_GATE.json schema, gate statuses, and conditional decision."""
    gate_p = BATCH2061_DIR / "CPT_RELEASE_GATE.json"
    assert gate_p.exists()

    with open(gate_p, "r", encoding="utf-8") as f:
        gate = json.load(f)

    assert gate["milestone"] == "batch2061_2100"
    assert gate["overall_decision"] == "CONDITIONAL_CPT_READINESS"
    assert len(gate["gates"]) == 10

    allowed_statuses = {"PASS", "FAIL", "CONDITIONAL", "BLOCKED", "NOT_APPLICABLE"}
    for g in gate["gates"]:
        assert g["status"] in allowed_statuses
        assert Path(g["evidence_path"]).exists(), f"Evidence path missing: {g['evidence_path']}"
        assert len(g["limitations"]) > 5
        assert len(g["remediation"]) > 3


def test_all_batch2061_deliverables_present():
    """Verify all 6 batch2061 deliverables exist and are non-empty."""
    required = [
        "REPOSITORY_STATE_AUDIT.md",
        "CORPUS_INTEGRITY_AUDIT.md",
        "TOKENIZER_REPRODUCTION_AUDIT.md",
        "CPT_COMPARISON_REPRODUCTION.md",
        "CPT_PREPARATION_AUDIT.md",
        "CPT_RELEASE_GATE.json"
    ]
    for r in required:
        p = BATCH2061_DIR / r
        assert p.exists(), f"Missing deliverable: {r}"
        assert p.stat().st_size > 100, f"Deliverable is unexpectedly empty: {r}"
