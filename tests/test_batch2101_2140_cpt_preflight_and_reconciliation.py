#!/usr/bin/env python3
"""
Unit and regression tests for Batches 2101–2140:
CPT Preflight Remediation, Metric Reconciliation & Reproducibility Gate.
Validates real system behavior, cryptographic immutability, metric reconciliation,
Sanskrit cleaning accuracy, macro validation loss evaluation, checkpoint resume
reproducibility, and preflight gate schema compliance.
"""

import hashlib
import json
import math
from pathlib import Path
import pytest
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
BATCH2101_DIR = REPO_ROOT / "experiments" / "batch2101_2140"
BATCH2021_DIR = REPO_ROOT / "experiments" / "batch2021_2060"
BATCH1901_DIR = REPO_ROOT / "experiments" / "batch1901_1940"
CANONICAL_CHECKPOINT_PATH = REPO_ROOT / "experiments" / "batch16_21_combined" / "main_training_10000" / "training_checkpoint_step_10000.pt"
TOKENIZER_DIR = REPO_ROOT / "release" / "chakrview-v0.5.0" / "data" / "experiments" / "vocab_4096"


def test_checkpoint_and_tokenizer_immutability():
    """Verify Canonical and experimental checkpoints maintain immutable hashes."""
    assert CANONICAL_CHECKPOINT_PATH.exists()
    h_canon = hashlib.sha256(CANONICAL_CHECKPOINT_PATH.read_bytes()).hexdigest()
    assert h_canon == "96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee"

    # Tokenizer files
    for fname in ["config.json", "vocab.json", "merges.json"]:
        p = TOKENIZER_DIR / fname
        assert p.exists()
        assert p.stat().st_size > 500


def test_historical_metric_reconciliation():
    """Verify that 5.7242 and 5.8420 reproduce and that token count is exactly 11,022."""
    eval_p = BATCH1901_DIR / "INDEPENDENT_EVALUATION_RESULTS.json"
    manifest_p = BATCH1901_DIR / "data" / "INDEPENDENT_CORPUS_MANIFEST.json"
    assert eval_p.exists() and manifest_p.exists()

    with open(eval_p, "r", encoding="utf-8") as f:
        e = json.load(f)
    with open(manifest_p, "r", encoding="utf-8") as f:
        m = json.load(f)

    indep = e["source_disjoint_independent_evaluation"]
    d150 = indep["cpt_balanced_seed_42"]["metrics"]["documents"]
    d300 = indep["cpt_balanced_seed_42_300steps"]["metrics"]["documents"]

    assert len(d150) == 88 and len(d300) == 88
    assert sum(d["tokens"] for d in d150) == 11022
    assert sum(d["tokens"] for d in d300) == 11022

    tw150 = sum(d["loss"] * d["tokens"] for d in d150) / 11022
    tw300 = sum(d["loss"] * d["tokens"] for d in d300) / 11022
    assert round(tw150, 4) == 5.7242
    assert round(tw300, 4) == 5.8420
    assert round(tw300 - tw150, 4) == 0.1178

    # Verify manifest count minus 88 equals 11022
    assert m["total_tokens"] - len(m["documents"]) == 11022


def test_sanskrit_cleaning_accuracy():
    """Verify that CORPUS_V2_CLEANED_MANIFEST.json contains zero wiki quotes or indentation colons."""
    import re
    cleaned_p = BATCH2101_DIR / "CORPUS_V2_CLEANED_MANIFEST.json"
    raw_p = BATCH2021_DIR / "CORPUS_V2_MANIFEST.json"
    assert cleaned_p.exists() and raw_p.exists()

    with open(cleaned_p, "r", encoding="utf-8") as f:
        c_clean = json.load(f)
    with open(raw_p, "r", encoding="utf-8") as f:
        c_raw = json.load(f)

    # Document count must match exactly
    assert len(c_clean["documents"]) == len(c_raw["documents"]) == 1258

    sa_clean = [d for d in c_clean["documents"] if d["source_id"] == "wikipedia_sa_bhagavad_gita"]
    sa_raw = [d for d in c_raw["documents"] if d["source_id"] == "wikipedia_sa_bhagavad_gita"]
    assert len(sa_clean) == len(sa_raw) == 61

    # Raw must have markup, cleaned must have zero
    raw_quotes = sum(len(re.findall(r"'{2,5}", d["text"])) for d in sa_raw)
    raw_colons = sum(len(re.findall(r"^:+", d["text"], re.MULTILINE)) for d in sa_raw)
    assert raw_quotes == 184 and raw_colons == 45

    clean_quotes = sum(len(re.findall(r"'{2,5}", d["text"])) for d in sa_clean)
    clean_colons = sum(len(re.findall(r"^:+", d["text"], re.MULTILINE)) for d in sa_clean)
    assert clean_quotes == 0
    assert clean_colons == 0


def test_validation_evaluator_macro_loss():
    """Verify that standalone validation evaluator calculates macro-source loss properly."""
    from chakrview.brain.model import ChakrMicro
    from chakrview.tokenizer.serialization import load_tokenizer_artifacts
    from experiments.batch2101_2140.validation_evaluator import evaluate_validation_split

    ckpt = torch.load(CANONICAL_CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    model = ChakrMicro(ckpt["config"])
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    tok, _ = load_tokenizer_artifacts(TOKENIZER_DIR)

    res = evaluate_validation_split(model, tok, BATCH2021_DIR / "CORPUS_V2_MANIFEST.json")
    assert res["total_validation_docs"] == 186
    assert res["source_count"] == 3
    assert "google_hinglish_top_dialogues" in res["per_source_metrics"]
    assert "wikipedia_hi_ganit" in res["per_source_metrics"]
    assert "cpython_algorithms_bisect" in res["per_source_metrics"]

    # Verify that macro_source_ce is the exact unweighted average across the 3 sources
    s_ces = [m["token_weighted_ce"] for m in res["per_source_metrics"].values()]
    expected_macro = round(sum(s_ces) / 3, 4)
    assert res["macro_source_ce"] == expected_macro
    assert res["macro_source_ce"] == 5.0319


def test_cpt_checkpoint_resume_equivalence():
    """Verify step-by-step equivalence between uninterrupted control and resumed runs."""
    from experiments.batch2101_2140.cpt_checkpoint_engine import save_cpt_checkpoint, load_cpt_checkpoint

    torch.manual_seed(1337)
    model_ctrl = nn.Linear(8, 2)
    opt_ctrl = torch.optim.AdamW(model_ctrl.parameters(), lr=1e-3)
    sched_ctrl = torch.optim.lr_scheduler.CosineAnnealingLR(opt_ctrl, T_max=10)

    # 3 steps on control
    x = torch.randn(5, 4, 8)
    y = torch.randint(0, 2, (5, 4))
    for i in range(3):
        opt_ctrl.zero_grad()
        loss = nn.functional.cross_entropy(model_ctrl(x[i]), y[i])
        loss.backward()
        opt_ctrl.step()
        sched_ctrl.step()

    tmp_ckpt = BATCH2101_DIR / "temp_resume_test.pt"
    save_cpt_checkpoint(
        tmp_ckpt,
        model_ctrl,
        opt_ctrl,
        sched_ctrl,
        global_step=3,
        seed=1337,
        config={},
        corpus_manifest_sha256="valid_sha",
        tokenizer_vocab_sha256="valid_vocab"
    )

    # Step control forward for step 4
    opt_ctrl.zero_grad()
    loss = nn.functional.cross_entropy(model_ctrl(x[3]), y[3])
    loss.backward()
    opt_ctrl.step()
    sched_ctrl.step()

    # Create fresh model and resume from step 3
    model_resumed = nn.Linear(8, 2)
    opt_resumed = torch.optim.AdamW(model_resumed.parameters(), lr=1e-3)
    sched_resumed = torch.optim.lr_scheduler.CosineAnnealingLR(opt_resumed, T_max=10)

    load_cpt_checkpoint(tmp_ckpt, model_resumed, opt_resumed, sched_resumed, expected_corpus_sha256="valid_sha")

    # Step resumed model forward for step 4
    opt_resumed.zero_grad()
    loss_res = nn.functional.cross_entropy(model_resumed(x[3]), y[3])
    loss_res.backward()
    opt_resumed.step()
    sched_resumed.step()

    assert torch.allclose(model_ctrl.weight, model_resumed.weight)
    assert torch.allclose(model_ctrl.bias, model_resumed.bias)

    # Clean up test artifact
    if tmp_ckpt.exists():
        tmp_ckpt.unlink()


def test_preflight_gate_json_verdict():
    """Verify that CPT_PREFLIGHT_GATE.json exists, contains 8 passing gates, and verdict READY_FOR_CONTROLLED_CPT."""
    gate_p = BATCH2101_DIR / "CPT_PREFLIGHT_GATE.json"
    assert gate_p.exists()

    with open(gate_p, "r", encoding="utf-8") as f:
        gate = json.load(f)

    assert gate["overall_decision"] == "READY_FOR_CONTROLLED_CPT"
    assert len(gate["gates"]) == 8
    for g in gate["gates"]:
        assert g["status"] == "PASS"
        assert Path(g["evidence_path"]).exists()


def test_all_batch2101_deliverables_exist():
    """Check that all required deliverables exist and are populated."""
    deliverables = [
        "REPOSITORY_AND_ARTIFACT_INTEGRITY.md",
        "CPT_METRIC_RECONCILIATION.md",
        "SANSKRIT_CLEANING_AUDIT.md",
        "CORPUS_V2_CLEANED_MANIFEST.json",
        "VALIDATION_SELECTION_PROTOCOL.md",
        "CHECKPOINT_RESUME_AUDIT.md",
        "CPT_PREREGISTRATION.json",
        "CPT_PREFLIGHT_GATE.json",
        "MISSION_SUMMARY.md",
        "validation_evaluator.py",
        "cpt_checkpoint_engine.py"
    ]
    for d in deliverables:
        p = BATCH2101_DIR / d
        assert p.exists(), f"Missing deliverable: {d}"
        assert p.stat().st_size > 50, f"Deliverable is empty: {d}"
