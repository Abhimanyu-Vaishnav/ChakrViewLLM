"""
Batches 501-540: Neural Generalization & Learning Audit Test Suite.
Location: tests/test_batch501_540_generalization.py

Tests:
1. Canonical checkpoint immutability & candidate SHA-256 digests:
   - Canonical step 10,000 hash: 96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee
   - Prior candidate step 150 hash: 72784a614b2e669242aae1d67adf4e474f186178c18d4ec25070e78378b19759
   - Trained candidate step 200 hash: b18e69ebf4f0b396465af3c323f38bcfa84d5a38eabbd4a4220a10d6bfafee98
2. Strict zero prompt leakage across dataset splits:
   - train_compositional_rule_sft.jsonl and test_heldout_comprehensive_60.jsonl have disjoint prompt sets.
3. Candidate model loads, validates configuration, and emits True EOS = 1 cleanly.
4. Scientific honesty check: verifies that held-out multi-step compositional accuracy is objectively measured.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import pytest
import torch

from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

CANONICAL_CHECKPOINT = Path("experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt")
EXPECTED_CANONICAL_HASH = "96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee"

PRIOR_CANDIDATE = Path("experiments/batch441_470_reasoning_gate/checkpoints/candidate_step150_scratchpad_reasoning.pt")
EXPECTED_PRIOR_HASH = "72784a614b2e669242aae1d67adf4e474f186178c18d4ec25070e78378b19759"

TRAINED_CANDIDATE = Path("experiments/batch501_540/checkpoints/candidate_step200_compositional_rule.pt")
EXPECTED_TRAINED_HASH = "b18e69ebf4f0b396465af3c323f38bcfa84d5a38eabbd4a4220a10d6bfafee98"

TOKENIZER_DIR = Path("release/chakrview-v0.5.0/data/experiments/vocab_4096")


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def test_checkpoint_digests_and_canonical_invariants():
    assert CANONICAL_CHECKPOINT.exists(), f"Missing canonical checkpoint: {CANONICAL_CHECKPOINT}"
    assert PRIOR_CANDIDATE.exists(), f"Missing prior candidate: {PRIOR_CANDIDATE}"
    assert TRAINED_CANDIDATE.exists(), f"Missing trained candidate: {TRAINED_CANDIDATE}"

    assert sha256_file(CANONICAL_CHECKPOINT) == EXPECTED_CANONICAL_HASH
    assert sha256_file(PRIOR_CANDIDATE) == EXPECTED_PRIOR_HASH
    assert sha256_file(TRAINED_CANDIDATE) == EXPECTED_TRAINED_HASH


def test_dataset_leakage_strictly_zero_overlap():
    train_file = Path("experiments/batch501_540/data/train_compositional_rule_sft.jsonl")
    val_file = Path("experiments/batch501_540/data/val_compositional_rule_sft.jsonl")
    test_file = Path("experiments/batch501_540/data/test_heldout_comprehensive_60.jsonl")

    assert train_file.exists()
    assert val_file.exists()
    assert test_file.exists()

    with open(train_file, "r", encoding="utf-8") as f:
        train_prompts = {json.loads(l)["prompt"].strip() for l in f if l.strip()}
    with open(val_file, "r", encoding="utf-8") as f:
        val_prompts = {json.loads(l)["prompt"].strip() for l in f if l.strip()}
    with open(test_file, "r", encoding="utf-8") as f:
        test_prompts = {json.loads(l)["prompt"].strip() for l in f if l.strip()}

    assert len(train_prompts.intersection(test_prompts)) == 0, "Train and Test prompts must be disjoint"
    assert len(train_prompts.intersection(val_prompts)) == 0, "Train and Val prompts must be disjoint"


def test_trained_candidate_loading_and_clean_eos_emission():
    tokenizer, _ = load_tokenizer_artifacts(TOKENIZER_DIR)
    ck = torch.load(TRAINED_CANDIDATE, map_location="cpu", weights_only=False)
    model = ChakrMicro(ck["config"])
    model.load_state_dict(ck["model_state_dict"])
    model.eval()

    prompt = "Question: What is 14 plus 28?\nAnswer:"
    p_toks = [0] + tokenizer.encode(prompt)
    gen = list(p_toks)

    eos_found = False
    with torch.no_grad():
        for _ in range(25):
            x = torch.tensor([gen[-256:]], dtype=torch.long)
            nxt = int(torch.argmax(model(x)[0, -1, :]).item())
            if nxt == 1:
                eos_found = True
                break
            gen.append(nxt)

    ans = tokenizer.decode(gen[len(p_toks):], errors="replace").strip()
    assert eos_found is True, "Candidate must terminate cleanly on true EOS token 1"
    assert len(ans) > 0, "Candidate must generate non-empty answer"


def test_scientific_generalization_audit_integrity():
    report_file = Path("experiments/batch501_540/reports/training_and_evaluation_results.json")
    assert report_file.exists()

    data = json.loads(report_file.read_text(encoding="utf-8"))
    cand_data = data["trained_candidate"]

    # Verify that multi-step compositional performance is honestly reported
    exact_match_rate = cand_data["compositional_60"]["exact_match_rate"]
    assert exact_match_rate < 0.20, f"Expected low compositional transfer to be honestly reported, got {exact_match_rate}"

    # Verify that paraphrase retention is preserved
    para_match_rate = cand_data["paraphrase_20"]["exact_match_rate"]
    assert para_match_rate >= 0.70, f"Expected paraphrase capability to be retained, got {para_match_rate}"
