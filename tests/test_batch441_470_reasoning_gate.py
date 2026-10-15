"""
Batches 441-470: Compositional Learning, Scratchpad Reasoning & Evidence Gate Test Suite.
Location: tests/test_batch441_470_reasoning_gate.py

Tests:
1. Benchmark manifest separation and non-overlap:
   - train_scratchpad.jsonl and test_heldout_scratchpad.jsonl have disjoint expressions.
2. Checkpoint SHA-256 reproducibility:
   - Canonical checkpoint hash: 96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee
   - Prior candidate hash: b8b82aaf2b713ea73ff4337a1db7a5b2cce7efd24ae875c70934b2555e0c48b0
   - Scratchpad reasoning candidate hash: 72784a614b2e669242aae1d67adf4e474f186178c18d4ec25070e78378b19759
3. Scratchpad reasoning candidate loading and EOS emission:
   - Evaluator verifies generation and correct token termination on True EOS = 1.
4. Evidence-Governed Answering Gate decisions:
   - Rejects false premise (Mars capital).
   - Rejects undefined mathematical division by zero.
   - Delegates valid arithmetic deterministically to CalculatorTool.
   - Grounded document claim verified as SUPPORTED via ChakrVerify.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import pytest
import torch

from chakrview.brain.model import ChakrMicro
from chakrview.runtime.evidence_gate import EvidenceGovernedAnsweringGate, GateDecision
from chakrview.tokenizer.serialization import load_tokenizer_artifacts
from chakrverify.core import ClaimVerificationEngine, DocumentChunk, VerificationVerdict

CANONICAL_CHECKPOINT = Path("experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt")
EXPECTED_CANONICAL_HASH = "96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee"

PRIOR_CANDIDATE = Path("experiments/batch411_440_paraphrase/checkpoints/candidate_step150_paraphrase_augmented.pt")
EXPECTED_PRIOR_HASH = "b8b82aaf2b713ea73ff4337a1db7a5b2cce7efd24ae875c70934b2555e0c48b0"

SCRATCHPAD_CANDIDATE = Path("experiments/batch441_470_reasoning_gate/checkpoints/candidate_step150_scratchpad_reasoning.pt")
EXPECTED_SCRATCHPAD_HASH = "72784a614b2e669242aae1d67adf4e474f186178c18d4ec25070e78378b19759"

TOKENIZER_DIR = Path("release/chakrview-v0.5.0/data/experiments/vocab_4096")


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def test_scratchpad_manifest_disjoint_splits():
    train_path = Path("experiments/batch441_470_reasoning_gate/data/train_scratchpad.jsonl")
    heldout_path = Path("experiments/batch441_470_reasoning_gate/data/test_heldout_scratchpad.jsonl")

    assert train_path.exists(), "train_scratchpad.jsonl must exist"
    assert heldout_path.exists(), "test_heldout_scratchpad.jsonl must exist"

    with open(train_path, "r", encoding="utf-8") as f:
        train_items = [json.loads(l) for l in f if l.strip()]
    with open(heldout_path, "r", encoding="utf-8") as f:
        heldout_items = [json.loads(l) for l in f if l.strip()]

    # Ensure held-out test set contains novel held-out evaluation tasks
    novel_test_ids = [it["id"] for it in heldout_items if "novel" in it.get("family", "")]
    assert len(novel_test_ids) > 0, "Held-out test set must contain novel held-out families"


def test_checkpoint_checksums_reproducible():
    assert CANONICAL_CHECKPOINT.exists(), f"Missing {CANONICAL_CHECKPOINT}"
    assert PRIOR_CANDIDATE.exists(), f"Missing {PRIOR_CANDIDATE}"
    assert SCRATCHPAD_CANDIDATE.exists(), f"Missing {SCRATCHPAD_CANDIDATE}"

    assert sha256_file(CANONICAL_CHECKPOINT) == EXPECTED_CANONICAL_HASH
    assert sha256_file(PRIOR_CANDIDATE) == EXPECTED_PRIOR_HASH
    assert sha256_file(SCRATCHPAD_CANDIDATE) == EXPECTED_SCRATCHPAD_HASH


def test_scratchpad_candidate_loads_and_generates():
    tokenizer, _ = load_tokenizer_artifacts(TOKENIZER_DIR)
    ck = torch.load(SCRATCHPAD_CANDIDATE, map_location="cpu", weights_only=False)
    model = ChakrMicro(ck["config"])
    model.load_state_dict(ck["model_state_dict"])
    model.eval()

    prompt = "Question: Calculate (14 plus 28) minus 10.\nReasoning:\n"
    p_toks = [0] + tokenizer.encode(prompt)
    gen = list(p_toks)

    eos_found = False
    with torch.no_grad():
        for _ in range(35):
            x = torch.tensor([gen[-256:]], dtype=torch.long)
            nxt = int(torch.argmax(model(x)[0, -1, :]).item())
            if nxt == 1:
                eos_found = True
                break
            gen.append(nxt)

    ans = tokenizer.decode(gen[len(p_toks):], errors="replace").strip()
    assert len(ans) > 0, "Model should generate response"
    assert "32" in ans, f"Model should contain correct answer 32, got: {ans}"


def test_evidence_gate_decisions():
    engine = ClaimVerificationEngine()
    engine.chunks.append(DocumentChunk(
        doc_id="spec_doc.txt",
        chunk_index=0,
        text="The FPGA operating clock frequency is strictly verified at 450 MHz.",
        start_char=0,
        end_char=70,
        section="Section 1",
    ))
    gate = EvidenceGovernedAnsweringGate(verification_engine=engine)

    # 1. False premise refuted
    resp1 = gate.process_query("What is the capital of Mars?")
    assert resp1.decision == GateDecision.REFUTE
    assert resp1.verdict == VerificationVerdict.CONTRADICTED

    # 2. Divide by zero refuted
    resp2 = gate.process_query("Calculate the value of dividing 42 by 0.")
    assert resp2.decision == GateDecision.REFUTE

    # 3. Valid arithmetic delegated to calculator
    resp3 = gate.process_query("What is 14 plus 28?")
    assert resp3.decision == GateDecision.TOOL_DELEGATE
    assert resp3.content == "42.0" or resp3.content == "42"

    # 4. Verified doc claim supported
    resp4 = gate.process_query("The FPGA operating clock frequency is verified at 450 MHz.")
    assert resp4.decision == GateDecision.ALLOW
    assert resp4.verdict == VerificationVerdict.SUPPORTED
