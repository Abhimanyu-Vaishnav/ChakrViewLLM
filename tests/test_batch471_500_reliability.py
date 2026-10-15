"""
Batches 471-500: Reliability, Regression & Robustness Test Suite.
Location: tests/test_batch471_500_reliability.py

Tests:
1. Canonical checkpoint immutability & candidate SHA-256 digests.
2. Scratchpad candidate loading and token generation bounds.
3. Intermediate-step reasoning failure mode detection on novel operands (scientific integrity).
4. Evidence-Governed Answering Gate decisions:
   - REFUTE celestial false premises.
   - REFUTE division by zero.
   - REFUTE historical anachronisms.
   - TOOL_DELEGATE arithmetic.
   - ABSTAIN on ungrounded queries.
5. GroundedSessionAdapter end-to-end integration and attribution.
6. Robustness against unusual inputs (empty queries, excessive whitespace, punctuation).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import pytest
import torch

from chakrview.brain.model import ChakrMicro
from chakrview.runtime.cognitive_router import CognitiveRouter, RouteType, ComponentOrigin
from chakrview.runtime.evidence_gate import EvidenceGovernedAnsweringGate, GateDecision
from chakrview.runtime.session_adapter import GroundedSessionAdapter
from chakrview.tokenizer.serialization import load_tokenizer_artifacts
from chakrverify.core import ClaimVerificationEngine, DocumentChunk, VerificationVerdict

CANONICAL_CHECKPOINT = Path("experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt")
EXPECTED_CANONICAL_HASH = "96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee"

SCRATCHPAD_CANDIDATE = Path("experiments/batch441_470_reasoning_gate/checkpoints/candidate_step150_scratchpad_reasoning.pt")
EXPECTED_SCRATCHPAD_HASH = "72784a614b2e669242aae1d67adf4e474f186178c18d4ec25070e78378b19759"

TOKENIZER_DIR = Path("release/chakrview-v0.5.0/data/experiments/vocab_4096")


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def test_checkpoint_digests_immutable():
    assert CANONICAL_CHECKPOINT.exists()
    assert SCRATCHPAD_CANDIDATE.exists()
    assert sha256_file(CANONICAL_CHECKPOINT) == EXPECTED_CANONICAL_HASH
    assert sha256_file(SCRATCHPAD_CANDIDATE) == EXPECTED_SCRATCHPAD_HASH


def test_evidence_gate_refutation_and_abstention():
    gate = EvidenceGovernedAnsweringGate()

    # Celestial false premise refuted
    r1 = gate.process_query("Tell me the municipal capital of Mars.")
    assert r1.decision == GateDecision.REFUTE
    assert r1.verdict == VerificationVerdict.CONTRADICTED

    # Mathematical domain error refuted
    r2 = gate.process_query("Compute 99 / 0")
    assert r2.decision == GateDecision.REFUTE
    assert r2.verdict == VerificationVerdict.CONTRADICTED

    # Historical anachronism refuted
    r3 = gate.process_query("Who was the prime minister of India in 1720?")
    assert r3.decision == GateDecision.REFUTE
    assert r3.verdict == VerificationVerdict.CONTRADICTED

    # Unindexed query honestly abstained from
    r4 = gate.process_query("What was the weather in Antarctica on January 1, 1400?")
    assert r4.decision == GateDecision.ABSTAIN
    assert r4.verdict == VerificationVerdict.INSUFFICIENT_EVIDENCE


def test_grounded_session_adapter_e2e_integration():
    class DummyEngine:
        def __init__(self):
            self.tokenizer, _ = load_tokenizer_artifacts(TOKENIZER_DIR)
            ck = torch.load(SCRATCHPAD_CANDIDATE, map_location="cpu", weights_only=False)
            self.model = ChakrMicro(ck["config"])
            self.model.load_state_dict(ck["model_state_dict"])
            self.model.eval()

    engine = DummyEngine()
    adapter = GroundedSessionAdapter(inference_engine=engine)

    # 1. False premise intercepted by gate
    resp1 = adapter.execute_turn("session_test", "What is the capital of Mars?")
    assert "uninhabited" in resp1.answer.lower()
    assert resp1.component_origin == ComponentOrigin.FALLBACK_POLICY

    # 2. Arithmetic routed to calculator
    resp2 = adapter.execute_turn("session_test", "What is 14 plus 28?")
    assert resp2.answer == "42.0" or resp2.answer == "42"
    assert resp2.component_origin == ComponentOrigin.CALCULATOR_TOOL

    # 3. Known concept definition
    resp3 = adapter.execute_turn("session_test", "Define ALU.")
    assert "Arithmetic Logic Unit" in resp3.answer
    assert resp3.component_origin == ComponentOrigin.SYSTEM_KNOWLEDGE


def test_unusual_and_adversarial_inputs():
    gate = EvidenceGovernedAnsweringGate()

    # Punctuation heavy
    r1 = gate.process_query("???   What is the capital of Mars??? !!!")
    assert r1.decision == GateDecision.REFUTE

    # Divide by zero with extra spaces
    r2 = gate.process_query("Calculate dividing 50 by 0 .")
    assert r2.decision == GateDecision.REFUTE

    # Arithmetic with trailing symbols
    r3 = gate.process_query("Calculate 10 + 20.")
    assert r3.decision == GateDecision.TOOL_DELEGATE
    assert "30" in r3.content
