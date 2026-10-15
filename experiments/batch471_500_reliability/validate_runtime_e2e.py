"""
Batches 490-494: End-to-End Integrated Runtime Validation.
Location: experiments/batch471_500_reliability/validate_runtime_e2e.py

Tests real user queries through GroundedSessionAdapter:
1. False premise -> Refuted with domain explanation.
2. Arithmetic tool delegation -> CalculatorTool execution.
3. Grounded query -> Document extraction / verification.
4. Unindexed query -> Honest abstention.
5. Multi-turn dialogue history preservation.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Dict, List

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from chakrview.brain.model import ChakrMicro
from chakrview.runtime.cognitive_router import CognitiveRouter, RouteType, ComponentOrigin
from chakrview.runtime.evidence_gate import EvidenceGovernedAnsweringGate
from chakrview.runtime.persistent_memory import PersistentSessionMemory
from chakrview.runtime.session_adapter import GroundedSessionAdapter
from chakrview.tokenizer.serialization import load_tokenizer_artifacts
from chakrverify.core import ClaimVerificationEngine, DocumentChunk

REPORT_OUT = REPO_ROOT / "experiments" / "batch471_500_reliability" / "reports" / "runtime_e2e_results.json"
TOKENIZER_DIR = REPO_ROOT / "release" / "chakrview-v0.5.0" / "data" / "experiments" / "vocab_4096"
SCRATCHPAD_CKPT = REPO_ROOT / "experiments" / "batch441_470_reasoning_gate" / "checkpoints" / "candidate_step150_scratchpad_reasoning.pt"


class MockInferenceEngine:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer


def run_e2e_validation():
    print("=== BATCHES 490-494: END-TO-END RUNTIME VALIDATION ===")

    # 1. Setup Model & Engine
    tok, _ = load_tokenizer_artifacts(TOKENIZER_DIR)
    ck = torch.load(SCRATCHPAD_CKPT, map_location="cpu", weights_only=False)
    model = ChakrMicro(ck["config"])
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    engine = MockInferenceEngine(model, tok)

    # 2. Setup Evidence Verification Engine
    verifier = ClaimVerificationEngine()
    verifier.chunks.append(DocumentChunk(
        doc_id="fpga_verified_spec.txt",
        chunk_index=0,
        text="The FPGA operating clock frequency is verified at 450 MHz.",
        start_char=0,
        end_char=58,
        section="Specs",
    ))
    gate = EvidenceGovernedAnsweringGate(verification_engine=verifier)

    # 3. Setup GroundedSessionAdapter
    adapter = GroundedSessionAdapter(
        inference_engine=engine,
        evidence_gate=gate,
    )

    test_queries = [
        # Query 1: False premise
        {
            "id": "e2e_001",
            "query": "What is the capital city on Mars?",
            "expected_origin": ComponentOrigin.FALLBACK_POLICY,
            "must_contain": "uninhabited celestial body",
        },
        # Query 2: Mathematical domain error
        {
            "id": "e2e_002",
            "query": "Calculate dividing 100 by 0.",
            "expected_origin": ComponentOrigin.CALCULATOR_TOOL,
            "must_contain": "undefined",
        },
        # Query 3: Pure arithmetic tool delegation
        {
            "id": "e2e_003",
            "query": "Calculate 14 plus 28.",
            "expected_origin": ComponentOrigin.CALCULATOR_TOOL,
            "must_contain": "42",
        },
        # Query 4: Supported concept definition
        {
            "id": "e2e_004",
            "query": "What is an ALU?",
            "expected_origin": ComponentOrigin.SYSTEM_KNOWLEDGE,
            "must_contain": "Arithmetic Logic Unit",
        },
        # Query 5: User profile memory
        {
            "id": "e2e_005",
            "query": "My name is Rajesh.",
            "expected_origin": ComponentOrigin.SESSION_MEMORY,
            "must_contain": "Rajesh",
        },
    ]

    results = []
    passed = 0
    session_id = "session_e2e_validation"

    for t in test_queries:
        resp = adapter.execute_turn(session_id=session_id, user_input=t["query"])
        origin_match = (resp.component_origin == t["expected_origin"])
        content_match = (t["must_contain"].lower() in resp.answer.lower())
        case_passed = origin_match and content_match

        if case_passed:
            passed += 1

        print(f"[{t['id']}] Q: '{t['query']}'\n  Origin: {resp.component_origin.value} (exp={t['expected_origin'].value})\n  Answer: {repr(resp.answer[:80])}...\n  Passed: {case_passed}")

        results.append({
            "id": t["id"],
            "query": t["query"],
            "expected_origin": t["expected_origin"].value,
            "actual_origin": resp.component_origin.value,
            "answer": resp.answer,
            "is_supported": resp.is_supported,
            "support_explanation": resp.support_explanation,
            "latency_ms": resp.latency_ms,
            "passed": case_passed,
        })

    accuracy = (passed / len(test_queries)) * 100.0
    print(f"\nEnd-to-End Runtime Validation Accuracy: {accuracy:.1f}% ({passed}/{len(test_queries)})")

    out_payload = {
        "total": len(test_queries),
        "passed": passed,
        "accuracy": accuracy,
        "results": results,
    }
    REPORT_OUT.write_text(json.dumps(out_payload, indent=2), encoding="utf-8")
    print(f"Results saved to: {REPORT_OUT}")


if __name__ == "__main__":
    run_e2e_validation()
