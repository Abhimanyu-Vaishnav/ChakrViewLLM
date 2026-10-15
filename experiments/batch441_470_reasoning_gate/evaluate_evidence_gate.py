"""
Batches 460-464: Evaluation of Evidence-Governed Answering Gate.
Location: experiments/batch441_470_reasoning_gate/evaluate_evidence_gate.py

Evaluates:
1. False-premise rejection (Mars municipal capital -> REFUTE).
2. Undefined math rejection (divide by zero -> REFUTE).
3. Historical anachronism rejection (PM in 1750 -> REFUTE).
4. Supported arithmetic routing (14 + 28 -> TOOL_DELEGATE -> 42).
5. Document-grounded claim support via ChakrVerify.
6. Neural core generation for coding / definitions.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, ".")

from chakrview.runtime.evidence_gate import EvidenceGovernedAnsweringGate, GateDecision
from chakrverify.core import ClaimVerificationEngine, DocumentChunk, VerificationVerdict

OUT_JSON = Path("experiments/batch441_470_reasoning_gate/reports/evidence_gate_evaluation.json")


def run_evaluation():
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

    test_queries = [
        # 1. False premise
        {"q": "What is the capital of Mars?", "expected_decision": GateDecision.REFUTE, "verdict": VerificationVerdict.CONTRADICTED},
        # 2. Mathematical error
        {"q": "Calculate the value of dividing 42 by 0.", "expected_decision": GateDecision.REFUTE, "verdict": VerificationVerdict.CONTRADICTED},
        # 3. Anachronism
        {"q": "Who was the prime minister of India in the year 1750?", "expected_decision": GateDecision.REFUTE, "verdict": VerificationVerdict.CONTRADICTED},
        # 4. Arithmetic routing
        {"q": "What is 14 plus 28?", "expected_decision": GateDecision.TOOL_DELEGATE, "verdict": VerificationVerdict.SUPPORTED},
        # 5. Document-grounded claim
        {"q": "The FPGA operating clock frequency is verified at 450 MHz.", "expected_decision": GateDecision.ALLOW, "verdict": VerificationVerdict.SUPPORTED},
        # 6. Neural coding query
        {"q": "Write a Python function to compute the square of x.", "expected_decision": GateDecision.ALLOW, "verdict": VerificationVerdict.SUPPORTED},
    ]

    def mock_neural_gen(prompt):
        return "def square(x):\n    return x * x"

    results = []
    correct_decisions = 0

    for item in test_queries:
        res = gate.process_query(item["q"], neural_generator_fn=mock_neural_gen)
        is_ok = (res.decision == item["expected_decision"]) and (res.verdict == item["verdict"])
        if is_ok:
            correct_decisions += 1
        results.append({
            "query": item["q"],
            "expected_decision": item["expected_decision"].value,
            "actual_decision": res.decision.value,
            "expected_verdict": item["verdict"].value,
            "actual_verdict": res.verdict.value,
            "content": res.content,
            "origin": res.origin.value,
            "is_correct": is_ok,
        })

    summary = {
        "total_queries": len(test_queries),
        "correct_decisions": correct_decisions,
        "gate_accuracy": correct_decisions / len(test_queries),
        "details": results,
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Evidence Gate Accuracy: {summary['gate_accuracy']*100:.1f}% ({correct_decisions}/{len(test_queries)})")
    print(f"Report written to: {OUT_JSON}")


if __name__ == "__main__":
    run_evaluation()
