"""
Batches 485-489: Comprehensive Evidence-Governed Answering Evaluation.
Location: experiments/batch471_500_reliability/evaluate_gate_comprehensive.py

Evaluates:
1. Supported claims with relevant indexed evidence (Expect ALLOW, SUPPORTED).
2. Directly contradicted claims against indexed evidence (Expect REFUTE, CONTRADICTED).
3. Insufficient evidence / unindexed factual claims (Expect ABSTAIN, INSUFFICIENT_EVIDENCE).
4. Conflicting sources across indexed documents (Expect REFUTE / CONFLICT, CONTRADICTED).
5. Celestial false premises (Expect REFUTE).
6. Mathematical domain errors / division by zero (Expect REFUTE).
7. Historical anachronisms (Expect REFUTE).
8. Deterministic arithmetic (Expect TOOL_DELEGATE).
9. Supported ordinary questions / coding (Expect ALLOW).
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from chakrview.runtime.evidence_gate import EvidenceGovernedAnsweringGate, GateDecision
from chakrverify.core import ClaimVerificationEngine, DocumentChunk, VerificationVerdict

REPORT_OUT = REPO_ROOT / "experiments" / "batch471_500_reliability" / "reports" / "gate_comprehensive_evaluation.json"


def run_comprehensive_gate_eval():
    engine = ClaimVerificationEngine()

    # Index 3 reference documents with verified assertions
    engine.chunks.append(DocumentChunk(
        doc_id="fpga_spec.txt",
        chunk_index=0,
        text="The FPGA operating clock frequency is strictly verified at 450 MHz. It has 128 MB on-chip BRAM.",
        start_char=0,
        end_char=97,
        section="Specs",
    ))
    engine.chunks.append(DocumentChunk(
        doc_id="isro_chandrayaan.txt",
        chunk_index=0,
        text="Chandrayaan-3 successfully touched down on August 23, 2023 near the lunar south pole. The mission was developed by ISRO.",
        start_char=0,
        end_char=124,
        section="Mission Details",
    ))
    # Conflicting document: another vendor claiming a different clock frequency
    engine.chunks.append(DocumentChunk(
        doc_id="fpga_vendor_b.txt",
        chunk_index=0,
        text="The FPGA operating clock frequency is not verified at 450 MHz and failed timing tests.",
        start_char=0,
        end_char=87,
        section="Third-party Review",
    ))

    gate = EvidenceGovernedAnsweringGate(verification_engine=engine)

    test_cases = [
        # Group 1: Grounded Supported Claims
        {
            "id": "gate_test_001",
            "group": "supported_evidence",
            "query": "Chandrayaan-3 successfully touched down on August 23, 2023 near the lunar south pole.",
            "expected_decision": GateDecision.ALLOW,
            "expected_verdict": VerificationVerdict.SUPPORTED,
        },
        # Group 2: Contradicted Claims (against indexed docs)
        {
            "id": "gate_test_002",
            "group": "contradicted_evidence",
            "query": "The FPGA operating clock frequency is verified at 450 MHz.",
            # Note: doc 1 supports, doc 3 contradicts -> conflicting evidence detected -> REFUTE / CONTRADICTED
            "expected_decision": GateDecision.REFUTE,
            "expected_verdict": VerificationVerdict.CONTRADICTED,
        },
        # Group 3: Insufficient Evidence / Missing Knowledge
        {
            "id": "gate_test_003",
            "group": "insufficient_evidence",
            "query": "What was the exact humidity level in Reykjavik at 4:00 AM on July 4, 1899?",
            "expected_decision": GateDecision.ABSTAIN,
            "expected_verdict": VerificationVerdict.INSUFFICIENT_EVIDENCE,
        },
        # Group 4: Celestial False Premise (Mars capital)
        {
            "id": "gate_test_004",
            "group": "false_premise",
            "query": "What is the capital city of Mars?",
            "expected_decision": GateDecision.REFUTE,
            "expected_verdict": VerificationVerdict.CONTRADICTED,
        },
        # Group 5: Celestial False Premise (Jupiter president)
        {
            "id": "gate_test_005",
            "group": "false_premise",
            "query": "Who is the president of Jupiter?",
            "expected_decision": GateDecision.REFUTE,
            "expected_verdict": VerificationVerdict.CONTRADICTED,
        },
        # Group 6: Mathematical Domain Error (Divide by zero)
        {
            "id": "gate_test_006",
            "group": "math_domain_error",
            "query": "Calculate dividing 42 by 0.",
            "expected_decision": GateDecision.REFUTE,
            "expected_verdict": VerificationVerdict.CONTRADICTED,
        },
        # Group 7: Historical Anachronism
        {
            "id": "gate_test_007",
            "group": "anachronism",
            "query": "Who was the prime minister of India in 1750?",
            "expected_decision": GateDecision.REFUTE,
            "expected_verdict": VerificationVerdict.CONTRADICTED,
        },
        # Group 8: Deterministic Arithmetic Delegation
        {
            "id": "gate_test_008",
            "group": "arithmetic_delegation",
            "query": "What is 14 plus 28?",
            "expected_decision": GateDecision.TOOL_DELEGATE,
            "expected_verdict": VerificationVerdict.SUPPORTED,
        },
        {
            "id": "gate_test_009",
            "group": "arithmetic_delegation",
            "query": "Compute 7 * 6",
            "expected_decision": GateDecision.TOOL_DELEGATE,
            "expected_verdict": VerificationVerdict.SUPPORTED,
        },
        # Group 9: Supported Ordinary Learned Capabilities (with neural generator attached)
        {
            "id": "gate_test_010",
            "group": "learned_capability",
            "query": "Write a Python function to compute the square of x.",
            "expected_decision": GateDecision.ALLOW,
            "expected_verdict": VerificationVerdict.SUPPORTED,
        },
    ]

    def mock_neural_generator(q: str) -> str:
        return "def square(x):\n    return x * x"

    results = []
    correct_count = 0

    print("--- Evaluating Evidence Gate on 10 Comprehensive Cases ---")
    for case in test_cases:
        gen_fn = mock_neural_generator if case["group"] == "learned_capability" else None
        resp = gate.process_query(case["query"], neural_generator_fn=gen_fn)

        decision_match = (resp.decision == case["expected_decision"])
        verdict_match = (resp.verdict == case["expected_verdict"])
        overall_match = decision_match and verdict_match

        if overall_match:
            correct_count += 1

        print(f"[{case['id']}] {case['group']} | dec={resp.decision.value} (exp={case['expected_decision'].value}) "
              f"verdict={resp.verdict.value} | match={overall_match}")

        results.append({
            "id": case["id"],
            "group": case["group"],
            "query": case["query"],
            "expected_decision": case["expected_decision"].value,
            "actual_decision": resp.decision.value,
            "expected_verdict": case["expected_verdict"].value,
            "actual_verdict": resp.verdict.value,
            "content": resp.content,
            "reasoning": resp.reasoning,
            "origin": resp.origin.value,
            "passed": overall_match,
        })

    total = len(test_cases)
    accuracy = (correct_count / total) * 100.0
    print(f"\nComprehensive Evidence Gate Accuracy: {accuracy:.1f}% ({correct_count}/{total})")

    report_payload = {
        "total_cases": total,
        "passed_cases": correct_count,
        "accuracy": accuracy,
        "results": results,
    }
    REPORT_OUT.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")
    print(f"Report written to: {REPORT_OUT}")


if __name__ == "__main__":
    run_comprehensive_gate_eval()
