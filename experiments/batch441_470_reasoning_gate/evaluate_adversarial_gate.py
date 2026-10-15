"""
Batch 465-467: Adversarial and wording robustness evaluation for Evidence-Governed Answering.
Tests:
1. Wording variations of false premises (e.g. asking indirectly about Mars capital).
2. Direct vs indirect contradiction assertions in document verification.
3. Unsupported historical claims with varying phrasing.
4. Positive control preservation (ensuring supported claims are not refused).
"""

import sys
import json
import logging
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from chakrview.runtime.evidence_gate import EvidenceGovernedAnsweringGate, GateDecision
from chakrverify.core import ClaimVerificationEngine, DocumentChunk, VerificationVerdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_adversarial_evaluation():
    engine = ClaimVerificationEngine()
    
    # Add indexed reference documents
    doc_text = (
        "The Chandrayaan-3 lunar lander successfully touched down near the south pole of the Moon "
        "on August 23, 2023. It was launched by ISRO."
    )
    engine.chunks.append(DocumentChunk(
        doc_id="chandrayaan_mission.txt",
        chunk_index=0,
        text=doc_text,
        start_char=0,
        end_char=len(doc_text),
        section="Summary",
    ))
    gate = EvidenceGovernedAnsweringGate(verification_engine=engine)
    
    test_cases = [
        # Adversarial 1: Paraphrased false premise
        {
            "query": "Could you tell me the name of the capital city on the planet Mars?",
            "expected_decision": GateDecision.REFUTE,
            "type": "paraphrased_false_premise"
        },
        # Adversarial 2: Mathematical impossibility / undefined operation
        {
            "query": "Compute what happens when dividing 9999 by zero",
            "expected_decision": GateDecision.REFUTE,
            "type": "division_by_zero"
        },
        # Adversarial 3: Contradictory claim against indexed evidence
        {
            "query": "Chandrayaan-3 was launched by NASA in the year 2020.",
            "expected_decision": GateDecision.REFUTE,
            "type": "contradiction_rejection"
        },
        # Adversarial 4: Subtle wording supported claim
        {
            "query": "On what date did the Chandrayaan-3 lander touch down near the lunar south pole?",
            "expected_decision": GateDecision.ALLOW,
            "type": "supported_evidence"
        },
        # Adversarial 5: Honest abstention on completely unknown/unsupported fact
        {
            "query": "What was the exact temperature in Vladivostok at 3:15 AM on October 12, 1912?",
            "expected_decision": GateDecision.ABSTAIN,
            "type": "honest_abstention"
        },
        # Adversarial 6: Deterministic arithmetic without simple keywords
        {
            "query": "Calculate the value of (25 * 4) + 17",
            "expected_decision": GateDecision.TOOL_DELEGATE,
            "type": "arithmetic_delegation"
        }
    ]
    
    results = []
    passed = 0
    
    for case in test_cases:
        res = gate.process_query(case["query"])
        is_pass = (res.decision == case["expected_decision"])
        if is_pass:
            passed += 1
        results.append({
            "query": case["query"],
            "expected_decision": case["expected_decision"].value,
            "actual_decision": res.decision.value,
            "verdict": res.verdict.value,
            "content": res.content,
            "reasoning": res.reasoning,
            "passed": is_pass,
            "type": case["type"]
        })
        
    accuracy = (passed / len(test_cases)) * 100.0
    print(f"Adversarial Gate Accuracy: {accuracy:.1f}% ({passed}/{len(test_cases)})")
    
    out_path = REPO_ROOT / "experiments" / "batch441_470_reasoning_gate" / "reports" / "adversarial_gate_evaluation.json"
    out_path.write_text(json.dumps({
        "accuracy": accuracy,
        "total": len(test_cases),
        "passed": passed,
        "cases": results
    }, indent=2), encoding="utf-8")
    print(f"Adversarial report written to: {out_path}")

if __name__ == "__main__":
    run_adversarial_evaluation()
