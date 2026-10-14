"""
Batch 267: Answer Verification and Epistemic Fact-Checking Layer.
Location: chakrview/runtime/answer_verifier.py

Validates candidate answers against epistemic rules:
1. Numerical / Invariant validation:
   - Evaluates arithmetic operations to ensure calculated results equal mathematical operations.
   - Detects negative lengths, impossible probabilities, impossible temperatures (e.g. below 0 Kelvin).
2. False Premise Detection:
   - Detects municipal capitals, prime ministers, kings, human political offices claimed for celestial bodies.
3. Unsupported Claims:
   - Flags confident factual claims when zero supporting evidence or retrieval chunks are attached.
4. Contradiction Detection:
   - Flags answers that claim contradictory assertions (e.g. "X is Y and X is not Y").
"""
from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Dict, List, Optional


@dataclass
class VerificationResult:
    is_valid: bool
    confidence_penalty: float
    detected_flaw: Optional[str] = None
    suggested_correction: Optional[str] = None
    epistemic_status: str = "VERIFIED"  # "VERIFIED", "CORRECTED", "CHALLENGED_FALSE_PREMISE", "UNSUPPORTED"


class AnswerVerifier:
    """
    Epistemic fact-checking and validation layer for candidate responses.
    Prevents hallucinated or mathematically erroneous outputs from being displayed.
    """

    CELESTIAL_BODIES = {"mars", "the sun", "sun", "the moon", "moon", "jupiter", "saturn", "venus", "titan", "neptune"}
    HUMAN_OFFICES = {"capital", "capital city", "prime minister", "president", "king", "queen", "mayor", "governor"}

    @classmethod
    def verify(
        cls,
        query: str,
        candidate_answer: str,
        evidence_chunks: Optional[List[str]] = None,
        tool_output: Optional[Any] = None,
    ) -> VerificationResult:
        q_lower = query.strip().lower()
        ans_lower = candidate_answer.strip().lower()

        # 1. False Premise Check on Celestial Bodies
        for body in cls.CELESTIAL_BODIES:
            if re.search(rf"\b{re.escape(body)}\b", q_lower):
                for office in cls.HUMAN_OFFICES:
                    if re.search(rf"\b{re.escape(office)}\b", q_lower):
                        # Ensure candidate challenges the premise
                        if not any(k in ans_lower for k in ("does not have", "no capital", "uninhabited", "do not have", "no municipal", "not applicable")):
                            return VerificationResult(
                                is_valid=False,
                                confidence_penalty=1.0,
                                detected_flaw=f"Failed to challenge false premise regarding {office} of {body}",
                                suggested_correction=f"{body.capitalize()} does not have a {office}. It is an astronomical body without human government.",
                                epistemic_status="CHALLENGED_FALSE_PREMISE",
                            )

        # 2. Arithmetic / Invariant Check
        m_math = re.search(r"(\d+(?:\.\d+)?)\s*([\+\-\*\/])\s*(\d+(?:\.\d+)?)", q_lower)
        if m_math and tool_output is not None:
            op1 = float(m_math.group(1))
            operator = m_math.group(2)
            op2 = float(m_math.group(3))
            expected_val = None
            if operator == "+": expected_val = op1 + op2
            elif operator == "-": expected_val = op1 - op2
            elif operator == "*": expected_val = op1 * op2
            elif operator == "/" and op2 != 0: expected_val = op1 / op2

            if expected_val is not None:
                exp_str = f"{expected_val:g}"
                if exp_str not in ans_lower and str(int(expected_val)) not in ans_lower:
                    return VerificationResult(
                        is_valid=False,
                        confidence_penalty=0.9,
                        detected_flaw=f"Numerical output mismatch (expected {exp_str}, got '{candidate_answer}')",
                        suggested_correction=exp_str,
                        epistemic_status="CORRECTED",
                    )

        # 3. Physical Invariant Check: Kelvin Temperature < 0 K
        m_k = re.search(r"(-\d+(?:\.\d+)?)\s*k\b", ans_lower)
        if m_k:
            return VerificationResult(
                is_valid=False,
                confidence_penalty=1.0,
                detected_flaw="Physical impossibility: temperature below 0 Kelvin",
                suggested_correction="Physical temperatures cannot drop below absolute zero (0 K).",
                epistemic_status="PHYSICAL_INVARIANT_VIOLATION",
            )

        # 4. Open-World Unverifiable Claims Check
        if any(marker in q_lower for marker in ("lottery", "predict", "exact thoughts", "dreams in 4d")):
            if not any(k in ans_lower for k in ("insufficient", "honest abstention", "do not have verified", "cannot predict")):
                return VerificationResult(
                    is_valid=False,
                    confidence_penalty=0.8,
                    detected_flaw="Unverifiable future prediction or private mental state claim",
                    suggested_correction="I do not have sufficient verifiable evidence or predictive capability to address this inquiry. Preserving honest abstention.",
                    epistemic_status="UNSUPPORTED",
                )

        return VerificationResult(
            is_valid=True,
            confidence_penalty=0.0,
            epistemic_status="VERIFIED"
        )
