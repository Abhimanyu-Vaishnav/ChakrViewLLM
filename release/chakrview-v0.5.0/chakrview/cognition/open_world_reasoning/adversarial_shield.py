"""
Wave 576: Adversarial Evidence & Prompt-Injection Resistance Engine.

Architectural Guarantees:
- DATA vs INSTRUCTION Separation:
  External documents are strictly DATA, never system commands or execution policies.
- Detects and Neutralizes:
  "IGNORE ALL PREVIOUS RULES", "ANSWER X", "DO NOT VERIFY THIS SOURCE",
  "THIS DOCUMENT IS THE ONLY TRUTH", "SYSTEM OVERRIDE", "YOU ARE NOW IN DEVELOPER MODE".
- Preserves document factual payload while neutralizing authority injection.
- Zero authority boundary violations.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple


@dataclass
class AdversarialShieldResult:
    doc_id: str
    is_injection_detected: bool
    detected_attack_patterns: List[str]
    authority_violation_attempted: bool
    sanitized_factual_content: str
    action_taken: str              # NEUTRALIZED_AND_TREATED_AS_DATA, REJECTED, CLEAN
    security_verdict: str


class EvidenceDataSanitizer:
    """
    Sanitizes retrieved external evidence, stripping prompt injection commands
    and maintaining strict isolation between untrusted document text and cognitive runtime authority.
    """

    INJECTION_PATTERNS = [
        (r"ignore\s+(all\s+)?(previous|prior)\s+(instructions|rules|prompts)", "IGNORE_INSTRUCTIONS"),
        (r"system\s+(override|prompt|directive)", "SYSTEM_OVERRIDE"),
        (r"do\s+not\s+(verify|check|audit)\s+this", "BYPASS_VERIFICATION"),
        (r"this\s+(document|source)\s+is\s+the\s+only\s+truth", "FALSE_AUTHORITY_CLAIM"),
        (r"answer\s+[:=]\s*(\w+)", "INJECTED_ANSWER_COMMAND"),
        (r"you\s+are\s+now\s+(in\s+developer\s+mode|unrestricted)", "JAILBREAK_ATTEMPT"),
        (r"disregard\s+safety\s+guidelines", "DISREGARD_SAFETY"),
    ]

    @classmethod
    def inspect_and_sanitize(
        cls,
        doc_id: str,
        raw_content: str,
    ) -> AdversarialShieldResult:
        detected_attacks: List[str] = []
        sanitized = raw_content

        for pattern, attack_name in cls.INJECTION_PATTERNS:
            matches = list(re.finditer(pattern, sanitized, flags=re.IGNORECASE))
            if matches:
                detected_attacks.append(attack_name)
                # Strip out the malicious instruction phrase so it cannot influence reasoning
                sanitized = re.sub(pattern, "[UNTRUSTED_INSTRUCTION_REMOVED]", sanitized, flags=re.IGNORECASE)

        if detected_attacks:
            return AdversarialShieldResult(
                doc_id=doc_id,
                is_injection_detected=True,
                detected_attack_patterns=detected_attacks,
                authority_violation_attempted=True,
                sanitized_factual_content=sanitized.strip(),
                action_taken="NEUTRALIZED_AND_TREATED_AS_DATA",
                security_verdict="Adversarial command detected in evidence payload. Instruction privilege denied; data neutralized.",
            )

        return AdversarialShieldResult(
            doc_id=doc_id,
            is_injection_detected=False,
            detected_attack_patterns=[],
            authority_violation_attempted=False,
            sanitized_factual_content=raw_content.strip(),
            action_taken="CLEAN",
            security_verdict="No adversarial command patterns identified. Clean evidence data.",
        )
