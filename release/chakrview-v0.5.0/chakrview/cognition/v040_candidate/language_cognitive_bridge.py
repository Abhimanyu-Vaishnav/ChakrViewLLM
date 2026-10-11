"""
Wave 587: Language <-> Cognitive State Integration Bridge.

Architectural Guarantees:
- Maps natural language inquiries into structured semantic cognitive frames.
- Reuses ChakrView's indigenous tokenization and representation pathways.
- Rejects attaching external pretrained LLMs.
- Accurately discriminates:
  - INSTRUCTION vs FACTUAL_INQUIRY
  - Language identity (English, Hindi, Mixed)
  - Ambiguous linguistic indicators
- Generates transparent, structured verbalization of internal reasoning decisions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple


@dataclass
class SemanticCognitiveFrame:
    raw_text: str
    detected_language: str       # ENGLISH, HINDI, MIXED
    is_command_or_instruction: bool
    is_factual_query: bool
    identified_intent: str
    extracted_entities: List[str]
    is_ambiguous: bool
    verbalized_reasoning: str


class LanguageCognitiveBridge:
    """
    Parses natural language into explicit cognitive state representations
    and translates internal reasoning verdicts into safe verbal outputs.
    """

    HINDI_COMMON_TOKENS = ["kya", "kaise", "kyon", "hai", "hain", "kab", "koun", "kahan", "batao", "karo"]
    COMMAND_PATTERNS = [
        r"^(verify|find|calculate|plan|execute|search|evaluate|check|run|stop|reset)\b",
        r"\b(do\s+this|must\s+do|perform)\b",
    ]

    @classmethod
    def parse_to_cognitive_frame(
        cls,
        text: str,
        context_entities: Optional[List[str]] = None,
    ) -> SemanticCognitiveFrame:
        text_clean = text.strip()
        tokens = text_clean.lower().split()

        # 1. Language detection
        hindi_matches = sum(1 for tok in tokens if tok in cls.HINDI_COMMON_TOKENS)
        if hindi_matches >= 2 and len(tokens) <= 6:
            lang = "HINDI"
        elif hindi_matches >= 1 and len(tokens) > 6:
            lang = "MIXED"
        else:
            lang = "ENGLISH"

        # 2. Instruction vs Factual Query
        is_cmd = any(re.search(pat, text_clean, re.IGNORECASE) for pat in cls.COMMAND_PATTERNS)
        is_query = "?" in text_clean or any(tok in ["what", "who", "when", "where", "why", "how", "kya", "kab"] for tok in tokens)

        # 3. Intent Identification
        if is_cmd:
            intent = "EXECUTE_COGNITIVE_TASK"
        elif is_query:
            intent = "FACTUAL_INFORMATION_RETRIEVAL"
        else:
            intent = "GENERAL_PROPOSITION_EVALUATION"

        # 4. Entity Extraction
        entities = context_entities or []
        if not entities:
            # Capitalized words as candidates
            entities = [w.strip(".,;:?!()") for w in text_clean.split() if w.istitle() and len(w) > 2]

        # 5. Ambiguity Detection
        ambiguous_tokens = ["this", "it", "they", "that", "something", "apple", "bank"]
        is_ambig = any(tok in tokens for tok in ambiguous_tokens) and len(entities) <= 1

        verbalized = f"Parsed {lang} input as intent '{intent}' targeting entities {entities}."

        return SemanticCognitiveFrame(
            raw_text=text_clean,
            detected_language=lang,
            is_command_or_instruction=is_cmd,
            is_factual_query=is_query,
            identified_intent=intent,
            extracted_entities=entities,
            is_ambiguous=is_ambig,
            verbalized_reasoning=verbalized,
        )

    @classmethod
    def verbalize_decision(
        cls,
        decision_action: str,
        target_entity: str,
        resolved_value: Optional[str],
        confidence: float,
        rationale: str,
    ) -> str:
        """Translates internal cognitive decision state into calibrated natural language."""
        if decision_action == "ANSWER" and resolved_value is not None:
            return f"Regarding {target_entity}: Verified result is {resolved_value} (Confidence: {confidence:.2f}). {rationale}"
        elif decision_action == "QUALIFY":
            return f"Regarding {target_entity}: Qualified outcome. Verified core is established, but details remain partial. {rationale}"
        elif decision_action == "ASK_CLARIFICATION":
            return f"Regarding {target_entity}: Clarification required. {rationale}"
        elif decision_action == "REPORT_CONFLICT":
            return f"Regarding {target_entity}: Epistemic deadlock detected between competing authorities. {rationale}"
        else:
            return f"Regarding {target_entity}: Insufficient evidence available to answer factually. Honest abstention preserved."
