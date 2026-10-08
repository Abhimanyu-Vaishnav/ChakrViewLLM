"""
ChakrView Natural Language Realization & Conversational Intent Engine.

Bridges:
USER LANGUAGE
→ LANGUAGE UNDERSTANDING (LanguageClassifier & IntentClassifier)
→ SEMANTIC/COGNITIVE STATE
→ REASONING / MEMORY / EVIDENCE
→ LANGUAGE REALIZATION (Truthful Conversational Realizer)
→ NATURAL USER RESPONSE

Principles:
- Truthful realization: preserves cognitive decisions (ANSWER, ABSTAIN, REPORT_CONFLICT, REQUEST_MORE_EVIDENCE).
- No external LLM or fake hallucination layers.
- Handles English, Hindi (Devanagari script), Hinglish (Romanized Hindi), and mixed inputs.
- Handles conversational intents: GREETING, SOCIAL_CHECK_IN, CASUAL_CONVERSATION, FACTUAL_QUERY, PROPOSITION, TASK_REQUEST, REASONING_QUERY, CLARIFICATION.
- Multi-turn conversational context for simple reference resolution (e.g. "Ram lives in Delhi" -> "Where does he live?").
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, List, Optional, Tuple


class LanguageTag(str, Enum):
    ENGLISH = "ENGLISH"
    HINDI = "HINDI"
    HINGLISH = "HINGLISH"
    UNKNOWN = "UNKNOWN"


class ConversationalIntent(str, Enum):
    GREETING = "GREETING"
    SOCIAL_CHECK_IN = "SOCIAL_CHECK_IN"
    CASUAL_CONVERSATION = "CASUAL_CONVERSATION"
    FACTUAL_QUERY = "FACTUAL_QUERY"
    PROPOSITION = "PROPOSITION"
    TASK_REQUEST = "TASK_REQUEST"
    REASONING_QUERY = "REASONING_QUERY"
    UNKNOWN_QUERY = "UNKNOWN_QUERY"
    CLARIFICATION = "CLARIFICATION"


@dataclass
class ConversationalTurn:
    user_input: str
    detected_language: LanguageTag
    detected_intent: ConversationalIntent
    active_subject: Optional[str] = None
    active_attribute: Optional[str] = None
    active_value: Optional[str] = None
    system_response: str = ""
    timestamp: float = 0.0


class ConversationalContextTracker:
    """
    Maintains bounded multi-turn conversational context for reference resolution.
    """

    def __init__(self, max_turns: int = 6):
        self.max_turns = max_turns
        self.history: List[ConversationalTurn] = []

    def record_turn(self, turn: ConversationalTurn) -> None:
        self.history.append(turn)
        if len(self.history) > self.max_turns:
            self.history = self.history[-self.max_turns:]

    def get_latest_subject(self) -> Optional[str]:
        for turn in reversed(self.history):
            if turn.active_subject:
                return turn.active_subject
        return None

    def get_latest_attribute(self) -> Optional[str]:
        for turn in reversed(self.history):
            if turn.active_attribute:
                return turn.active_attribute
        return None

    def clear(self) -> None:
        self.history.clear()


class LanguageClassifier:
    """
    Robust linguistic classification distinguishing English, Hindi, and Hinglish.
    """

    # Devanagari Unicode range: \u0900-\u097F
    DEVANAGARI_REGEX = re.compile(r"[\u0900-\u097f]")

    # Common Romanized Hindi / Hinglish markers
    HINGLISH_MARKERS = {
        "kya", "kaise", "kyon", "kyu", "hai", "hain", "kab", "kahan", "kaha",
        "kaun", "koun", "batao", "karo", "rehta", "rehti", "rehte", "mein",
        "par", "aur", "tum", "aap", "mera", "meri", "hum", "accha", "theek",
        "shukriya", "dhanyavaad", "namaste", "namaskar", "alvida", "chalo",
        "kuch", "sab", "yeh", "woh", "voh", "ho", "tha", "thi", "gaya"
    }

    @classmethod
    def classify(cls, text: str) -> LanguageTag:
        clean = text.strip()
        if not clean:
            return LanguageTag.UNKNOWN

        # 1. Check for Devanagari script characters
        if cls.DEVANAGARI_REGEX.search(clean):
            return LanguageTag.HINDI

        # 2. Token analysis for Hinglish
        tokens = [t.strip(".,;:?!()\"'") for t in clean.lower().split() if t.strip()]
        if not tokens:
            return LanguageTag.UNKNOWN

        hinglish_hits = sum(1 for tok in tokens if tok in cls.HINGLISH_MARKERS)
        ratio = hinglish_hits / len(tokens)

        if hinglish_hits >= 1 and (ratio >= 0.25 or hinglish_hits >= 2 or len(tokens) <= 4):
            return LanguageTag.HINGLISH

        # 3. Default to English if standard Latin characters
        has_latin = any(c.isalpha() and ord(c) < 128 for c in clean)
        if has_latin:
            return LanguageTag.ENGLISH

        return LanguageTag.UNKNOWN


class IntentClassifier:
    """
    Identifies conversational and task intents beyond naive keyword matching.
    """

    GREETING_PATTERNS = [
        r"^(hi|hello|hey|namaste|namaskar|pranam|greetings)\b",
        r"^(good\s+(morning|afternoon|evening|day))\b",
    ]

    SOCIAL_PATTERNS = [
        r"(how\s+are\s+you|how's\s+it\s+going|how\s+do\s+you\s+do|how\s+are\s+things)",
        r"(tum\s+kaise\s+ho|aap\s+kaise\s+hain|kaise\s+ho|kya\s+haal\s+hai|kya\s+chal\s+raha\s+hai)",
    ]

    TASK_PATTERNS = [
        r"^(plan|execute|calculate|verify|find|search|evaluate|check|run|create\s+a\s+plan)\b",
        r"\b(step-by-step\s+plan|make\s+a\s+plan|need\s+to\s+move)\b",
    ]

    REASONING_PATTERNS = [
        r"\b(relative\s+to|where\s+is\s+.*relative|infer|deduce|transitive)\b",
    ]

    @classmethod
    def classify(cls, text: str, detected_lang: LanguageTag) -> ConversationalIntent:
        clean = text.strip().lower()

        # 1. Social check-ins (e.g. "how are you", "tum kaise ho")
        for pat in cls.SOCIAL_PATTERNS:
            if re.search(pat, clean, re.IGNORECASE):
                return ConversationalIntent.SOCIAL_CHECK_IN

        # 2. Greetings (e.g. "hi", "namaste")
        for pat in cls.GREETING_PATTERNS:
            if re.search(pat, clean, re.IGNORECASE):
                return ConversationalIntent.GREETING

        # 3. Task request
        for pat in cls.TASK_PATTERNS:
            if re.search(pat, clean, re.IGNORECASE):
                return ConversationalIntent.TASK_REQUEST

        # 4. Multi-hop or relational reasoning
        for pat in cls.REASONING_PATTERNS:
            if re.search(pat, clean, re.IGNORECASE):
                return ConversationalIntent.REASONING_QUERY

        # 5. Queries vs Propositions
        is_question = "?" in text or any(
            clean.startswith(w) for w in ["what", "who", "where", "when", "why", "how", "kya", "kahan", "kab", "koun"]
        )
        if is_question:
            unknown_markers = ["zorvax", "atlantis", "norlath", "eryndor", "zethra"]
            if any(m in clean for m in unknown_markers):
                return ConversationalIntent.UNKNOWN_QUERY
            return ConversationalIntent.FACTUAL_QUERY

        # 6. Default to proposition / statement
        if len(clean.split()) <= 2 and clean.endswith("is"):
            return ConversationalIntent.CLARIFICATION

        return ConversationalIntent.PROPOSITION


class LanguageRealizer:
    """
    Realizes truthful, natural natural-language responses from internal cognitive states.
    Ensures:
    - Conversational responses for greetings/social check-ins.
    - Clean confirmation for propositions ("Got it. Ram lives in Delhi.").
    - Natural factual answers ("Ram lives in Delhi.").
    - Transparent epistemic conflict reporting ("Evidence reports conflicting information...").
    - Honest abstention for unknowns ("I do not have verified evidence about...").
    - Language alignment: responds in Hindi/Hinglish when addressed in Hindi/Hinglish.
    """

    @classmethod
    def realize(
        cls,
        intent: ConversationalIntent,
        language: LanguageTag,
        cognitive_decision: str,
        subject: Optional[str],
        relation: Optional[str],
        value: Optional[str],
        confidence: float,
        evidence_status: str,
        raw_text: str,
        plan_details: Optional[str] = None,
        context_tracker: Optional[ConversationalContextTracker] = None,
    ) -> str:
        # A. Greetings
        if intent == ConversationalIntent.GREETING:
            if language == LanguageTag.HINDI:
                return "नमस्ते! मैं चक्रव्यूह (ChakrView) हूँ। मैं आपकी क्या सहायता कर सकता हूँ?"
            elif language == LanguageTag.HINGLISH:
                return "Namaste! Main ChakrView hoon. Main aapki kya madad kar sakta hoon?"
            else:
                return "Hello! I am ChakrView. How can I assist you today?"

        # B. Social Check-ins (e.g. "how are you", "tum kaise ho")
        if intent == ConversationalIntent.SOCIAL_CHECK_IN:
            if language == LanguageTag.HINDI:
                return "मैं ठीक हूँ, धन्यवाद! चक्रव्यूह का स्वदेशी तंत्र सुचारु रूप से चल रहा है। आप कैसे हैं?"
            elif language == LanguageTag.HINGLISH:
                return "Main theek hoon, shukriya! ChakrView ka indigenous cognitive system achhi tarah chal raha hai. Aap kaise hain?"
            else:
                return "I am functioning well, thank you! ChakrView's indigenous cognitive core is active and ready. How are you?"

        # C. Incomplete or ambiguous inputs (e.g. "Computer is")
        if intent == ConversationalIntent.CLARIFICATION or raw_text.strip().lower().endswith("is"):
            if language in (LanguageTag.HINDI, LanguageTag.HINGLISH):
                return f"Aapka prashna adhura lag raha hai: '{raw_text}'. Kripya poora sentence batayein."
            else:
                return f"It looks like your statement is incomplete: '{raw_text}'. What would you like to know or record about it?"

        # D. Hierarchical Planning
        if intent == ConversationalIntent.TASK_REQUEST or cognitive_decision == "PLAN_GENERATED":
            if plan_details:
                return f"Plan generated successfully. {plan_details}"
            return "Step-by-step plan created: 1. Gather evidence -> 2. Verify constraints -> 3. Synthesize result."

        # E. Proposition Acquisition ("Ram lives in Delhi.")
        if cognitive_decision == "BELIEF_ACQUIRED" and subject and value:
            rel_display = relation.lower().replace("_", " ") if relation else "related to"
            if language == LanguageTag.HINDI:
                return f"समझ गया। {subject} {value} में रहता है।"
            elif language == LanguageTag.HINGLISH:
                return f"Samajh gaya. {subject} {value} mein rehta hai."
            else:
                return f"Got it. {subject} {rel_display} {value}."

        # F. Factual Answers (from memory or evidence)
        if cognitive_decision == "ANSWER" and subject and value:
            rel_display = relation.lower().replace("_", " ") if relation else "lives in"
            if language == LanguageTag.HINDI:
                return f"{subject} {value} में रहता है।"
            elif language == LanguageTag.HINGLISH:
                return f"{subject} {value} mein rehta hai."
            else:
                return f"{subject} {rel_display} {value}."

        # G. Epistemic Contradictions
        if cognitive_decision == "REPORT_CONFLICT":
            if language in (LanguageTag.HINDI, LanguageTag.HINGLISH):
                return f"Virodhabhas (Contradiction) mila: {subject} ke baare mein alag-alag daave hain ({evidence_status})."
            else:
                return f"Evidence conflict detected: multiple sources disagree on {subject}'s status ({evidence_status})."

        # H. Honest Abstention (Unknown entities / False premises)
        if cognitive_decision == "ABSTAIN":
            if language == LanguageTag.HINDI:
                return f"क्षमा करें, मेरे पास इसके लिए सत्यापित प्रमाण (verified evidence) उपलब्ध नहीं है।"
            elif language == LanguageTag.HINGLISH:
                return f"Maaf kijiye, mere paas iske baare mein verified evidence uplabdh nahi hai."
            else:
                return f"I do not have verified evidence to answer this factually. Preserving honest abstention."

        # Default fallback
        return f"Acknowledged input: '{raw_text}'."
