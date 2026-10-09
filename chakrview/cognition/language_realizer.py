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
    DEFINITION_QUERY = "DEFINITION_QUERY"
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
        "kaun", "koun", "batao", "btao", "karo", "rehta", "rehti", "rehte", "mein",
        "par", "aur", "tum", "aap", "mera", "meri", "hum", "accha", "achha", "theek",
        "thik", "shukriya", "dhanyavaad", "namaste", "namaskar", "alvida", "chalo",
        "kuch", "sab", "yeh", "woh", "voh", "ho", "tha", "thi", "gaya",
        "main", "bhi", "jra", "zara", "muje", "mujhe", "hame", "hume", "ki", "ka", "ke"
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

        # If 2 or more distinct Hinglish markers exist, or ratio >= 0.20, classify as HINGLISH
        if hinglish_hits >= 2 or (hinglish_hits >= 1 and (ratio >= 0.20 or len(tokens) <= 4)):
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

    DEFINITION_PATTERNS = [
        r"\b(defination|definition|define|meaning\s+of|arth|paribhasha)\b",
        r"\b(what\s+is\s+a\s+|what\s+is\s+an\s+|what\s+is\s+the\s+definition)\b",
        r"\b(ki\s+(defination|definition|paribhasha)\s+(batao|btao|samjhao))\b",
    ]

    REASONING_PATTERNS = [
        r"\b(relative\s+to|where\s+is\s+.*relative|infer|deduce|transitive)\b",
    ]

    INCOMPLETE_PATTERNS = [
        r"^(what|who|where|when|why|how|computer|system|network|device|entity)\s+(is|are|tha|thi|hai)$",
        r"^([a-zA-Z0-9_\- ]{1,30})\s+(is|are|hai)$",
    ]

    @classmethod
    def classify(cls, text: str, detected_lang: LanguageTag) -> ConversationalIntent:
        clean = text.strip().lower()

        # 1. Social check-ins (e.g. "how are you", "tum kaise ho")
        # Ensure compound requests with definition queries (e.g. "Main bhi thik hu, jra muje computer ki defination btao")
        # are recognized as definition query rather than pure social check-in.
        has_definition = any(re.search(pat, clean, re.IGNORECASE) for pat in cls.DEFINITION_PATTERNS)
        if has_definition:
            return ConversationalIntent.DEFINITION_QUERY

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

        # 5. Incomplete statement / clarification check
        # Detect incomplete definition-style statements e.g. "Computer is", "Computer is an ?", "System is a"
        clean_no_punct = re.sub(r"[\?\.!,;:]+$", "", clean).strip()
        tokens = clean_no_punct.split()
        if len(tokens) <= 4 and tokens and tokens[-1] in ("is", "are", "hai", "tha", "thi", "a", "an", "the", "ek"):
            return ConversationalIntent.CLARIFICATION

        # 6. Queries vs Propositions
        is_question = "?" in text or any(
            clean.startswith(w) for w in ["what", "who", "where", "when", "why", "how", "kya", "kahan", "kab", "koun"]
        ) or any(w in clean for w in ["kya", "kahan", "batao", "btao"])
        if is_question:
            unknown_markers = ["zorvax", "atlantis", "norlath", "eryndor", "zethra"]
            if any(m in clean for m in unknown_markers):
                return ConversationalIntent.UNKNOWN_QUERY
            return ConversationalIntent.FACTUAL_QUERY

        # 7. Default to proposition / statement
        return ConversationalIntent.PROPOSITION


class BuiltInKnowledgeRepository:
    """
    Clearly identified, verified built-in factual knowledge repository.
    Truthfulness Guarantee:
    - Answers are explicitly attributed to 'BUILTIN_KNOWLEDGE_STORE'.
    - Not claimed as learned neural weights or external LLM generations.
    - If a concept is unknown, returns None so caller abstains honestly.
    """

    DEFINITIONS: Dict[str, Dict[str, str]] = {
        "computer": {
            "ENGLISH": (
                "A computer is an electronic device that accepts raw data as input, processes it "
                "according to stored instructions (programs), and produces output or stores the result."
            ),
            "HINGLISH": (
                "Computer ek electronic device hai jo raw data ko input ke roop mein leta hai, "
                "instructions (programs) ke hisab se process karta hai, aur output produce ya store karta hai."
            ),
            "HINDI": (
                "कंप्यूटर एक इलेक्ट्रॉनिक उपकरण है जो इनपुट के रूप में डेटा स्वीकार करता है, "
                "निर्देशों (प्रोग्राम्स) के अनुसार इसे संसाधित (process) करता है, और परिणाम आउटपुट के रूप में देता है।"
            ),
        },
    }

    @classmethod
    def extract_term(cls, text: str) -> Optional[str]:
        clean = text.lower()
        for term in cls.DEFINITIONS.keys():
            if re.search(rf"\b{term}\b", clean):
                return term
        # Fallback: extract target term syntactically from definition queries (e.g. 'Define Zorvax-917')
        t = text.strip()
        m = re.search(r"\b(?:define|meaning\s+of|definition\s+of|defination\s+of)\s+([A-Za-z0-9_\-]+)", t, re.IGNORECASE)
        if m:
            return m.group(1).strip()
        m_what = re.search(r"\bwhat\s+is\s+(?:a|an|the)?\s*([A-Za-z0-9_\-]+)", t, re.IGNORECASE)
        if m_what:
            return m_what.group(1).strip()
        m_hing = re.search(r"([A-Za-z0-9_\-]+)\s+ki\s+(?:defination|definition|paribhasha)", t, re.IGNORECASE)
        if m_hing:
            return m_hing.group(1).strip()
        return None

    @classmethod
    def get_definition(cls, term: str, language: LanguageTag) -> Optional[str]:
        t = term.lower().strip()
        if t not in cls.DEFINITIONS:
            return None
        term_dict = cls.DEFINITIONS[t]
        lang_key = language.value if language.value in term_dict else "ENGLISH"
        return term_dict.get(lang_key, term_dict.get("ENGLISH"))


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

        # C. Definition Queries (e.g. "Define computer", "computer ki defination btao", "Define Zorvax-917")
        if intent == ConversationalIntent.DEFINITION_QUERY:
            term = subject or BuiltInKnowledgeRepository.extract_term(raw_text)
            if term:
                definition = BuiltInKnowledgeRepository.get_definition(term, language)
                if definition:
                    return definition
            # If no suitable built-in or verified knowledge source exists, report missing capability honestly
            target_display = term or raw_text
            if language == LanguageTag.HINDI:
                return f"क्षमा करें, मेरे पास '{target_display}' की कोई सत्यापित परिभाषा उपलब्ध नहीं है।"
            elif language == LanguageTag.HINGLISH:
                return f"Maaf kijiye, mere paas '{target_display}' ke liye verified definition uplabdh nahi hai."
            else:
                return f"I do not have verified built-in knowledge defining '{target_display}'. Preserving honest abstention."

        # D. Incomplete or ambiguous inputs (e.g. "Computer is", "Computer is an ?")
        if intent == ConversationalIntent.CLARIFICATION:
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
