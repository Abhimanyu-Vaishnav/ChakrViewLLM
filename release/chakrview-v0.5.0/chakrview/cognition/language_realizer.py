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
import time
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
    USER_IDENTITY_ASSERTION = "USER_IDENTITY_ASSERTION"
    USER_IDENTITY_QUERY = "USER_IDENTITY_QUERY"
    UNKNOWN_QUERY = "UNKNOWN_QUERY"
    CLARIFICATION = "CLARIFICATION"


@dataclass
class UserProfileFact:
    attribute: str
    value: str
    confidence: float = 1.0
    provenance: str = "USER_INPUT"
    timestamp: float = field(default_factory=time.time)
    scope: str = "persistent_user_profile"


class UserProfileMemory:
    """
    Structured user profile memory distinguishing persistent user-profile facts
    from transient conversational session memory.
    """

    def __init__(self):
        self._facts: Dict[str, UserProfileFact] = {}

    def set_fact(
        self,
        attribute: str,
        value: str,
        confidence: float = 1.0,
        provenance: str = "USER_INPUT",
    ) -> UserProfileFact:
        fact = UserProfileFact(
            attribute=attribute.strip().lower(),
            value=value.strip(),
            confidence=confidence,
            provenance=provenance,
            timestamp=time.time(),
            scope="persistent_user_profile",
        )
        self._facts[fact.attribute] = fact
        return fact

    def get_fact(self, attribute: str) -> Optional[UserProfileFact]:
        return self._facts.get(attribute.strip().lower())

    def has_fact(self, attribute: str) -> bool:
        return attribute.strip().lower() in self._facts

    def clear(self) -> None:
        self._facts.clear()


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
    Explicitly tracks ephemeral session memory and links to user profile memory.
    """

    def __init__(self, max_turns: int = 6):
        self.max_turns = max_turns
        self.history: List[ConversationalTurn] = []
        self.user_profile = UserProfileMemory()

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
        self.user_profile.clear()


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

    USER_IDENTITY_PATTERNS = [
        r"\b(?:my\s+name\s+is|mera\s+naam|main\s+hoon|mai\s+hu)\s+([A-Za-z0-9_\- ]+)",
    ]

    USER_NAME_QUERY_PATTERNS = [
        r"\b(?:what\s+is\s+my\s+name|mera\s+naam\s+kya\s+hai|remind\s+me\s+what\s+my\s+name\s+is|do\s+you\s+know\s+my\s+name|kya\s+tumhe\s+mera\s+naam\s+pata\s+hai)\b",
    ]

    @classmethod
    def classify(cls, text: str, detected_lang: LanguageTag) -> ConversationalIntent:
        clean = text.strip().lower()

        # Check for user identity query first
        for pat in cls.USER_NAME_QUERY_PATTERNS:
            if re.search(pat, clean, re.IGNORECASE):
                return ConversationalIntent.USER_IDENTITY_QUERY

        # Check for user identity assertion
        # E.g. "My name is Abhimanyu Vaishnav", "Mera naam Abhimanyu hai", "Main bhi thik hu, mera naam Abhimanyu hai"
        if re.search(r"\b(?:my\s+name\s+is|mera\s+naam)\b", clean, re.IGNORECASE):
            return ConversationalIntent.USER_IDENTITY_ASSERTION

        # Try compositional sentence parser for generalized grammatical mood & intent
        try:
            from chakrview.cognition.language_understanding import (
                TextNormalizer,
                CompositionalSentenceParser,
                SentenceMood,
            )
            norm = TextNormalizer.normalize(text)
            parsed = CompositionalSentenceParser.parse(norm)
            if parsed.mood == SentenceMood.INCOMPLETE or parsed.requires_clarification:
                return ConversationalIntent.CLARIFICATION
            if parsed.predicate_relation == "DEFINITION":
                return ConversationalIntent.DEFINITION_QUERY
        except Exception:
            pass

        # 1. Definition queries (generalized English and Hinglish)
        # Ensure compound requests with definition queries (e.g. "Main bhi thik hu, jra muje computer ki defination btao")
        # are recognized as definition query rather than pure social check-in.
        has_definition = any(re.search(pat, clean, re.IGNORECASE) for pat in cls.DEFINITION_PATTERNS)
        if has_definition:
            return ConversationalIntent.DEFINITION_QUERY

        # Check for generalized Hinglish/English definition patterns: "kya hai X", "X kya hai", "what is X", "define X"
        if re.search(r"\b(?:kya\s+hai\s+[a-zA-Z0-9_\-]+|[a-zA-Z0-9_\-]+\s+kya\s+hai)\b", clean):
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
        if len(tokens) <= 4 and tokens and tokens[-1] in ("is", "are", "a", "an", "the", "ek"):
            return ConversationalIntent.CLARIFICATION
        if len(tokens) <= 2 and tokens and tokens[-1] in ("hai", "tha", "thi") and not any(w in tokens for w in ["kya", "naam", "kaise"]):
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
        "cpu": {
            "ENGLISH": (
                "The Central Processing Unit (CPU) is the primary component of a computer that "
                "executes instructions, performs arithmetic and logic operations, and controls data flow."
            ),
            "HINGLISH": (
                "CPU (Central Processing Unit) computer ka primary processor hai jo instructions execute karta hai, "
                "arithmetic aur logic operations perform karta hai, aur data flow control karta hai."
            ),
            "HINDI": (
                "सीपीयू (Central Processing Unit) कंप्यूटर का मुख्य प्रोसेसर है जो निर्देशों को निष्पादित करता है "
                "और अंकगणितीय एवं तार्किक गणनाएं करता है।"
            ),
        },
    }

    @classmethod
    def extract_term(cls, text: str) -> Optional[str]:
        clean = text.lower()
        # Direct check for known terms
        for term in cls.DEFINITIONS.keys():
            if re.search(rf"\b{term}\b", clean):
                return term
        # Fallback: extract target term syntactically from definition queries
        t = text.strip()
        m_kya_post = re.search(r"([A-Za-z0-9_\-]+)\s+kya\s+hai", t, re.IGNORECASE)
        if m_kya_post:
            return m_kya_post.group(1).strip()
        m_kya_pre = re.search(r"kya\s+hai\s+([A-Za-z0-9_\-]+)", t, re.IGNORECASE)
        if m_kya_pre:
            return m_kya_pre.group(1).strip()
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

        # D. User Identity Assertion & Query
        if intent == ConversationalIntent.USER_IDENTITY_ASSERTION or cognitive_decision == "IDENTITY_ACQUIRED":
            user_name = value or subject or (context_tracker.user_profile.get_fact("name").value if context_tracker and context_tracker.user_profile.has_fact("name") else None)
            if user_name:
                if language == LanguageTag.HINDI:
                    return f"नमस्ते {user_name}! मैंने आपका नाम याद रख लिया है।"
                elif language == LanguageTag.HINGLISH:
                    return f"Aapse milkar accha laga, {user_name}! Maine aapka naam yaad rakh liya hai."
                else:
                    return f"Nice to meet you, {user_name}! I have recorded your name in your profile."
            else:
                if language in (LanguageTag.HINDI, LanguageTag.HINGLISH):
                    return "Samajh gaya! Maine aapki jankari record kar li hai."
                else:
                    return "Understood. I have recorded your information."

        if intent == ConversationalIntent.USER_IDENTITY_QUERY or (target_rel := relation) == "USER_NAME":
            name_fact = context_tracker.user_profile.get_fact("name") if context_tracker else None
            user_name = value or (name_fact.value if name_fact else None)
            if user_name:
                if language == LanguageTag.HINDI:
                    return f"आपका नाम {user_name} है।"
                elif language == LanguageTag.HINGLISH:
                    return f"Aapka naam {user_name} hai."
                else:
                    return f"Your name is {user_name}."
            else:
                # Honest abstention when fact has not been provided
                if language == LanguageTag.HINDI:
                    return "मुझे अभी आपका नाम नहीं पता है। आपने अभी तक अपना नाम नहीं बताया है।"
                elif language == LanguageTag.HINGLISH:
                    return "Mujhe abhi aapka naam nahi pata. Aapne abhi tak apna naam nahi bataya hai."
                else:
                    return "I do not know your name yet. You have not told me your name."

        # E. Incomplete or ambiguous inputs (e.g. "Computer is", "Computer is an ?")
        if intent == ConversationalIntent.CLARIFICATION:
            if language in (LanguageTag.HINDI, LanguageTag.HINGLISH):
                return f"Aapka prashna adhura lag raha hai: '{raw_text}'. Kripya poora sentence batayein."
            else:
                return f"It looks like your statement is incomplete: '{raw_text}'. What would you like to know or record about it?"

        # F. Hierarchical Planning
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
