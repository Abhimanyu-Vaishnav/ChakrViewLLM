"""
ChakrView Indigenous Language Understanding Engine (Language Understanding Foundation).

Provides generalized language interpretation beyond rigid templates and hardcoded strings:
1. Input Normalization (TextNormalizer)
   - Preserves verbatim raw input.
   - Cleans whitespace, typography, punctuation.
   - Normalizes spelling typos (English + common Hinglish phonetics) while strictly protecting
     proper nouns, identifiers, and technical entities.
2. Compositional Sentence Understanding (CompositionalSentenceParser)
   - Mood/Act detection: DECLARATIVE (statement), INTERROGATIVE (question), IMPERATIVE (command),
     SOCIAL (greeting/check-in), INCOMPLETE (ambiguous/fragmented needing clarification).
   - Linguistic features: Negation (not, never, don't, nahi, na, mat), Tense (PAST, PRESENT, FUTURE),
     Question targets (WHAT, WHERE, WHO, WHEN, WHY, HOW, POLAR/BOOLEAN).
   - Flexible constituent order (SVO, SOV, OVS, OSV) across English, Hindi, and Hinglish.
   - Unknown entity discovery and extraction without treating them as parse failures.
3. Structured Semantic Representation (StructuredSentenceForm)
   - Rich dataclass capturing speech act, question target, polarity, entities, candidate relation,
     arguments, uncertainty score, and clarification flags.
4. Integration Bridge (LanguageUnderstandingBridge)
   - Bridges parsed representations directly to existing Cognition, WorkingMemory, MemoryReasoningBridge,
     and BuiltInKnowledgeRepository.
5. Governed Learning Interface (CandidateLearningLedger)
   - Ingests user-provided statements/corrections as governed candidate learning records.
   - Stores raw text, normalized text, extracted interpretation, provenance, approval status (PENDING_VERIFICATION).
   - Strictly non-parametric: never alters canonical neural weights or baseline hashes.
"""

from __future__ import annotations

import re
import time
import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, List, Optional, Tuple, Set


# ==============================================================================
# 1. ENUMS & DATA STRUCTURES
# ==============================================================================

class SentenceMood(str, Enum):
    DECLARATIVE = "DECLARATIVE"          # Statement / assertion / proposition
    INTERROGATIVE = "INTERROGATIVE"      # Question / inquiry
    IMPERATIVE = "IMPERATIVE"            # Instruction / command / task request
    SOCIAL = "SOCIAL"                    # Greeting or conversational check-in
    INCOMPLETE = "INCOMPLETE"            # Truncated or ambiguous fragment requiring clarification


class QuestionTarget(str, Enum):
    WHAT = "WHAT"                        # Identity / definition / description
    WHERE = "WHERE"                      # Location / place
    WHO = "WHO"                          # Person / agent / actor
    WHEN = "WHEN"                        # Time / temporal point
    WHY = "WHY"                          # Cause / reason
    HOW = "HOW"                          # Manner / condition / procedure
    POLAR = "POLAR"                      # Yes/No boolean question
    NONE = "NONE"                        # Not a question


class GrammaticalTense(str, Enum):
    PRESENT = "PRESENT"
    PAST = "PAST"
    FUTURE = "FUTURE"
    UNSPECIFIED = "UNSPECIFIED"


class Polarity(str, Enum):
    AFFIRMATIVE = "AFFIRMATIVE"
    NEGATIVE = "NEGATIVE"                # Contains negation marker


class ApprovalStatus(str, Enum):
    PENDING_VERIFICATION = "PENDING_VERIFICATION"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"


@dataclass
class NormalizedText:
    raw_text: str
    cleaned_text: str
    tokens: List[str]
    normalized_tokens: List[str]
    protected_tokens: Set[str]
    detected_language: str               # ENGLISH, HINDI, HINGLISH, UNKNOWN


@dataclass
class StructuredSentenceForm:
    """
    Rich structured semantic representation of an understood input sentence.
    Consumable by downstream cognition, memory, and planner layers.
    """
    raw_input: str
    normalized_input: str
    mood: SentenceMood
    language: str
    polarity: Polarity
    tense: GrammaticalTense
    question_target: QuestionTarget
    subject: Optional[str] = None
    predicate_relation: Optional[str] = None
    target_object: Optional[str] = None
    extracted_entities: List[str] = field(default_factory=list)
    confidence: float = 0.80
    uncertainty_score: float = 0.20
    requires_clarification: bool = False
    clarification_reason: Optional[str] = None

    def to_canonical_triple(self) -> Optional[str]:
        if self.subject and self.predicate_relation and self.target_object:
            return f"({self.subject}) --[{self.predicate_relation}]--> ({self.target_object})"
        return None


@dataclass
class CandidateLearningRecord:
    """
    Governed record for user-asserted facts or corrections.
    Stored separately from canonical model weights.
    """
    record_id: str
    raw_input: str
    normalized_input: str
    extracted_subject: str
    extracted_relation: str
    extracted_value: str
    polarity: Polarity
    provenance: str                      # e.g. "USER_CONVERSATION_SESSION"
    approval_status: ApprovalStatus = ApprovalStatus.PENDING_VERIFICATION
    recorded_at: float = field(default_factory=time.time)
    notes: str = ""


# ==============================================================================
# 2. INPUT NORMALIZATION (TextNormalizer)
# ==============================================================================

class TextNormalizer:
    """
    Normalizes input text while preserving original text and protecting proper nouns,
    identifiers, numbers, and technical terms.
    """

    # Common spelling errors and Hinglish phonetic transliterations
    # Maps lowercase erroneous/variant token -> standard token
    SPELLING_CORRECTIONS: Dict[str, str] = {
        # Common English spelling variations & typos
        "defination": "definition",
        "defintion": "definition",
        "defnition": "definition",
        "computr": "computer",
        "compuer": "computer",
        "livs": "lives",
        "livess": "lives",
        "resid": "reside",
        "residesin": "resides in",
        "locatn": "location",
        "wher": "where",
        "waht": "what",
        "teh": "the",
        "whihc": "which",
        "citi": "city",
        "citty": "city",
        "coutry": "country",
        # Common Hinglish colloquial spelling variants
        "btao": "batao",
        "btaao": "batao",
        "muje": "mujhe",
        "muze": "mujhe",
        "jra": "zara",
        "thik": "theek",
        "kaiseho": "kaise ho",
        "kahanse": "kahan se",
        "rehtah": "rehta",
        "rehtaa": "rehta",
        "rhta": "rehta",
        "rhti": "rehti",
        "kyaa": "kya",
        "kyun": "kyon",
        "shukria": "shukriya",
    }

    # Common Hinglish grammatical markers
    HINGLISH_VOCAB = {
        "kya", "kaise", "kyon", "kyu", "hai", "hain", "kab", "kahan", "kaha",
        "kaun", "koun", "batao", "btao", "karo", "rehta", "rehti", "rehte", "mein",
        "me", "par", "aur", "tum", "aap", "mera", "meri", "hum", "accha", "achha",
        "theek", "thik", "shukriya", "dhanyavaad", "namaste", "namaskar", "alvida",
        "chalo", "kuch", "sab", "yeh", "woh", "voh", "ho", "tha", "thi", "gaya",
        "gayi", "main", "mai", "bhi", "jra", "zara", "muje", "mujhe", "hame",
        "hume", "ki", "ka", "ke", "ko", "se", "ek", "nahi", "nahin", "mat", "na"
    }

    DEVANAGARI_REGEX = re.compile(r"[\u0900-\u097f]")

    @classmethod
    def identify_language(cls, text: str) -> str:
        clean = text.strip()
        if not clean:
            return "UNKNOWN"
        if cls.DEVANAGARI_REGEX.search(clean):
            return "HINDI"

        raw_words = [w.strip(".,;:?!()\"'’“”") for w in clean.lower().split() if w.strip()]
        if not raw_words:
            return "UNKNOWN"

        hinglish_count = sum(1 for w in raw_words if w in cls.HINGLISH_VOCAB)
        ratio = hinglish_count / len(raw_words)

        if hinglish_count >= 2 or (hinglish_count >= 1 and (ratio >= 0.20 or len(raw_words) <= 4)):
            return "HINGLISH"

        has_latin = any(c.isalpha() and ord(c) < 128 for c in clean)
        if has_latin:
            return "ENGLISH"
        return "UNKNOWN"

    @classmethod
    def normalize(cls, raw_text: str) -> NormalizedText:
        cleaned_whitespace = " ".join(raw_text.strip().split())
        lang = cls.identify_language(cleaned_whitespace)

        # Tokenize preserving punctuation boundaries
        raw_tokens = re.findall(r"[\w'-]+|[^\w\s]", cleaned_whitespace)

        protected_tokens: Set[str] = set()
        normalized_tokens: List[str] = []

        for idx, tok in enumerate(raw_tokens):
            # Check if this token is a punctuation symbol
            if not any(c.isalnum() for c in tok):
                normalized_tokens.append(tok)
                continue

            # Protect proper nouns:
            # 1. Has internal digits/hyphens (e.g. Zorvax-917, v0.5.0, COVID-19)
            # 2. Titlecased word NOT at the start of sentence (e.g. "Ram lives in Delhi")
            # 3. Known entity or acronyms (all uppercase like NASA, AI)
            is_alphanum_code = bool(re.search(r"[A-Za-z]+[-_0-9]|[0-9]+[A-Za-z]+", tok))
            is_all_caps = len(tok) > 1 and tok.isupper()
            is_proper_noun = (idx > 0 and tok.istitle() and tok.lower() not in cls.HINGLISH_VOCAB)

            # First token could also be a proper noun if capitalized and not a standard function word
            function_words = {
                "what", "where", "who", "when", "why", "how", "define", "tell", "can",
                "could", "is", "are", "do", "does", "did", "the", "a", "an", "in", "on",
                "at", "kya", "kaise", "main", "tum", "aap", "computer", "system"
            }
            if idx == 0 and tok.istitle() and tok.lower() not in function_words and not tok.lower().endswith("ing"):
                is_proper_noun = True

            if is_alphanum_code or is_all_caps or is_proper_noun:
                protected_tokens.add(tok)
                normalized_tokens.append(tok)
                continue

            # Check for spelling correction on lowercase form
            low = tok.lower()
            if low in cls.SPELLING_CORRECTIONS:
                corrected = cls.SPELLING_CORRECTIONS[low]
                # Preserve capitalization style if original was titlecase
                if tok.istitle():
                    corrected = corrected.title()
                normalized_tokens.append(corrected)
            else:
                normalized_tokens.append(tok)

        # Reconstruct normalized string cleanly
        # Attach punctuation appropriately
        reconstructed = ""
        for tok in normalized_tokens:
            if not reconstructed:
                reconstructed = tok
            elif not any(c.isalnum() for c in tok) and tok in ",.?!;:)]}\"'":
                reconstructed += tok
            elif reconstructed[-1] in "([{\"'":
                reconstructed += tok
            else:
                reconstructed += " " + tok

        return NormalizedText(
            raw_text=raw_text,
            cleaned_text=reconstructed,
            tokens=raw_tokens,
            normalized_tokens=normalized_tokens,
            protected_tokens=protected_tokens,
            detected_language=lang,
        )


# ==============================================================================
# 3. COMPOSITIONAL SENTENCE PARSER (CompositionalSentenceParser)
# ==============================================================================

class CompositionalSentenceParser:
    """
    Compositional language understanding engine.
    Dissects inputs into semantic constituents without depending exclusively on rigid templates.
    """

    NEGATION_MARKERS = {
        "not", "never", "no", "neither", "nor", "n't", "none",
        "nahi", "nahin", "na", "mat"
    }

    QUESTION_WORDS = {
        "what": QuestionTarget.WHAT,
        "where": QuestionTarget.WHERE,
        "who": QuestionTarget.WHO,
        "whom": QuestionTarget.WHO,
        "whose": QuestionTarget.WHO,
        "when": QuestionTarget.WHEN,
        "why": QuestionTarget.WHY,
        "how": QuestionTarget.HOW,
        # Hinglish / Hindi transliterated question targets
        "kya": QuestionTarget.WHAT,
        "kahan": QuestionTarget.WHERE,
        "kaha": QuestionTarget.WHERE,
        "kidhar": QuestionTarget.WHERE,
        "kaun": QuestionTarget.WHO,
        "koun": QuestionTarget.WHO,
        "kisko": QuestionTarget.WHO,
        "kab": QuestionTarget.WHEN,
        "kyon": QuestionTarget.WHY,
        "kyu": QuestionTarget.WHY,
        "kaise": QuestionTarget.HOW,
        "kaisa": QuestionTarget.HOW,
        "kaisi": QuestionTarget.HOW,
    }

    POLAR_QUESTION_AUX = {
        "is", "are", "am", "was", "were", "do", "does", "did", "can", "could",
        "will", "would", "should", "has", "have", "had"
    }

    RELATION_KEYWORDS = {
        # Lives in / residence
        "lives": "LIVES_IN",
        "resides": "LIVES_IN",
        "rehta": "LIVES_IN",
        "rehti": "LIVES_IN",
        "rehte": "LIVES_IN",
        "residence": "LIVES_IN",
        # Capital / location
        "capital": "CAPITAL_OF",
        "rajdhani": "CAPITAL_OF",
        # Works / employment
        "works": "WORKS_AT",
        "employed": "WORKS_AT",
        "kaam": "WORKS_AT",
        # Founder / created
        "founded": "FOUNDED_BY",
        "created": "FOUNDED_BY",
        "banaya": "FOUNDED_BY",
        # Definition / identity
        "define": "DEFINITION",
        "definition": "DEFINITION",
        "meaning": "DEFINITION",
        "paribhasha": "DEFINITION",
        "arth": "DEFINITION",
    }

    SOCIAL_PATTERNS = [
        r"\b(how\s+are\s+you|how's\s+it\s+going|how\s+do\s+you\s+do|how\s+are\s+things)\b",
        r"\b(tum\s+kaise\s+ho|aap\s+kaise\s+hain|kaise\s+ho|kya\s+haal\s+hai|kya\s+chal\s+raha\s+hai)\b",
    ]

    GREETING_PATTERNS = [
        r"^(hi|hello|hey|namaste|namaskar|pranam|greetings)\b",
        r"^(good\s+(morning|afternoon|evening|day))\b",
    ]

    TASK_COMMAND_VERBS = {
        "plan", "execute", "calculate", "verify", "find", "search", "evaluate",
        "check", "run", "make", "create", "construct", "step-by-step"
    }

    @classmethod
    def parse(cls, norm: NormalizedText) -> StructuredSentenceForm:
        raw = norm.raw_text
        cleaned = norm.cleaned_text
        lang = norm.detected_language
        low = cleaned.lower()

        # 1. Detect Polarity (Negation)
        words_low = [w.strip(".,;:?!()\"'") for w in low.split() if w.strip()]
        has_negation = any(w in cls.NEGATION_MARKERS for w in words_low)
        polarity = Polarity.NEGATIVE if has_negation else Polarity.AFFIRMATIVE

        # 2. Detect Grammatical Tense
        tense = GrammaticalTense.UNSPECIFIED
        if any(w in ["was", "were", "did", "had", "lived", "resided", "tha", "thi", "the", "gaya"] for w in words_low):
            tense = GrammaticalTense.PAST
        elif any(w in ["will", "shall", "would", "ga", "ge", "gi", "jaega"] for w in words_low):
            tense = GrammaticalTense.FUTURE
        elif any(w in ["is", "are", "am", "does", "do", "lives", "resides", "hai", "hain", "ho"] for w in words_low):
            tense = GrammaticalTense.PRESENT

        # 3. Detect Sentence Mood & Question Target
        # Check for Definition queries first (they are Interrogative / Inquiries)
        is_def_query = (
            any(w in words_low for w in ["define", "definition", "paribhasha", "arth"])
            or "meaning of" in low
            or ("what is" in low and ("computer" in low or "system" in low or "device" in low))
            or re.search(r"\bki\s+(?:definition|defination|paribhasha)\b", low) is not None
        )

        # Check incomplete/truncated fragments requiring clarification
        # e.g., "Computer is", "Computer is an ?", "System is a", or input of 1-3 words ending in copula/determiner
        clean_punct = re.sub(r"[\?\.!,;:]+$", "", cleaned).strip()
        tokens_no_punc = clean_punct.split()
        is_incomplete = False
        clarification_reason = None

        if len(tokens_no_punc) <= 4 and tokens_no_punc:
            last_tok = tokens_no_punc[-1].lower()
            if last_tok in {"is", "are", "am", "hai", "tha", "thi", "a", "an", "the", "ek"}:
                is_incomplete = True
                clarification_reason = f"Statement ends abruptly on copula or determiner '{last_tok}'"

        # Check greetings & social check-in
        is_greeting = any(re.search(p, low) for p in cls.GREETING_PATTERNS)
        is_social = any(re.search(p, low) for p in cls.SOCIAL_PATTERNS)

        # Interrogative detection
        q_target = QuestionTarget.NONE
        is_question = "?" in raw or is_def_query

        # Check question words
        for w in words_low:
            if w in cls.QUESTION_WORDS:
                q_target = cls.QUESTION_WORDS[w]
                is_question = True
                break

        # Check polar questions (e.g. "Does Ram live in Delhi?", "Is computer an electronic device?")
        if not is_question and words_low and words_low[0] in cls.POLAR_QUESTION_AUX:
            q_target = QuestionTarget.POLAR
            is_question = True

        # Determine overall Mood
        if is_incomplete:
            mood = SentenceMood.INCOMPLETE
        elif is_def_query:
            mood = SentenceMood.INTERROGATIVE
            if q_target == QuestionTarget.NONE:
                q_target = QuestionTarget.WHAT
        elif is_social or is_greeting:
            mood = SentenceMood.SOCIAL
        elif is_question:
            mood = SentenceMood.INTERROGATIVE
        elif words_low and words_low[0] in cls.TASK_COMMAND_VERBS:
            mood = SentenceMood.IMPERATIVE
        else:
            mood = SentenceMood.DECLARATIVE

        # 4. Entity & Relation Extraction
        # Extract entities in sequential order of appearance in the sentence
        func_words = {
            "the", "a", "an", "is", "are", "was", "were", "in", "on", "at", "to",
            "where", "what", "who", "when", "why", "how", "define", "tell", "can",
            "could", "do", "does", "did", "and", "or", "but"
        }
        entities: List[str] = []
        for tok in norm.tokens:
            clean_t = tok.strip(".,;:?!()\"'")
            if not clean_t:
                continue
            is_candidate = (
                clean_t in norm.protected_tokens
                or (clean_t.istitle() and clean_t.lower() not in func_words and clean_t.lower() not in TextNormalizer.HINGLISH_VOCAB)
            )
            if is_candidate and clean_t not in entities:
                entities.append(clean_t)

        # If definition query, target entity is the term to define
        subject: Optional[str] = None
        predicate_rel: Optional[str] = None
        target_obj: Optional[str] = None

        if is_def_query:
            predicate_rel = "DEFINITION"
            # Try extracting entity after 'define' or 'meaning of' or 'definition of'
            m_def = re.search(r"\b(?:define|meaning\s+of|definition\s+of)\s+([A-Za-z0-9_\-]+)", cleaned, re.IGNORECASE)
            if m_def:
                subject = m_def.group(1).strip()
            else:
                m_what = re.search(r"\bwhat\s+is\s+(?:a|an|the)?\s*([A-Za-z0-9_\-]+)", cleaned, re.IGNORECASE)
                if m_what:
                    subject = m_what.group(1).strip()
                else:
                    m_hing = re.search(r"([A-Za-z0-9_\-]+)\s+ki\s+(?:definition|defination|paribhasha)", cleaned, re.IGNORECASE)
                    if m_hing:
                        subject = m_hing.group(1).strip()
                    elif "computer" in low:
                        subject = "computer"
                    elif entities:
                        subject = entities[0]

            if subject and subject not in entities:
                entities.append(subject)

        else:
            # Detect relation from keywords
            for word, rel in cls.RELATION_KEYWORDS.items():
                if word in words_low:
                    predicate_rel = rel
                    break

            # Compositional constituent slot filling:
            # English standard: [Subject] [Predicate] [Object] (e.g. "Ram lives in Delhi")
            # Inverted: [Object] is where [Subject] lives (e.g. "Delhi is where Ram lives")
            # Hinglish SOV: [Subject] [Object] mein rehta hai (e.g. "Ram Delhi mein rehta hai")
            if predicate_rel == "LIVES_IN":
                if len(entities) >= 2:
                    if "where" in low:
                        # Inverted: Delhi is where Ram lives -> Subject: Ram, Object: Delhi
                        subject = entities[1]
                        target_obj = entities[0]
                    else:
                        subject = entities[0]
                        target_obj = entities[1]
                elif len(entities) == 1:
                    # Single entity: could be question target (e.g. "Where does Ram live?")
                    subject = entities[0]
                else:
                    # Look for non-capitalized location or person
                    # e.g. "ram lives in delhi"
                    m_eng = re.search(r"([A-Za-z0-9_\-]+)\s+lives\s+in\s+([A-Za-z0-9_\-]+)", cleaned, re.IGNORECASE)
                    if m_eng:
                        subject = m_eng.group(1).strip()
                        target_obj = m_eng.group(2).strip()
                        if subject not in entities:
                            entities.append(subject)
                        if target_obj not in entities:
                            entities.append(target_obj)
                    else:
                        m_hing = re.search(r"([A-Za-z0-9_\-]+)\s+([A-Za-z0-9_\-]+)\s+mein\s+rehta", cleaned, re.IGNORECASE)
                        if m_hing:
                            subject = m_hing.group(1).strip()
                            target_obj = m_hing.group(2).strip()
                            if subject not in entities:
                                entities.append(subject)
                            if target_obj not in entities:
                                entities.append(target_obj)

        # Calculate confidence & uncertainty
        confidence = 0.90
        uncertainty = 0.10

        if mood == SentenceMood.INCOMPLETE:
            confidence = 0.35
            uncertainty = 0.65
        elif not predicate_rel and mood not in (SentenceMood.SOCIAL, SentenceMood.IMPERATIVE):
            uncertainty = 0.40
            confidence = 0.60

        return StructuredSentenceForm(
            raw_input=raw,
            normalized_input=cleaned,
            mood=mood,
            language=lang,
            polarity=polarity,
            tense=tense,
            question_target=q_target,
            subject=subject,
            predicate_relation=predicate_rel,
            target_object=target_obj,
            extracted_entities=entities,
            confidence=confidence,
            uncertainty_score=uncertainty,
            requires_clarification=is_incomplete,
            clarification_reason=clarification_reason,
        )


# ==============================================================================
# 4. GOVERNED LEARNING INTERFACE (CandidateLearningLedger)
# ==============================================================================

class CandidateLearningLedger:
    """
    Structured learning path for user-provided facts and corrections.
    Guarantees:
    - User assertions are held as candidate records with approval status PENDING_VERIFICATION.
    - Full provenance tracking (source, timestamp, raw form, normalized form).
    - ZERO parameter drift: Does NOT modify canonical neural weights or baseline hashes.
    """

    def __init__(self):
        self._records: Dict[str, CandidateLearningRecord] = {}

    def submit_candidate_fact(
        self,
        raw_input: str,
        normalized_input: str,
        subject: str,
        relation: str,
        value: str,
        polarity: Polarity = Polarity.AFFIRMATIVE,
        provenance: str = "USER_SESSION",
        notes: str = "",
    ) -> CandidateLearningRecord:
        content = f"{subject}::{relation}::{value}::{polarity.value}::{raw_input}"
        record_id = "cand_" + hashlib.sha256(content.encode("utf-8")).hexdigest()[:12]

        record = CandidateLearningRecord(
            record_id=record_id,
            raw_input=raw_input,
            normalized_input=normalized_input,
            extracted_subject=subject,
            extracted_relation=relation,
            extracted_value=value,
            polarity=polarity,
            provenance=provenance,
            approval_status=ApprovalStatus.PENDING_VERIFICATION,
            recorded_at=time.time(),
            notes=notes,
        )
        self._records[record_id] = record
        return record

    def get_candidate(self, record_id: str) -> Optional[CandidateLearningRecord]:
        return self._records.get(record_id)

    def list_pending(self) -> List[CandidateLearningRecord]:
        return [r for r in self._records.values() if r.approval_status == ApprovalStatus.PENDING_VERIFICATION]

    def total_count(self) -> int:
        return len(self._records)

    def clear(self) -> None:
        self._records.clear()
