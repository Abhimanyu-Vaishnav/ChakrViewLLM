"""
ChakrView Hybrid Conversational Memory Engine (UCIP-01).

Provides:
- Working memory (session turns, active entities, conversation state)
- Semantic key-value memory store (facts, user attributes, preferences)
- State updates & corrections handling (superseding outdated facts)
- Session & User isolation (multi-tenant safe, privacy preserving)
- Deterministic relevance retrieval and memory context formatting for neural prompt packing
"""

from typing import Dict, List, Optional, Any, Tuple
import time
import re

class StructuredFact:
    def __init__(
        self,
        subject: str,
        predicate: str,
        value: str,
        source: str = "user_explicit",
        confidence: float = 1.0,
        turn_index: int = 0,
        previous_value: Optional[str] = None
    ):
        self.subject = subject.strip().lower()
        self.predicate = predicate.strip().lower()
        self.value = value.strip()
        self.source = source
        self.confidence = confidence
        self.turn_index = turn_index
        self.timestamp = time.time()
        self.previous_value = previous_value
        self.is_active = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "subject": self.subject,
            "predicate": self.predicate,
            "value": self.value,
            "source": self.source,
            "confidence": self.confidence,
            "turn_index": self.turn_index,
            "timestamp": self.timestamp,
            "previous_value": self.previous_value,
            "is_active": self.is_active
        }


class HybridMemoryEngine:
    """
    Local deterministic structured memory engine that interfaces directly with
    ChakrView conversational runtimes.
    """
    def __init__(self, session_id: str = "default_session", user_id: str = "default_user"):
        self.session_id = session_id
        self.user_id = user_id
        self.facts: Dict[str, StructuredFact] = {}  # key: f"{subject}:{predicate}"
        self.history: List[Dict[str, str]] = []
        self.working_entities: Dict[str, str] = {}

    def extract_and_store_facts(self, user_text: str, turn_index: int) -> List[StructuredFact]:
        """
        Deterministic rule-based extractor for personal attributes, project renames,
        and user preferences across English, Hindi, and Hinglish.
        """
        extracted = []
        text_lower = user_text.lower()
        
        # Guard: do not extract facts if the sentence is a pure question asking about a fact
        is_pure_question = any(q in text_lower for q in ["what is", "where do", "who is", "who am", "kya hai", "kaun hoon", "kahan rehta"])
        if is_pure_question and not any(k in text_lower for k in ["renamed", "badal kar", "actually, i moved"]):
            return extracted

        # Name patterns
        name_patterns = [
            r"(?:my name is|i am|call me|myself)\s+([A-Z][a-zA-Z]+)",
            r"(?:mera naam|mera name)\s+([A-Z]?[a-zA-Z]+)\s+(?:hai|h)",
            r"(?:mujhe|mujhko)\s+([A-Z]?[a-zA-Z]+)\s+(?:bolo|bulao)",
            r"(?:मेरा नाम)\s+([^\s]+)\s+(?:है)"
        ]
        for pat in name_patterns:
            m = re.search(pat, user_text, re.IGNORECASE)
            if m:
                val = m.group(1).strip()
                fact = self.set_fact("user", "name", val, turn_index=turn_index)
                extracted.append(fact)
                break

        # City / Location patterns
        city_patterns = [
            r"(?:i live in|i am from|my city is|located in)\s+([A-Z][a-zA-Z]+)",
            r"(?:main|mai)\s+([A-Z]?[a-zA-Z]+)\s+(?:se hoon|me rehta hoon|me rehti hoon|rehta hu)",
            r"(?:mera shahar|mera city)\s+([A-Z]?[a-zA-Z]+)\s+(?:hai|h)",
            r"(?:मैं)\s+([^\s]+)\s+(?:में रहता हूँ|में रहती हूँ|से हूँ)"
        ]
        for pat in city_patterns:
            m = re.search(pat, user_text, re.IGNORECASE)
            if m:
                val = m.group(1).strip()
                fact = self.set_fact("user", "city", val, turn_index=turn_index)
                extracted.append(fact)
                break

        # Project patterns & renames (search renames first)
        rename_patterns = [
            r"(?:renamed it to|rename it to|renamed my project to)\s+([A-Za-z0-9_-]+)",
            r"(?:project ka naam badal kar)\s+([A-Za-z0-9_-]+)\s+(?:kar diya|h)"
        ]
        renamed = False
        for pat in rename_patterns:
            m = re.search(pat, user_text, re.IGNORECASE)
            if m:
                val = m.group(1).strip()
                fact = self.set_fact("user", "project", val, turn_index=turn_index)
                extracted.append(fact)
                renamed = True
                break

        if not renamed:
            project_patterns = [
                r"(?:my project is|working on|project called)\s+([A-Za-z0-9_-]+)",
                r"(?:mera project|mera project ka naam)\s+([A-Za-z0-9_-]+)\s+(?:hai|h)"
            ]
            for pat in project_patterns:
                m = re.search(pat, user_text, re.IGNORECASE)
                if m:
                    val = m.group(1).strip()
                    fact = self.set_fact("user", "project", val, turn_index=turn_index)
                    extracted.append(fact)
                    break

        # Favorite / Preference
        pref_patterns = [
            r"(?:my favorite tool is|i love using|preferred tool is)\s+([A-Za-z0-9_-]+)",
            r"(?:mera favorite tool)\s+([A-Za-z0-9_-]+)\s+(?:hai|h)"
        ]
        for pat in pref_patterns:
            m = re.search(pat, user_text, re.IGNORECASE)
            if m:
                val = m.group(1).strip()
                fact = self.set_fact("user", "favorite_tool", val, turn_index=turn_index)
                extracted.append(fact)
                break

        return extracted

    def set_fact(self, subject: str, predicate: str, value: str, turn_index: int = 0) -> StructuredFact:
        key = f"{subject.lower()}:{predicate.lower()}"
        prev_val = self.facts[key].value if key in self.facts else None
        fact = StructuredFact(
            subject=subject,
            predicate=predicate,
            value=value,
            turn_index=turn_index,
            previous_value=prev_val
        )
        self.facts[key] = fact
        self.working_entities[predicate.lower()] = value
        return fact

    def get_fact(self, subject: str, predicate: str) -> Optional[StructuredFact]:
        key = f"{subject.lower()}:{predicate.lower()}"
        return self.facts.get(key)

    def retrieve_relevant_facts(self, query: str) -> List[StructuredFact]:
        """
        Retrieves active facts relevant to the incoming user query.
        """
        query_lower = query.lower()
        relevant = []

        for key, fact in self.facts.items():
            if not fact.is_active:
                continue
            pred = fact.predicate
            # Relevance match
            if pred == "name" and any(w in query_lower for w in ["name", "who am i", "naam", "kaun hoon"]):
                relevant.append(fact)
            elif pred == "city" and any(w in query_lower for w in ["city", "where do i live", "shahar", "kahan rehta"]):
                relevant.append(fact)
            elif pred == "project" and any(w in query_lower for w in ["project", "working on", "kya bana raha"]):
                relevant.append(fact)
            elif pred == "favorite_tool" and any(w in query_lower for w in ["tool", "favorite", "pasand"]):
                relevant.append(fact)
            elif fact.value.lower() in query_lower:
                relevant.append(fact)

        return relevant

    def format_memory_context(self, relevant_facts: List[StructuredFact]) -> str:
        """
        Formats active structured memory facts into a clean prefix for prompt conditioning.
        """
        if not relevant_facts:
            return ""
        items = []
        for f in relevant_facts:
            items.append(f"{f.predicate.capitalize()}: {f.value}")
        return f"[Memory: {', '.join(items)}]\n"

    def reset_session(self):
        """
        Clears working memory for the session.
        """
        self.facts.clear()
        self.history.clear()
        self.working_entities.clear()
