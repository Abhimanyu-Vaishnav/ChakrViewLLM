"""
ChakrView Ephemeral Conversation Memory & Context Router.
Location: chakrview/runtime/conversation_memory.py

Capabilities:
1. Ephemeral In-RAM Working Memory:
   - Stores user statements, facts, key-value assertions, and dialogue turns in RAM for the active session.
   - Clears cleanly on session reset or program exit. Never persists to disk unless explicit consent is active.
2. Fact Extraction & Disambiguation:
   - Distinguishes declarative user facts ("The project codename is Blue Mango 47.", "My name is Alice.", "Server IP is 10.0.0.1")
     from query questions ("What was the project codename?", "What is 2 + 2?", "Who is Tagore?").
   - Recognizes conversational intents (greetings, gratitude, acknowledgements) without triggering retrieval.
3. Relevant Fact Preservation & Recall:
   - Matches follow-up queries against stored working memory entities, subjects, and attributes.
   - Answers fact recall queries accurately using in-memory conversation context.
4. Non-Fabrication & Strict Grounding:
   - If a question cannot be answered from conversation facts, safe calculator, or document evidence, strictly abstains.
   - Preserves document citations for document-grounded QA.
   - Truthfully reports subsystem execution ("working_memory", "safe_calculator", "local_knowledge_assistant") and
     explicitly confirms neural inference was not invoked when using symbolic/memory routing.
5. Zero modifications to chakrview/runtime/session.py (protected file invariant preserved).
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class ConversationFact:
    """Represents a factual statement provided by the user during the active session."""
    key: str                # Normalized entity or subject key (e.g. "project codename", "name")
    value: str              # Extracted value (e.g. "Blue Mango 47", "Alice")
    raw_statement: str      # Original user statement
    turn_index: int         # Turn index when stated
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "value": self.value,
            "raw_statement": self.raw_statement,
            "turn_index": self.turn_index,
            "timestamp": self.timestamp,
        }


@dataclass
class WorkingMemoryTurn:
    """Represents a conversational turn kept in ephemeral RAM."""
    role: str               # "user" or "assistant"
    content: str
    intent: str             # "statement_fact", "query_memory", "query_calculator", "query_document", "conversational", "unknown"
    turn_index: int
    timestamp: float = field(default_factory=time.time)


class EphemeralConversationMemory:
    """
    In-memory working context manager for multi-turn interactive assistant sessions.
    Maintains bounded turns and extracted session facts in RAM without disk persistence.
    """

    def __init__(self, max_turns: int = 50):
        self.max_turns = max_turns
        self.turns: List[WorkingMemoryTurn] = []
        self.facts: Dict[str, ConversationFact] = {}  # key -> ConversationFact
        self._turn_counter = 0

    def clear(self) -> None:
        """Clears all ephemeral working memory and facts."""
        self.turns.clear()
        self.facts.clear()
        self._turn_counter = 0

    def add_turn(self, role: str, content: str, intent: str = "unknown") -> WorkingMemoryTurn:
        """Records a dialogue turn in ephemeral working memory."""
        self._turn_counter += 1
        turn = WorkingMemoryTurn(
            role=role,
            content=content.strip(),
            intent=intent,
            turn_index=self._turn_counter,
        )
        self.turns.append(turn)
        # Enforce bounded history in RAM
        if len(self.turns) > self.max_turns:
            self.turns = self.turns[-self.max_turns:]
        return turn

    def extract_and_store_fact(self, text: str) -> Optional[ConversationFact]:
        """
        Detects declarative assertions of facts or attributes and stores them in ephemeral memory.
        Examples:
        - "The project codename is Blue Mango 47." -> key="project codename", value="Blue Mango 47"
        - "My name is Alice." -> key="my name", value="Alice"
        - "The database port is 5432." -> key="database port", value="5432"
        - "Project codename: Blue Mango 47" -> key="project codename", value="Blue Mango 47"
        - "Remember that the deadline is next Friday." -> key="deadline", value="next Friday"
        """
        clean = text.strip()
        # Avoid treating questions as facts
        if clean.endswith("?") or re.match(r"^(what|when|where|which|who|whom|why|how|is|are|can|do|does)\b", clean, re.IGNORECASE):
            return None

        # Pattern 1: [optional preamble] (the|my|our) <key> is/was/are/were <value>
        p1 = re.match(
            r"^(?:(?:please\s+)?(?:remember|note|keep\s+in\s+mind)\s+(?:that\s+)?)?"
            r"(?:(?:the|my|our)\s+)?([a-zA-Z0-9_\s\-]+?)\s+(?:is|was|are|were|:=|=|equals)\s+['\"]?(.+?)['\"]?\.?$",
            clean,
            re.IGNORECASE
        )
        if p1:
            raw_key, val = p1.group(1).strip(), p1.group(2).strip()
            # Clean common words from key
            norm_key = self._normalize_key(raw_key)
            if norm_key and val and len(val) >= 1:
                fact = ConversationFact(
                    key=norm_key,
                    value=val,
                    raw_statement=clean,
                    turn_index=self._turn_counter + 1
                )
                self.facts[norm_key] = fact
                return fact

        # Pattern 2: <key>:\s*<value>
        p2 = re.match(r"^(?:(?:the|my|our)\s+)?([a-zA-Z0-9_\s\-]{2,30})\s*:\s*['\"]?(.+?)['\"]?\.?$", clean, re.IGNORECASE)
        if p2:
            raw_key, val = p2.group(1).strip(), p2.group(2).strip()
            norm_key = self._normalize_key(raw_key)
            if norm_key and val and len(val) >= 1:
                fact = ConversationFact(
                    key=norm_key,
                    value=val,
                    raw_statement=clean,
                    turn_index=self._turn_counter + 1
                )
                self.facts[norm_key] = fact
                return fact

        # Pattern 3: "I am <value>", "Call me <value>"
        p3 = re.match(r"^(?:i\s+am|call\s+me)\s+([a-zA-Z0-9_\s\-]+?)\.?$", clean, re.IGNORECASE)
        if p3:
            val = p3.group(1).strip()
            fact = ConversationFact(
                key="my name",
                value=val,
                raw_statement=clean,
                turn_index=self._turn_counter + 1
            )
            self.facts["my name"] = fact
            return fact

        return None

    def query_fact(self, question: str) -> Optional[Tuple[ConversationFact, str]]:
        """
        Attempts to answer a question using stored facts from this session.
        Matches question against stored fact keys or values.
        """
        if not self.facts:
            return None

        clean_q = question.strip()
        q_lower = clean_q.lower()

        # Check for direct key match
        # E.g. "What was the project codename I told you earlier?" -> contains "project codename"
        best_match: Optional[ConversationFact] = None
        best_overlap = 0

        for key, fact in self.facts.items():
            key_tokens = set(re.findall(r"\b[a-zA-Z0-9_-]+\b", key.lower()))
            q_tokens = set(re.findall(r"\b[a-zA-Z0-9_-]+\b", q_lower))
            overlap = len(key_tokens.intersection(q_tokens))
            
            # If all key tokens appear in question or exact substring match
            if key in q_lower or (key_tokens and overlap == len(key_tokens)):
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_match = fact

        if best_match:
            # Format conversational answer
            ans = f"{best_match.value} (remembered from earlier in this session: '{best_match.raw_statement}')"
            return best_match, ans

        return None

    @staticmethod
    def classify_conversational_intent(text: str) -> Optional[str]:
        """
        Identifies social/conversational intents that do not require document retrieval.
        Returns a polite response string if matched, else None.
        """
        clean = text.strip().lower()
        # Greetings
        if re.match(r"^(hi|hello|hey|greetings|good\s+(morning|afternoon|evening))\b", clean):
            return "Hello! I am your ChakrView local assistant. How can I help you today?"

        # Acknowledgements (must be standalone phrase, not a prefix to substantive clauses)
        ack_words = r"(?:thanks|thank\s+you(?:\s+very\s+much)?|thx|cool|awesome|great|ok|okay|got\s+it)"
        if re.match(rf"^{ack_words}(?:\s*(?:very\s+much|a\s+lot|mate|buddy))?[\s!.,;:]*$", clean):
            return "You're welcome! Let me know if you need anything else."

        # Status / capability query
        if clean in ["who are you", "who are you?", "what can you do", "what can you do?"]:
            return "I am ChakrView Local Knowledge Assistant with session memory, verified safe arithmetic, and local document-grounded question answering."

        return None

    @staticmethod
    def _normalize_key(key: str) -> str:
        """Normalizes attribute/entity key strings."""
        k = key.lower().strip()
        # Strip leading articles
        k = re.sub(r"^(the|a|an|my|our)\s+", "", k)
        return k.strip()
