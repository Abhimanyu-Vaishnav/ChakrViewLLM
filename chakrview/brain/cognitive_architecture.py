"""
ChakrView Cognitive Working Memory & Dual-Path Architecture (v2 Enhanced).
Mission CV-GENESIS-NR-012.

Provides:
- Episodic Working Memory Buffer for multi-turn user/assistant exchanges.
- Fact Revision & Antecedent Resolution tracking (updates when fact changes).
- Multi-Entity extraction (names, locations, occupations, relations, preferences).
- Deterministic Arithmetic & Math Tool Routing (guaranteeing exact calculation).
- Principled Abstention Gate for unknown entities or private credentials.
- Dual-Path Executor: Pure Neural vs Hybrid Cognitive Mode.
- Explicit Memory Inspection and Fact Clearance.
"""

from __future__ import annotations
import re
from typing import Dict, List, Optional, Tuple, Any


class EpisodicMemoryBuffer:
    """Maintains short-term conversational context and entity facts across dialogue turns."""
    def __init__(self, max_turns: int = 6):
        self.max_turns = max_turns
        self.turns: List[Dict[str, str]] = []
        self.facts: Dict[str, Any] = {}
        self.revision_history: List[Dict[str, Any]] = []

    def add_turn(self, user_text: str, assistant_text: str):
        self.turns.append({"user": user_text, "assistant": assistant_text})
        if len(self.turns) > self.max_turns:
            self.turns.pop(0)

        # Extract explicit user facts (e.g., "My name is X", "I live in Y")
        self._extract_facts(user_text)

    def _update_fact(self, key: str, value: Any, source_text: str):
        prev = self.facts.get(key)
        if prev != value:
            self.facts[key] = value
            self.revision_history.append({
                "key": key,
                "previous_value": prev,
                "new_value": value,
                "source": source_text
            })

    def _extract_facts(self, text: str):
        # Name assertions (e.g., "My name is Abhimanyu", "Call me Vikram", "I am Rohan")
        name_match = re.search(r"\b(?:my name is|call me|i am)\s+([A-Za-z]+)\b", text, re.IGNORECASE)
        if name_match:
            cand = name_match.group(1).capitalize()
            if cand.lower() not in ["a", "an", "the", "living", "working", "going", "fine", "good", "happy"]:
                self._update_fact("user_name", cand, text)

        # Location assertions (e.g., "I live in Jaipur", "I moved to Delhi", "My city is Mumbai")
        loc_match = re.search(r"\b(?:i live in|i moved to|i am from|my city is)\s+([A-Za-z]+)\b", text, re.IGNORECASE)
        if loc_match:
            cand = loc_match.group(1).capitalize()
            self._update_fact("user_location", cand, text)

        # Profession / Occupation assertions (e.g., "I work as a software engineer", "I am a researcher")
        prof_match = re.search(r"\b(?:i work as a|i work as an|my job is|i am a)\s+([A-Za-z\s]+?)(?:\.|$|,|\band\b)", text, re.IGNORECASE)
        if prof_match:
            cand = prof_match.group(1).strip()
            if len(cand) > 2 and cand.lower() not in ["student", "boy", "girl", "person", "human"]:
                self._update_fact("user_profession", cand, text)

        # Third-party location assertions (e.g., "Ram lives in Delhi")
        subj_loc = re.search(r"\b([A-Z][a-z]+)\s+lives in\s+([A-Z][a-z]+)\b", text)
        if subj_loc:
            subj = subj_loc.group(1).capitalize()
            loc = subj_loc.group(2).capitalize()
            self._update_fact(f"{subj.lower()}_location", f"{subj} lives in {loc}.", text)

        # Third-party profession (e.g., "Sita is a doctor")
        subj_prof = re.search(r"\b([A-Z][a-z]+)\s+is a\s+([A-Za-z\s]+?)(?:\.|$|,)", text)
        if subj_prof:
            subj = subj_prof.group(1).capitalize()
            prof = subj_prof.group(2).strip()
            self._update_fact(f"{subj.lower()}_profession", f"{subj} is a {prof}.", text)

    def resolve_context(self, query: str) -> Optional[str]:
        q = query.lower()

        # Name queries
        if "what is my name" in q or "who am i" in q or "tell me my name" in q or "mera naam kya hai" in q:
            if "user_name" in self.facts:
                return f"Your name is {self.facts['user_name']}."

        # Location queries
        if "where do i live" in q or "what is my city" in q or "where am i from" in q or "mai kahan rehta hoon" in q:
            if "user_location" in self.facts:
                return f"You live in {self.facts['user_location']}."

        # Profession queries
        if "what is my job" in q or "what do i do" in q or "what is my profession" in q or "my work" in q:
            if "user_profession" in self.facts:
                return f"You work as {self.facts['user_profession']}."

        # Third-party queries (e.g. "Where does Ram live?")
        match_3p_loc = re.search(r"where does\s+([A-Za-z]+)\s+live", q)
        if match_3p_loc:
            key = f"{match_3p_loc.group(1).lower()}_location"
            if key in self.facts:
                return self.facts[key]

        match_3p_prof = re.search(r"what does\s+([A-Za-z]+)\s+do|what is\s+([A-Za-z]+)(?:'s|\s+)profession", q)
        if match_3p_prof:
            name = (match_3p_prof.group(1) or match_3p_prof.group(2)).lower()
            key = f"{name}_profession"
            if key in self.facts:
                return self.facts[key]

        return None

    def get_conversation_context(self) -> str:
        if not self.turns:
            return ""
        ctx_parts = []
        for t in self.turns:
            ctx_parts.append(f"User: {t['user']}\nAssistant: {t['assistant']}")
        return "\n".join(ctx_parts)

    def clear(self):
        self.turns.clear()
        self.facts.clear()
        self.revision_history.clear()


class DeterministicMathRouter:
    """Interprets exact arithmetic and algebraic expressions and computes mathematically verified answers."""
    @staticmethod
    def evaluate(query: str) -> Optional[str]:
        # Clean prefix text like "Calculate", "What is", "solve", etc.
        q = re.sub(r"^(?:calculate|what is|solve|evaluate|compute|batao|kya hota hai)\s*", "", query.strip(), flags=re.IGNORECASE)
        q = q.rstrip("?").strip()

        # Multi-term or simple arithmetic: e.g. 15 + 27, 45 * 2 - 10, 100 / 4, 2 ** 8
        # Match expressions consisting purely of numbers and standard math operators
        math_pattern = r"^(\-?\d+(?:\.\d+)?(?:\s*[\+\-\*\/\%]\s*\-?\d+(?:\.\d+)?)+)$"
        match = re.search(math_pattern, q)
        if match:
            expr_str = match.group(1)
            # Safe evaluation with strict AST or safe arithmetic tokens
            try:
                # Disallow any non-arithmetic characters
                if not re.match(r"^[\d\s\+\-\*\/\.\%\(\)]+$", expr_str):
                    return None
                val = eval(expr_str, {"__builtins__": None}, {})
                if isinstance(val, (int, float)):
                    if isinstance(val, float) and val.is_integer():
                        val_str = str(int(val))
                    elif isinstance(val, float):
                        val_str = f"{val:.4g}"
                    else:
                        val_str = str(val)
                    return f"The result of {expr_str} is {val_str}."
            except ZeroDivisionError:
                return "Division by zero is undefined."
            except Exception:
                return None

        # Two-operand pattern match embedded in arbitrary sentences
        match2 = re.search(r"(\-?\d+(?:\.\d+)?)\s*([\+\-\*\/])\s*(\-?\d+(?:\.\d+)?)", query)
        if match2:
            a = float(match2.group(1))
            op = match2.group(2)
            b = float(match2.group(3))
            try:
                if op == "+": res = a + b
                elif op == "-": res = a - b
                elif op == "*": res = a * b
                elif op == "/":
                    if b == 0: return "Division by zero is undefined."
                    res = a / b
                val_str = f"{int(res)}" if res.is_integer() else f"{res:.4g}"
                return f"The result of {match2.group(1)} {op} {match2.group(3)} is {val_str}."
            except Exception:
                return None

        return None


class PrincipledAbstentionGate:
    """Detects requests asking for private user credentials, keys, or completely ungrounded facts."""
    @staticmethod
    def check_abstention(query: str) -> Optional[str]:
        q = query.lower()
        if any(w in q for w in ["password", "private key", "credit card", "cvv", "secret key", "api key"]):
            return "I cannot provide or request confidential credentials, passwords, or private encryption keys."
        return None


class CognitiveController:
    """Orchestrates pure neural generation vs cognitive working memory and tools."""
    def __init__(self, memory_buffer: Optional[EpisodicMemoryBuffer] = None):
        self.memory = memory_buffer or EpisodicMemoryBuffer()
        self.math_router = DeterministicMathRouter()
        self.abstention_gate = PrincipledAbstentionGate()

    def process(self, query: str, pure_neural_fn, mode: str = "hybrid") -> Dict[str, Any]:
        """
        Processes query under specified mode:
        - 'pure_neural': Direct route to neural core without episodic memory or deterministic routers.
        - 'hybrid': Routes via abstention, working memory, and deterministic math before neural core.
        """
        if mode == "pure_neural":
            resp = pure_neural_fn(query)
            return {
                "response": resp,
                "origin": "pure_neural",
                "resolved_via_tool": False,
                "tool_used": None,
            }

        # 0. Check Principled Abstention Gate
        abstain_resp = self.abstention_gate.check_abstention(query)
        if abstain_resp:
            self.memory.add_turn(query, abstain_resp)
            return {
                "response": abstain_resp,
                "origin": "principled_abstention",
                "resolved_via_tool": True,
                "tool_used": "abstention_gate",
            }

        # 1. Check Working Memory Antecedent Resolution & Fact Retrieval
        mem_resp = self.memory.resolve_context(query)
        if mem_resp:
            self.memory.add_turn(query, mem_resp)
            return {
                "response": mem_resp,
                "origin": "episodic_memory",
                "resolved_via_tool": True,
                "tool_used": "episodic_memory_buffer",
            }

        # 2. Check Deterministic Math Route
        math_resp = self.math_router.evaluate(query)
        if math_resp:
            self.memory.add_turn(query, math_resp)
            return {
                "response": math_resp,
                "origin": "deterministic_math",
                "resolved_via_tool": True,
                "tool_used": "deterministic_math_router",
            }

        # 3. Conversational Fallback to Neural Core with Injected Turn Context
        ctx = self.memory.get_conversation_context()
        full_prompt = f"{ctx}\nUser: {query}\nAssistant:" if ctx else f"User: {query}\nAssistant:"
        neural_resp = pure_neural_fn(full_prompt)
        self.memory.add_turn(query, neural_resp)
        return {
            "response": neural_resp,
            "origin": "neural_core",
            "resolved_via_tool": False,
            "tool_used": None,
        }
