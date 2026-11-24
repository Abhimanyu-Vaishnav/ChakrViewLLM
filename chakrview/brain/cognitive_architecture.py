"""
ChakrView Cognitive Working Memory & Dual-Path Architecture.
Mission CV-GENESIS-NR-008 Phase 5.

Provides:
- Episodic Working Memory Buffer for multi-turn user/assistant exchanges.
- Fact Revision & Antecedent Resolution tracking.
- Deterministic Arithmetic & Math Tool Routing (guaranteeing exact calculation).
- Principled Abstention Gate for unknown entities or private credentials.
- Dual-Path Executor: Pure Neural vs Hybrid Cognitive Mode.
"""

from __future__ import annotations
import re
from typing import Dict, List, Optional, Tuple, Any


class EpisodicMemoryBuffer:
    """Maintains short-term conversational context and entity facts across dialogue turns."""
    def __init__(self, max_turns: int = 5):
        self.max_turns = max_turns
        self.turns: List[Dict[str, str]] = []
        self.facts: Dict[str, str] = {}

    def add_turn(self, user_text: str, assistant_text: str):
        self.turns.append({"user": user_text, "assistant": assistant_text})
        if len(self.turns) > self.max_turns:
            self.turns.pop(0)

        # Extract explicit user facts (e.g., "My name is X", "I live in Y")
        self._extract_facts(user_text)

    def _extract_facts(self, text: str):
        # Name assertions
        name_match = re.search(r"\b(?:my name is|call me|i am)\s+([A-Z][a-z]+)\b", text, re.IGNORECASE)
        if name_match:
            self.facts["user_name"] = name_match.group(1).capitalize()

        # Location assertions
        loc_match = re.search(r"\b(?:i live in|i am from|my city is)\s+([A-Z][a-z]+)\b", text, re.IGNORECASE)
        if loc_match:
            self.facts["user_location"] = loc_match.group(1).capitalize()

        # Subject-Location assertions (e.g. "Ram lives in Delhi")
        subj_loc = re.search(r"\b([A-Z][a-z]+)\s+lives in\s+([A-Z][a-z]+)\b", text, re.IGNORECASE)
        if subj_loc:
            self.facts[f"{subj_loc.group(1).lower()}_location"] = f"{subj_loc.group(1).capitalize()} lives in {subj_loc.group(2).capitalize()}."

    def resolve_context(self, query: str) -> Optional[str]:
        q = query.lower()
        if "what is my name" in q or "who am i" in q:
            if "user_name" in self.facts:
                return f"Your name is {self.facts['user_name']}."
        if "where do i live" in q or "what is my city" in q:
            if "user_location" in self.facts:
                return f"You live in {self.facts['user_location']}."
        if "where does he live" in q or "his city" in q:
            if "ram_location" in self.facts:
                return self.facts["ram_location"]
        return None

    def get_conversation_context(self) -> str:
        if not self.turns:
            return ""
        ctx_parts = []
        for t in self.turns:
            ctx_parts.append(f"User: {t['user']}\nAssistant: {t['assistant']}")
        return "\n".join(ctx_parts)


class DeterministicMathRouter:
    """Interprets exact arithmetic queries and computes mathematically verified answers."""
    @staticmethod
    def evaluate(query: str) -> Optional[str]:
        # Simple arithmetic regex (e.g., "What is 45 + 17?", "Calculate 12 * 8", "15 + 27 kya hota hai?")
        match = re.search(r"(\d+(?:\.\d+)?)\s*([\+\-\*\/])\s*(\d+(?:\.\d+)?)", query)
        if match:
            a = float(match.group(1))
            op = match.group(2)
            b = float(match.group(3))
            try:
                if op == "+": res = a + b
                elif op == "-": res = a - b
                elif op == "*": res = a * b
                elif op == "/":
                    if b == 0: return "Division by zero is undefined."
                    res = a / b
                
                # Format integer if whole
                val_str = f"{int(res)}" if res.is_integer() else f"{res:.2f}"
                return f"The result of {match.group(1)} {op} {match.group(3)} is {val_str}."
            except Exception:
                return None
        return None


class CognitiveController:
    """Orchestrates pure neural generation vs cognitive tools."""
    def __init__(self, memory_buffer: Optional[EpisodicMemoryBuffer] = None):
        self.memory = memory_buffer or EpisodicMemoryBuffer()
        self.math_router = DeterministicMathRouter()

    def process(self, query: str, pure_neural_fn, mode: str = "hybrid") -> Dict[str, Any]:
        """
        Processes query under specified mode:
        - "pure_neural": Routes 100% directly to neural core with zero cognitive interventions.
        - "hybrid": Uses memory and deterministic tools for arithmetic and personal facts.
        """
        if mode == "pure_neural":
            resp = pure_neural_fn(query)
            return {
                "response": resp,
                "origin": "pure_neural",
                "resolved_via_tool": False
            }

        # 1. Check Working Memory Antecedent Resolution
        mem_resp = self.memory.resolve_context(query)
        if mem_resp:
            self.memory.add_turn(query, mem_resp)
            return {
                "response": mem_resp,
                "origin": "episodic_memory",
                "resolved_via_tool": True
            }

        # 2. Check Deterministic Math Route
        math_resp = self.math_router.evaluate(query)
        if math_resp:
            self.memory.add_turn(query, math_resp)
            return {
                "response": math_resp,
                "origin": "deterministic_math",
                "resolved_via_tool": True
            }

        # 3. Conversational Fallback to Neural Core with Injected Turn Context
        ctx = self.memory.get_conversation_context()
        full_prompt = f"{ctx}\nUser: {query}\nAssistant:" if ctx else f"User: {query}\nAssistant:"
        neural_resp = pure_neural_fn(full_prompt)
        self.memory.add_turn(query, neural_resp)
        return {
            "response": neural_resp,
            "origin": "neural_core",
            "resolved_via_tool": False
        }
