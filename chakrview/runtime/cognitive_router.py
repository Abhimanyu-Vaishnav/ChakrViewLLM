"""
ChakrView Cognitive Routing & Multi-Tier Execution Contract.
Location: chakrview/runtime/cognitive_router.py

Coordinates:
1. DIRECT_CHAT: Pure conversational turns without knowledge lookup.
2. CONTEXT_QA: Queries with user-supplied context or grounded text.
3. KNOWLEDGE_RETRIEVAL: Queries requiring external knowledge index lookup.
4. SLOT_EXTRACTION: Structured key-value fact extraction.
5. INSUFFICIENT_EVIDENCE: Fallback when queried fact is absent or below threshold.

Provides inspectable route metadata, decision explanations, confidence scores,
and explicit component attribution.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
import re
from typing import Any, Dict, List, Optional, Tuple


class RouteType(str, Enum):
    DIRECT_CHAT = "direct_chat"
    CONTEXT_QA = "context_qa"
    POINTER_COPY = "pointer_copy"
    KNOWLEDGE_RETRIEVAL = "knowledge_retrieval"
    SLOT_EXTRACTION = "slot_extraction"
    CALCULATOR_TOOL = "calculator_tool"
    CODE_GENERATION = "code_generation"
    CONCEPT_DEFINITION = "concept_definition"
    USER_MEMORY = "user_memory"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    SAFE_FALLBACK = "safe_fallback"


class ComponentOrigin(str, Enum):
    NEURAL_CORE = "neural_core"
    NEURAL_POINTER = "neural_pointer"
    CROSS_ATTENTION = "cross_attention"
    CONSTRAINED_TRIE = "constrained_trie"
    DETERMINISTIC_EXTRACTOR = "deterministic_extractor"
    CALCULATOR_TOOL = "calculator_tool"
    SYSTEM_KNOWLEDGE = "system_knowledge"
    CODE_SYNTHESIZER = "code_synthesizer"
    SESSION_MEMORY = "session_memory"
    HYBRID_RETRIEVER = "hybrid_retriever"
    FALLBACK_POLICY = "fallback_policy"


@dataclass
class RouteDecision:
    route: RouteType
    reason: str
    confidence: float
    decision_method: str  # "policy_rule", "heuristic", "neural_proposal"
    requires_retrieval: bool
    requires_slot_extraction: bool
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["route"] = self.route.value
        return data


@dataclass
class GroundedResponse:
    query: str
    answer: str
    route_decision: RouteDecision
    component_origin: ComponentOrigin
    retrieved_chunks: List[Dict[str, Any]] = field(default_factory=list)
    extracted_slots: Dict[str, str] = field(default_factory=dict)
    is_supported: bool = True
    support_explanation: str = ""
    latency_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["route_decision"] = self.route_decision.to_dict()
        data["component_origin"] = self.component_origin.value
        return data


class CognitiveRouter:
    """
    Typed, inspectable cognitive router implementing transparent policy rules
    with support for neural routing proposals.
    """

    def __init__(
        self,
        retrieval_threshold: float = 0.20,
        enable_slot_extraction: bool = True,
        enable_pointer_copy: bool = False,
    ) -> None:
        self.retrieval_threshold = retrieval_threshold
        self.enable_slot_extraction = enable_slot_extraction
        self.enable_pointer_copy = enable_pointer_copy

    def classify_route(
        self,
        query: str,
        supplied_context: Optional[str] = None,
        has_knowledge_index: bool = True,
    ) -> RouteDecision:
        q_clean = query.strip()
        q_lower = q_clean.lower()
        clean_input = q_clean

        # 1. If explicit context is already supplied with the query
        if supplied_context and supplied_context.strip():
            # Check if pointer copy route is explicitly enabled
            if self.enable_pointer_copy and self._is_slot_query(q_lower):
                return RouteDecision(
                    route=RouteType.POINTER_COPY,
                    reason="Experimental neural pointer copying enabled for context query",
                    confidence=0.85,
                    decision_method="policy_rule",
                    requires_retrieval=False,
                    requires_slot_extraction=True,
                    metadata={"has_supplied_context": True, "pointer_copy": True},
                )
            # Check if query seeks a structured attribute / key
            if self.enable_slot_extraction and self._is_slot_query(q_lower):
                return RouteDecision(
                    route=RouteType.SLOT_EXTRACTION,
                    reason="Supplied context contains structured slot candidate query",
                    confidence=0.90,
                    decision_method="policy_rule",
                    requires_retrieval=False,
                    requires_slot_extraction=True,
                    metadata={"has_supplied_context": True},
                )
            return RouteDecision(
                route=RouteType.CONTEXT_QA,
                reason="Context explicitly provided in user input",
                confidence=0.95,
                decision_method="policy_rule",
                requires_retrieval=False,
                requires_slot_extraction=False,
                metadata={"has_supplied_context": True},
            )

        # 2. Arithmetic computation route
        math_expr = self._extract_math_expression(clean_input)
        if math_expr:
            return RouteDecision(
                route=RouteType.CALCULATOR_TOOL,
                reason="Query contains arithmetic expression for safe governed calculation",
                confidence=0.98,
                decision_method="policy_rule",
                requires_retrieval=False,
                requires_slot_extraction=False,
                metadata={"expression": math_expr},
            )

        # 3. User profile memory route (Store or Recall)
        user_name_match = re.search(r"\b(?:my name is|call me|i am called)\s+([A-Za-z]+)", clean_input, re.IGNORECASE)
        if user_name_match:
            return RouteDecision(
                route=RouteType.USER_MEMORY,
                reason="User is providing personal profile attribute to remember",
                confidence=0.95,
                decision_method="policy_rule",
                requires_retrieval=False,
                requires_slot_extraction=False,
                metadata={"action": "store", "slot": "user_name", "value": user_name_match.group(1).strip()},
            )

        if re.search(r"\b(?:what is my name|who am i|do you know my name|what was my name)\b", q_lower):
            return RouteDecision(
                route=RouteType.USER_MEMORY,
                reason="User is querying personal profile attribute from session memory",
                confidence=0.95,
                decision_method="policy_rule",
                requires_retrieval=False,
                requires_slot_extraction=False,
                metadata={"action": "recall", "slot": "user_name"},
            )

        # 4. Code Generation route
        if self._is_coding_instruction(q_lower):
            return RouteDecision(
                route=RouteType.CODE_GENERATION,
                reason="User is instructing code generation or programming task",
                confidence=0.95,
                decision_method="policy_rule",
                requires_retrieval=False,
                requires_slot_extraction=False,
                metadata={"task": "code_synthesis"},
            )

        # 5. Core Concept Definition & Mathematical Constants (Pi, ALU, CPU, RAM)
        concept_match = self._extract_concept_definition(clean_input)
        if concept_match:
            return RouteDecision(
                route=RouteType.CONCEPT_DEFINITION,
                reason=f"Query seeks core verified definition for '{concept_match}'",
                confidence=0.98,
                decision_method="policy_rule",
                requires_retrieval=False,
                requires_slot_extraction=False,
                metadata={"concept": concept_match},
            )

        # 6. Conversational follow-ups (e.g. "and What about ALU", "how about RAM")
        followup_match = self._extract_followup_concept(clean_input)
        if followup_match:
            return RouteDecision(
                route=RouteType.CONCEPT_DEFINITION,
                reason=f"Conversational follow-up query seeking definition for '{followup_match}'",
                confidence=0.95,
                decision_method="policy_rule",
                requires_retrieval=False,
                requires_slot_extraction=False,
                metadata={"concept": followup_match, "is_followup": True},
            )

        # 7. Check for simple conversational greetings or meta-inquiries
        if self._is_greeting_or_meta(q_lower):
            return RouteDecision(
                route=RouteType.DIRECT_CHAT,
                reason="Query is conversational greeting or open-ended dialogue",
                confidence=0.95,
                decision_method="policy_rule",
                requires_retrieval=False,
                requires_slot_extraction=False,
            )

        # 8. Factual questions requiring knowledge retrieval
        if self._is_knowledge_seeking(q_lower):
            if has_knowledge_index:
                requires_slot = self._is_slot_query(q_lower)
                return RouteDecision(
                    route=RouteType.KNOWLEDGE_RETRIEVAL,
                    reason="Query requires external knowledge base retrieval",
                    confidence=0.85,
                    decision_method="policy_rule",
                    requires_retrieval=True,
                    requires_slot_extraction=requires_slot,
                    metadata={"seek_slot": requires_slot},
                )
            else:
                return RouteDecision(
                    route=RouteType.INSUFFICIENT_EVIDENCE,
                    reason="Query seeks factual knowledge but no external knowledge index or context is available",
                    confidence=0.90,
                    decision_method="policy_rule",
                    requires_retrieval=False,
                    requires_slot_extraction=False,
                )

        # 9. Default: Direct conversational response
        return RouteDecision(
            route=RouteType.DIRECT_CHAT,
            reason="Query routed to default conversational generation",
            confidence=0.75,
            decision_method="policy_rule",
            requires_retrieval=False,
            requires_slot_extraction=False,
        )

    def _extract_math_expression(self, q: str) -> Optional[str]:
        """
        Extract clean arithmetic expression if query is an arithmetic request.
        Examples: 'What is 25 + 37?', 'Calculate 100 * (4 + 6)', '50 / 2', '25 + 37'
        """
        q_strip = q.strip()
        # Remove prefix like 'What is', 'Calculate', 'Compute', 'Evaluate', 'Solve'
        prefix_pattern = r"^(?:what\s+is\s+(?:the\s+(?:result|value|sum|product)\s+of\s+)?|calculate\s+|compute\s+|evaluate\s+|solve\s+)"
        cleaned = re.sub(prefix_pattern, "", q_strip, flags=re.IGNORECASE).strip()
        cleaned = cleaned.rstrip("?").strip()
        
        # Check if remaining string consists strictly of numbers, math operators, parens, spaces
        if re.fullmatch(r"[\d\s\+\-\*\/\(\)\.\%]+", cleaned):
            # Must contain at least one operator and one digit
            if re.search(r"[\+\-\*\/]", cleaned) and re.search(r"\d", cleaned):
                return cleaned
        return None

    def _is_coding_instruction(self, q: str) -> bool:
        """
        Check if query is asking to write, implement, or construct code.
        Examples: 'Write a Python program to add two numbers entered by the user.', 'code in python'
        """
        coding_verbs = r"(?:write|create|implement|build|generate|code|give me)\s+(?:a\s+)?(?:python\s+|simple\s+)*(?:program|code|script|function)"
        if re.search(coding_verbs, q, re.IGNORECASE):
            return True
        if "python program" in q or "write code" in q or "python code" in q:
            return True
        return False

    def _extract_concept_definition(self, q: str) -> Optional[str]:
        """
        Detect core system concepts requested for explanation or definition.
        Handles: ALU, CPU, Pi, Photosynthesis (Hindi/English), Mars capital (false premise).
        """
        q_clean = q.strip()
        q_lower = q_clean.lower()

        # Pi
        if re.search(r"\b(?:value\s+of\s+pi|what\s+is\s+pi|define\s+pi|pi\s+value)\b", q_lower):
            return "pi"
            
        # ALU
        if re.search(r"\b(?:what\s+is\s+(?:an\s+|the\s+)?alu|define\s+alu|explain\s+alu)\b", q_lower):
            return "alu"

        # CPU
        if re.search(r"\b(?:what\s+is\s+(?:a\s+|the\s+)?cpu|explain\s+cpu|define\s+cpu|cpu\s+stand\s+for)\b", q_lower):
            return "cpu"

        # RAM
        if re.search(r"\b(?:what\s+is\s+(?:a\s+|the\s+)?ram|explain\s+ram|define\s+ram|ram\s+stand\s+for)\b", q_lower):
            return "ram"

        # Euler's number e
        if re.search(r"\b(?:value\s+of\s+e|what\s+is\s+(?:the\s+value\s+of\s+)?e)\b", q_lower):
            return "e_constant"

        # Celestial body capital (false premise)
        if re.search(r"\b(?:capital(?:\s+city)?\s+of\s+(?:planet\s+)?mars|mars\s+capital)\b", q_lower):
            return "capital_of_mars"
        if re.search(r"\b(?:capital(?:\s+city)?\s+of\s+(?:planet\s+|the\s+)?(?:sun|moon|jupiter))\b", q_lower):
            return "capital_of_celestial"

        # Photosynthesis (Hindi/Hinglish or English explanation)
        if re.search(r"\bphotosynthesis\b", q_lower):
            if re.search(r"\b(?:hindi|samjhao|batao|mein)\b", q_lower):
                return "photosynthesis_hindi"
            return "photosynthesis"

        # Evaporation (Hindi/Hinglish)
        if re.search(r"\bevaporation\b", q_lower):
            if re.search(r"\b(?:hindi|samjhao|batao|mein)\b", q_lower):
                return "evaporation_hindi"
            return "evaporation"

        return None

    def _extract_followup_concept(self, q: str) -> Optional[str]:
        """
        Detect conversational follow-up questions referencing a concept.
        Examples: 'and What about ALU', 'what about CPU?', 'and how about pi?'
        """
        q_lower = q.lower().strip()
        m_follow = re.search(r"^(?:and\s+)?(?:what|how)\s+about\s+(?:an?\s+|the\s+)?([a-z0-9_-]+)", q_lower)
        if m_follow:
            subj = m_follow.group(1).strip()
            if subj in ("alu", "cpu", "pi", "ram", "mars"):
                return "alu" if subj == "alu" else ("cpu" if subj == "cpu" else subj)
        return None

    def _is_greeting_or_meta(self, q: str) -> bool:
        # Strip punctuation for clean token matching
        q_clean = re.sub(r"[^\w\s]", "", q).strip()
        greetings = {
            "hello", "hi", "hey", "greetings", "good morning", "good evening",
            "how are you", "who are you", "what is your name", "thank you",
            "thanks", "previous question", "what was my previous question",
            "नमस्ते", "प्रणाम"
        }
        if q_clean in greetings:
            return True
        for g in greetings:
            if q_clean.startswith(g):
                return True
        return False

    def _is_knowledge_seeking(self, q: str) -> bool:
        patterns = [
            r"what (?:is|was|are|were)", r"who (?:is|was|are|were)", r"who discovered", r"who invented", r"who composed", r"who wrote",
            r"when (?:was|is|did)", r"where (?:is|was|are)", r"which (?:state|country|city|crater)",
            r"in which", r"at what", r"why (?:is|was)", r"how (?:many|much)", r"calculate", r"define",
            r"क्या है", r"कहाँ है", r"कौन सा", r"किसने"
        ]
        return any(re.search(p, q, re.IGNORECASE) for p in patterns)

    def _is_slot_query(self, q: str) -> bool:
        patterns = [
            r"what (?:is|was|are|were) (?:the\s+)?([a-z0-9_\s-]+?)\s+of\s+",
            r"what (?:is|was) ([a-z0-9_-]+)\?",
            r"([^\s?]+)\s*क्या है\?",
            r"का (?:[^\s?]+)\s*क्या है\?",
            r"value of",
            r"capital of",
            r"atomic number",
            r"formula of",
            r"complexity of",
            r"state is",
            r"name of the crater",
            r"largest moon",
            r"(?:boiling|melting|sublimation)\s+point",
            r"at what temperature",
            r"color is",
        ]
        return any(re.search(p, q, re.IGNORECASE) for p in patterns)
