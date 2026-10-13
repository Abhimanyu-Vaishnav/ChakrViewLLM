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
    KNOWLEDGE_RETRIEVAL = "knowledge_retrieval"
    SLOT_EXTRACTION = "slot_extraction"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    SAFE_FALLBACK = "safe_fallback"


class ComponentOrigin(str, Enum):
    NEURAL_CORE = "neural_core"
    DETERMINISTIC_EXTRACTOR = "deterministic_extractor"
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
    ) -> None:
        self.retrieval_threshold = retrieval_threshold
        self.enable_slot_extraction = enable_slot_extraction

    def classify_route(
        self,
        query: str,
        supplied_context: Optional[str] = None,
        has_knowledge_index: bool = True,
    ) -> RouteDecision:
        q_clean = query.strip()
        q_lower = q_clean.lower()

        # 1. If explicit context is already supplied with the query
        if supplied_context and supplied_context.strip():
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

        # 2. Check for simple conversational greetings or meta-inquiries
        if self._is_greeting_or_meta(q_lower):
            return RouteDecision(
                route=RouteType.DIRECT_CHAT,
                reason="Query is conversational greeting or open-ended dialogue",
                confidence=0.95,
                decision_method="policy_rule",
                requires_retrieval=False,
                requires_slot_extraction=False,
            )

        # 3. Factual questions requiring knowledge retrieval
        if has_knowledge_index and self._is_knowledge_seeking(q_lower):
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

        # 4. Default: Direct conversational response
        return RouteDecision(
            route=RouteType.DIRECT_CHAT,
            reason="Query routed to default conversational generation",
            confidence=0.75,
            decision_method="policy_rule",
            requires_retrieval=False,
            requires_slot_extraction=False,
        )

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
            r"what (?:is|was|are|were)", r"who (?:is|was|are|were)", r"who discovered", r"who invented",
            r"when (?:was|is|did)", r"where (?:is|was|are)", r"which (?:state|country|city|crater)",
            r"in which", r"at what", r"why (?:is|was)", r"how (?:many|much)", r"calculate", r"define",
            r"क्या है", r"कहाँ है", r"कौन सा", r"किसने"
        ]
        return any(re.search(p, q, re.IGNORECASE) for p in patterns)

    def _is_slot_query(self, q: str) -> bool:
        patterns = [
            r"what (?:is|was) ([a-z0-9_-]+)\?",
            r"([^\s?]+)\s*क्या है\?",
            r"value of",
            r"capital of",
            r"atomic number",
            r"state is",
            r"name of the crater",
            r"at what temperature",
            r"color is",
        ]
        return any(re.search(p, q, re.IGNORECASE) for p in patterns)
