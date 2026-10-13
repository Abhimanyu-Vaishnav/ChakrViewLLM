"""
Unified Cognitive Routing & Grounded Session Runtime Adapter.
Location: chakrview/runtime/session_adapter.py

Connects:
1. CognitiveRouter (route classification and decision recording).
2. RetrievalAdapter (HybridRetriever knowledge lookup).
3. EvidenceExtractor (deterministic slot verification and abstention).
4. PersistentSessionMemory (multi-turn history storage).
5. InferenceEngine (neural core generation).

Preserves the user's uncommitted modifications in chakrview/runtime/session.py
by operating as an external modular runtime adapter without touching session.py.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import torch

from chakrview.runtime.cognitive_router import (
    CognitiveRouter,
    RouteType,
    RouteDecision,
    GroundedResponse,
    ComponentOrigin,
)
from chakrview.runtime.conversation import ConversationTemplate, ChatMessage, Role
from chakrview.runtime.evidence_extractor import EvidenceExtractor, ExtractionResult
from chakrview.runtime.persistent_memory import PersistentSessionMemory, PersistentTurn
from chakrview.runtime.retrieval_adapter import RetrievalAdapter, RetrievedEvidence


class GroundedSessionAdapter:
    """
    Two-Tier Cognitive Session Adapter orchestrating neural generation,
    retrieval, deterministic slot extraction, and multi-turn persistence.
    """

    def __init__(
        self,
        inference_engine: Any,
        retrieval_adapter: Optional[RetrievalAdapter] = None,
        memory: Optional[PersistentSessionMemory] = None,
        router: Optional[CognitiveRouter] = None,
        default_system_prompt: str = "You are ChakrView, a helpful indigenous AI assistant.",
    ) -> None:
        self.inference_engine = inference_engine
        self.retrieval_adapter = retrieval_adapter or RetrievalAdapter()
        self.memory = memory
        self.router = router or CognitiveRouter()
        self.default_system_prompt = default_system_prompt

    def execute_turn(
        self,
        session_id: str,
        user_input: str,
        supplied_context: Optional[str] = None,
        max_new_tokens: int = 32,
    ) -> GroundedResponse:
        t0 = time.perf_counter()
        clean_input = user_input.strip()

        # 1. Evaluate cognitive route
        has_knowledge = (self.retrieval_adapter.total_chunks > 0)
        route_decision = self.router.classify_route(
            query=clean_input,
            supplied_context=supplied_context,
            has_knowledge_index=has_knowledge,
        )

        retrieved_chunks: List[RetrievedEvidence] = []
        active_context = supplied_context or ""

        # 2. Execute retrieval if route requires it
        if route_decision.requires_retrieval and self.retrieval_adapter:
            retrieved_chunks = self.retrieval_adapter.search(clean_input, top_k=2)
            if retrieved_chunks:
                # Merge top retrieved texts as active context
                active_context = "\n".join(c.text for c in retrieved_chunks)
            else:
                # Retrieval returned no chunks above threshold
                route_decision = RouteDecision(
                    route=RouteType.INSUFFICIENT_EVIDENCE,
                    reason="Knowledge retrieval found no matching evidence above score threshold",
                    confidence=0.90,
                    decision_method="policy_rule",
                    requires_retrieval=False,
                    requires_slot_extraction=False,
                )

        # 3. Handle Insufficient Evidence Route
        if route_decision.route == RouteType.INSUFFICIENT_EVIDENCE:
            answer = "I do not have sufficient information in my knowledge base to answer this question."
            resp = GroundedResponse(
                query=clean_input,
                answer=answer,
                route_decision=route_decision,
                component_origin=ComponentOrigin.FALLBACK_POLICY,
                retrieved_chunks=[c.to_dict() for c in retrieved_chunks],
                is_supported=False,
                support_explanation="Retriever score below confidence threshold or query is out-of-domain",
                latency_ms=(time.perf_counter() - t0) * 1000,
            )
            self._record_history(session_id, clean_input, answer, resp.component_origin)
            return resp

        # 4. Handle Slot Extraction Route (Deterministic priority when supported)
        extracted_slots: Dict[str, str] = {}
        if route_decision.requires_slot_extraction and active_context:
            ext_res = EvidenceExtractor.extract_slot(active_context, clean_input)
            if ext_res.success:
                answer = ext_res.extracted_value
                extracted_slots["primary_slot"] = answer
                resp = GroundedResponse(
                    query=clean_input,
                    answer=answer,
                    route_decision=route_decision,
                    component_origin=ComponentOrigin.DETERMINISTIC_EXTRACTOR,
                    retrieved_chunks=[c.to_dict() for c in retrieved_chunks],
                    extracted_slots=extracted_slots,
                    is_supported=True,
                    support_explanation=f"Exact match extracted from span: '{ext_res.supporting_span}'",
                    latency_ms=(time.perf_counter() - t0) * 1000,
                )
                self._record_history(session_id, clean_input, answer, resp.component_origin)
                return resp
            elif ext_res.has_conflicts:
                conflict_str = ", ".join(f"'{v}'" for v in ext_res.conflicting_values)
                answer = f"Conflicting evidence detected in context: values found include {conflict_str}."
                resp = GroundedResponse(
                    query=clean_input,
                    answer=answer,
                    route_decision=route_decision,
                    component_origin=ComponentOrigin.DETERMINISTIC_EXTRACTOR,
                    retrieved_chunks=[c.to_dict() for c in retrieved_chunks],
                    is_supported=False,
                    support_explanation="Context contains contradictory statements for the requested attribute",
                    latency_ms=(time.perf_counter() - t0) * 1000,
                )
                self._record_history(session_id, clean_input, answer, resp.component_origin)
                return resp
            elif ext_res.is_abstention:
                answer = "Not mentioned in context."
                resp = GroundedResponse(
                    query=clean_input,
                    answer=answer,
                    route_decision=route_decision,
                    component_origin=ComponentOrigin.DETERMINISTIC_EXTRACTOR,
                    retrieved_chunks=[c.to_dict() for c in retrieved_chunks],
                    is_supported=False,
                    support_explanation="Queried attribute is not present in provided context",
                    latency_ms=(time.perf_counter() - t0) * 1000,
                )
                self._record_history(session_id, clean_input, answer, resp.component_origin)
                return resp

        # 5. Neural Generation Route (DIRECT_CHAT or CONTEXT_QA fallback)
        prompt_str = self._build_chatml_prompt(session_id, clean_input, active_context)
        neural_raw = self._generate_neural(prompt_str, max_new_tokens=max_new_tokens)

        resp = GroundedResponse(
            query=clean_input,
            answer=neural_raw,
            route_decision=route_decision,
            component_origin=ComponentOrigin.NEURAL_CORE,
            retrieved_chunks=[c.to_dict() for c in retrieved_chunks],
            extracted_slots=extracted_slots,
            is_supported=bool(active_context),
            support_explanation="Generated by neural core conditioned on dialogue history" + (" and retrieved context" if active_context else ""),
            latency_ms=(time.perf_counter() - t0) * 1000,
        )
        self._record_history(session_id, clean_input, neural_raw, resp.component_origin)
        return resp

    def _build_chatml_prompt(self, session_id: str, user_input: str, context: str) -> str:
        messages: List[ChatMessage] = [
            ChatMessage(role=Role.SYSTEM, content=self.default_system_prompt)
        ]

        # Append persistent history if available
        if self.memory:
            history = self.memory.get_history(session_id)
            # Take last 6 turns for bounded context window
            for t in history[-6:]:
                role_enum = Role.USER if t.role == "user" else Role.ASSISTANT
                messages.append(ChatMessage(role=role_enum, content=t.content))

        # Format current query with context if present
        current_content = f"Context: {context}\nQuestion: {user_input}" if context else user_input
        messages.append(ChatMessage(role=Role.USER, content=current_content))

        return ConversationTemplate.format_conversation(messages, add_generation_prompt=True)

    def _generate_neural(self, prompt: str, max_new_tokens: int = 32) -> str:
        tok = self.inference_engine.tokenizer
        model = self.inference_engine.model
        device = next(model.parameters()).device

        prompt_tokens = [0] + tok.encode(prompt)
        curr = list(prompt_tokens)
        gen = []

        model.eval()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                inp = torch.tensor([curr], dtype=torch.long, device=device)
                nxt = torch.argmax(model(inp)[0, -1]).item()
                curr.append(nxt)
                gen.append(nxt)
                if nxt == 1:
                    break

        raw = tok.decode(gen, errors="replace")
        return raw.replace("<|im_end|>", "").strip()

    def _record_history(self, session_id: str, user_q: str, assistant_ans: str, origin: ComponentOrigin) -> None:
        if self.memory:
            self.memory.append_turn(session_id, "user", user_q)
            self.memory.append_turn(session_id, "assistant", assistant_ans, metadata={"origin": origin.value})
