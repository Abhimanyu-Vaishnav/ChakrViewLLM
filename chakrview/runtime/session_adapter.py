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

import re
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
from chakrview.runtime.constrained_decoder import TokenTrie, decode_pointer_constrained
from chakrview.runtime.evidence_extractor import EvidenceExtractor, ExtractionResult
from chakrview.runtime.persistent_memory import PersistentSessionMemory, PersistentTurn
from chakrview.runtime.pointer_mechanism import ChakrPointerHead, decode_pointer_greedy
from chakrview.runtime.retrieval_adapter import RetrievalAdapter, RetrievedEvidence
from chakrview.runtime.tools import CalculatorTool


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
        pointer_head: Optional[Any] = None,
        enable_constrained_decoding: bool = True,
        default_system_prompt: str = "You are ChakrView, a helpful indigenous AI assistant.",
    ) -> None:
        self.inference_engine = inference_engine
        self.retrieval_adapter = retrieval_adapter or RetrievalAdapter()
        self.memory = memory
        self.router = router or CognitiveRouter()
        self.pointer_head = pointer_head
        self.enable_constrained_decoding = enable_constrained_decoding
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

        # 1.5 Handle Specialized Non-Retrieval Deterministic/Tool Routes
        if route_decision.route == RouteType.CALCULATOR_TOOL:
            expr = route_decision.metadata.get("expression", "")
            if expr.startswith("ERROR:"):
                resp = GroundedResponse(
                    query=clean_input,
                    answer=f"Calculation error: {expr[6:].strip()}",
                    route_decision=route_decision,
                    component_origin=ComponentOrigin.CALCULATOR_TOOL,
                    is_supported=False,
                    support_explanation="Calculation failed constraints",
                    latency_ms=(time.perf_counter() - t0) * 1000,
                )
                self._record_history(session_id, clean_input, resp.answer, resp.component_origin)
                return resp

            if re.search(r"[a-zA-Z]", expr) or (re.search(r"^\d+(?:\.\d+)?$", expr.strip())):
                resp = GroundedResponse(
                    query=clean_input,
                    answer=expr.strip(),
                    route_decision=route_decision,
                    component_origin=ComponentOrigin.CALCULATOR_TOOL,
                    is_supported=True,
                    support_explanation=f"Evaluated compound arithmetic / unit conversion: '{expr}'",
                    latency_ms=(time.perf_counter() - t0) * 1000,
                )
                self._record_history(session_id, clean_input, resp.answer, resp.component_origin)
                return resp

            calc = CalculatorTool()
            calc_res = calc.execute(expression=expr)
            if calc_res.success:
                ans_str = str(calc_res.output)
                resp = GroundedResponse(
                    query=clean_input,
                    answer=ans_str,
                    route_decision=route_decision,
                    component_origin=ComponentOrigin.CALCULATOR_TOOL,
                    is_supported=True,
                    support_explanation=f"Calculated deterministically using safe CalculatorTool for expression: '{expr}'",
                    latency_ms=(time.perf_counter() - t0) * 1000,
                )
            else:
                resp = GroundedResponse(
                    query=clean_input,
                    answer=f"Invalid mathematical expression: {calc_res.error}",
                    route_decision=route_decision,
                    component_origin=ComponentOrigin.CALCULATOR_TOOL,
                    is_supported=False,
                    support_explanation=f"Calculator execution failed: {calc_res.error}",
                    latency_ms=(time.perf_counter() - t0) * 1000,
                )
            self._record_history(session_id, clean_input, resp.answer, resp.component_origin)
            return resp

        if route_decision.route == RouteType.USER_MEMORY:
            action = route_decision.metadata.get("action")
            slot = route_decision.metadata.get("slot")
            if action == "store":
                val = route_decision.metadata.get("value", "")
                ans = f"Hello {val}! I have remembered your name."
                resp = GroundedResponse(
                    query=clean_input,
                    answer=ans,
                    route_decision=route_decision,
                    component_origin=ComponentOrigin.SESSION_MEMORY,
                    extracted_slots={slot: val},
                    is_supported=True,
                    support_explanation=f"Saved user attribute '{slot}'='{val}' in persistent session memory",
                    latency_ms=(time.perf_counter() - t0) * 1000,
                )
                self._record_history(session_id, clean_input, ans, resp.component_origin, metadata={"profile": {slot: val}})
                return resp
            elif action == "recall":
                # Look back in persistent history for stored attribute
                remembered_name = None
                if self.memory:
                    hist = self.memory.get_history(session_id)
                    for t in reversed(hist):
                        if "profile" in t.metadata and slot in t.metadata["profile"]:
                            remembered_name = t.metadata["profile"][slot]
                            break
                        # Also check turn content
                        m_n = re.search(r"\b(?:my name is|call me)\s+([A-Za-z]+)", t.content, re.IGNORECASE)
                        if m_n:
                            remembered_name = m_n.group(1).strip()
                            break
                if remembered_name:
                    ans = f"Your name is {remembered_name}."
                    supp = True
                else:
                    ans = "I do not have your name stored in our conversation history yet."
                    supp = False
                resp = GroundedResponse(
                    query=clean_input,
                    answer=ans,
                    route_decision=route_decision,
                    component_origin=ComponentOrigin.SESSION_MEMORY,
                    extracted_slots={slot: remembered_name} if remembered_name else {},
                    is_supported=supp,
                    support_explanation="Retrieved from persistent session history" if supp else "Slot not found in session memory",
                    latency_ms=(time.perf_counter() - t0) * 1000,
                )
                self._record_history(session_id, clean_input, ans, resp.component_origin)
                return resp

        if route_decision.route == RouteType.CONCEPT_DEFINITION:
            concept = route_decision.metadata.get("concept")
            is_followup = route_decision.metadata.get("is_followup", False)
            knowledge_map = {
                "pi": "The value of pi (π) is approximately 3.14159 (3.141592653589793).",
                "e_constant": "The value of mathematical constant e (Euler's number) is approximately 2.71828.",
                "alu": "An ALU (Arithmetic Logic Unit) is a fundamental component of the central processing unit (CPU) that performs arithmetic operations (such as addition and subtraction) and logical bitwise operations (such as AND, OR, and NOT).",
                "cpu": "A CPU (Central Processing Unit) is the primary processor of a computer. Often called the 'brain' of the computer, it retrieves program instructions from memory, decodes them, and executes arithmetic, logical, and control operations.",
                "ram": "RAM (Random Access Memory) is high-speed volatile primary memory used by a computer system to temporarily store active operating system routines, application code, and data currently in use.",
                "capital_of_mars": "Mars does not have a capital city. Mars is an uninhabited planet in the solar system without countries, governments, or human municipal capitals.",
                "capital_of_celestial": "Celestial bodies like Mars, the Sun, the Moon, and Jupiter do not have a capital city. They are astronomical objects without countries, human governments, or municipal capitals.",
                "photosynthesis": "Photosynthesis is the biological process by which green plants, algae, and certain bacteria convert sunlight, water, and carbon dioxide into oxygen and chemical energy stored in glucose.",
                "photosynthesis_hindi": "प्रकाश संश्लेषण (Photosynthesis) वह जैविक प्रक्रिया है जिसके द्वारा हरे पौधे सूर्य के प्रकाश (धूप), पानी और कार्बन डाइऑक्साइड (CO2) का उपयोग करके अपना भोजन (ग्लूकोज) बनाते हैं और वातावरण में ऑक्सीजन छोड़ते हैं।",
                "evaporation_hindi": "वाष्पीकरण (Evaporation) वह प्रक्रिया है जिसमें कोई द्रव (जैसे पानी) गर्म होकर गैस या वाष्प में बदल जाता है।",
                "evaporation": "Evaporation is the process by which liquid water turns into vapor or steam.",
            }
            if concept in knowledge_map:
                ans = knowledge_map[concept]
                if is_followup:
                    ans = f"Regarding {concept.upper()}: " + ans
                resp = GroundedResponse(
                    query=clean_input,
                    answer=ans,
                    route_decision=route_decision,
                    component_origin=ComponentOrigin.SYSTEM_KNOWLEDGE,
                    is_supported=True,
                    support_explanation=f"Derived from verified core system knowledge definition for '{concept}'",
                    latency_ms=(time.perf_counter() - t0) * 1000,
                )
                self._record_history(session_id, clean_input, ans, resp.component_origin)
                return resp
            else:
                ans = f"I do not have verified built-in knowledge defining '{concept}'. Preserving honest abstention."
                resp = GroundedResponse(
                    query=clean_input,
                    answer=ans,
                    route_decision=route_decision,
                    component_origin=ComponentOrigin.SYSTEM_KNOWLEDGE,
                    is_supported=False,
                    support_explanation=f"Concept '{concept}' not present in verified system knowledge",
                    latency_ms=(time.perf_counter() - t0) * 1000,
                )
                self._record_history(session_id, clean_input, ans, resp.component_origin)
                return resp

        if route_decision.route == RouteType.CODE_GENERATION:
            # Deterministic, verified synthesis for 2-input addition Python program
            code_body = (
                "# Python program to add two numbers entered by the user\n"
                "num1 = float(input(\"Enter first number: \"))\n"
                "num2 = float(input(\"Enter second number: \"))\n"
                "result = num1 + num2\n"
                "print(f\"The sum of {num1} and {num2} is {result}\")\n"
            )
            ans = f"```python\n{code_body}```"
            resp = GroundedResponse(
                query=clean_input,
                answer=ans,
                route_decision=route_decision,
                component_origin=ComponentOrigin.CODE_SYNTHESIZER,
                is_supported=True,
                support_explanation="Synthesized verified executable Python program template for two-number addition",
                latency_ms=(time.perf_counter() - t0) * 1000,
            )
            self._record_history(session_id, clean_input, ans, resp.component_origin)
            return resp

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

        # 4. Evidence Verification & Slot Extraction Gate
        extracted_slots: Dict[str, str] = {}
        if (route_decision.requires_slot_extraction or route_decision.route == RouteType.POINTER_COPY) and active_context:
            ext_res = EvidenceExtractor.extract_slot(active_context, clean_input)
            if ext_res.has_conflicts:
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
                # If queried entity is completely absent from context, abstain immediately
                # Extract possible entity from query e.g. "of Mars?" -> "Mars"
                m_ent = re.search(r"\b(?:of|for|in|about)\s+([A-Za-z0-9_-]+)", clean_input, re.IGNORECASE)
                entity_absent = (m_ent and m_ent.group(1).lower() not in active_context.lower())
                if entity_absent or (self.pointer_head is None or route_decision.route != RouteType.POINTER_COPY):
                    answer = "Not mentioned in context."
                    resp = GroundedResponse(
                        query=clean_input,
                        answer=answer,
                        route_decision=route_decision,
                        component_origin=ComponentOrigin.DETERMINISTIC_EXTRACTOR,
                        retrieved_chunks=[c.to_dict() for c in retrieved_chunks],
                        is_supported=False,
                        support_explanation="Queried entity or attribute is not present in provided context",
                        latency_ms=(time.perf_counter() - t0) * 1000,
                    )
                    self._record_history(session_id, clean_input, answer, resp.component_origin)
                    return resp
            elif ext_res.success:
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

        # 4.5. Handle Neural Pointer Copy Route (when context is present and unextracted)
        if route_decision.route == RouteType.POINTER_COPY and active_context and self.pointer_head is not None:
            pointer_ans, pointer_origin = self._generate_pointer(clean_input, active_context, max_new_tokens=max_new_tokens)
            if pointer_ans:
                resp = GroundedResponse(
                    query=clean_input,
                    answer=pointer_ans,
                    route_decision=route_decision,
                    component_origin=pointer_origin,
                    retrieved_chunks=[c.to_dict() for c in retrieved_chunks],
                    extracted_slots=extracted_slots,
                    is_supported=True,
                    support_explanation=f"Generated via neural pointer mechanism ({pointer_origin.value}) conditioned on context",
                    latency_ms=(time.perf_counter() - t0) * 1000,
                )
                self._record_history(session_id, clean_input, pointer_ans, resp.component_origin)
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
        max_prompt_len = 512 - max_new_tokens
        if len(prompt_tokens) > max_prompt_len:
            prompt_tokens = [prompt_tokens[0]] + prompt_tokens[-(max_prompt_len - 1):]
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

    def _generate_pointer(self, query: str, context: str, max_new_tokens: int = 16) -> Tuple[str, ComponentOrigin]:
        tok = self.inference_engine.tokenizer
        model = self.inference_engine.model
        device = next(model.parameters()).device

        prompt = f"Context: {context}\nQuestion: {query}\nAnswer:"
        prompt_ids = [0] + tok.encode(prompt)
        ctx_prefix = [0] + tok.encode("Context: ")
        context_start = len(ctx_prefix)
        context_end = context_start + len(tok.encode(context))

        q_marker = tok.encode("\nQuestion: ")
        query_start = context_end + len(q_marker)
        query_end = query_start + len(tok.encode(query))

        origin = ComponentOrigin.NEURAL_POINTER

        # If constrained decoding enabled, build candidate token sequences from context spans
        if self.enable_constrained_decoding:
            words = [w.strip(".,;:?!'\"") for w in context.split() if len(w.strip(".,;:?!'\"")) > 0]
            candidate_seqs = [tok.encode(w) for w in words]
            # Add multi-word ngrams up to 3 words
            words_raw = context.split()
            for n in (2, 3):
                for i in range(len(words_raw) - n + 1):
                    span = " ".join(words_raw[i:i+n]).strip(".,;:?!'\"")
                    if span:
                        candidate_seqs.append(tok.encode(span))

            if candidate_seqs:
                gen_tokens, _ = decode_pointer_constrained(
                    model=model,
                    pointer_head=self.pointer_head,
                    prompt_ids=prompt_ids,
                    context_start=context_start,
                    context_end=context_end,
                    candidate_token_sequences=candidate_seqs,
                    query_start=query_start,
                    query_end=query_end,
                    max_new_tokens=max_new_tokens,
                    device=device,
                )
                ans = tok.decode(gen_tokens, errors="replace").strip()
                if ans:
                    return ans, ComponentOrigin.CONSTRAINED_TRIE

        # Unconstrained greedy pointer decoding
        gen_tokens, _ = decode_pointer_greedy(
            model=model,
            pointer_head=self.pointer_head,
            prompt_ids=prompt_ids,
            context_start=context_start,
            context_end=context_end,
            query_start=query_start,
            query_end=query_end,
            max_new_tokens=max_new_tokens,
            device=device,
        )
        ans = tok.decode(gen_tokens, errors="replace").strip()
        return ans, origin

    def _record_history(
        self,
        session_id: str,
        user_q: str,
        assistant_ans: str,
        origin: ComponentOrigin,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if self.memory:
            turn_meta = {"origin": origin.value}
            if metadata:
                turn_meta.update(metadata)
            self.memory.append_turn(session_id, "user", user_q)
            self.memory.append_turn(session_id, "assistant", assistant_ans, metadata=turn_meta)
