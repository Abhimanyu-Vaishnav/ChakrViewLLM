"""
Batches 615-618: Upgraded Neural Span Intent Router with Pointer Grounding.
Location: chakrview/runtime/neural_intent_router.py

Features:
1. Supports both Autoregressive ChakrMicro candidates and ChakrSpanPointerModel pointer candidates.
2. If span pointer checkpoint is provided, predicts intent and source token spans directly.
3. Automatically normalizes extracted expression/text spans (e.g. stripping 'expression:' or trailing '?').
4. Validates parameters against allowlisted ToolRegistry schemas.
5. Emits GroundedResponse with strict component attribution:
   - NEURAL_POINTER / NEURAL_CORE: Intent and span proposal.
   - CALCULATOR_TOOL / TEXT_UTILITY: Deterministic tool computation.
   - FALLBACK_POLICY: Evidence gate refutation and safety policy.
"""
from __future__ import annotations

import json
from pathlib import Path
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import torch

from chakrview.brain.model import ChakrMicro
from chakrview.brain.span_pointer_model import ChakrSpanPointerModel
from chakrview.runtime.cognitive_router import (
    CognitiveRouter,
    ComponentOrigin,
    GroundedResponse,
    RouteDecision,
    RouteType,
)
from chakrview.runtime.evidence_gate import EvidenceGovernedAnsweringGate, GateDecision
from chakrview.runtime.tools import (
    CalculatorTool,
    TextUtilityTool,
    ToolExecutor,
    ToolRegistry,
    get_standard_tool_registry,
)
from chakrview.tokenizer.serialization import load_tokenizer_artifacts


class NeuralIntentRouter:
    """
    Governed Neural Intent & Span Router integrating the trained ChakrView neural cores
    into the production runtime path.
    """

    def __init__(
        self,
        checkpoint_path: Path,
        tokenizer_dir: Path,
        tool_registry: Optional[ToolRegistry] = None,
        evidence_gate: Optional[EvidenceGovernedAnsweringGate] = None,
        device: Optional[torch.device] = None,
        max_new_tokens: int = 80,
    ) -> None:
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_new_tokens = max_new_tokens

        # Load tokenizer
        self.tokenizer, _ = load_tokenizer_artifacts(tokenizer_dir)

        # Inspect and load checkpoint
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.is_span_pointer = "candidate_step250_span_pointer" in str(checkpoint_path) or "base_config" in ckpt

        if self.is_span_pointer:
            base_cfg = ckpt.get("base_config")
            base_model = ChakrMicro(base_cfg).to(self.device)
            self.model = ChakrSpanPointerModel(base_model=base_model).to(self.device)
            self.model.load_state_dict(ckpt["model_state_dict"])
            self.model.eval()
        else:
            self.model = ChakrMicro(ckpt["config"]).to(self.device)
            self.model.load_state_dict(ckpt["model_state_dict"])
            self.model.eval()

        self.tool_registry = tool_registry or get_standard_tool_registry()
        self.tool_executor = ToolExecutor(self.tool_registry)
        self.evidence_gate = evidence_gate or EvidenceGovernedAnsweringGate()

    def predict_structured_intent(self, user_query: str) -> Tuple[Dict[str, Any], str]:
        """
        Runs the neural model to predict structured intent and arguments.
        Returns: (parsed_intent_dict, raw_generation_or_span_summary)
        """
        clean_q = user_query.strip()
        prompt = f"User: {clean_q}\nIntent:"
        input_ids = torch.tensor([[0] + self.tokenizer.encode(prompt)], dtype=torch.long, device=self.device)

        if self.is_span_pointer:
            # Neural Pointer Inference
            intent_name, span, conf = self.model.predict_span(input_ids)
            args: Dict[str, Any] = {}
            status = "ready"
            tool = None

            if intent_name == "calculator":
                tool = "calculator"
                if span:
                    tok_slice = input_ids[0, span[0]:span[1]].tolist()
                    raw_span = self.tokenizer.decode(tok_slice, errors="replace").strip()
                    # Clean prefix/suffix noise from span
                    clean_expr = raw_span.replace("expression:", "").strip().rstrip("?")
                    args["expression"] = clean_expr
                else:
                    status = "clarify"
                    args["missing_field"] = "expression"
            elif intent_name == "text_utility":
                tool = "text_utility"
                if span:
                    tok_slice = input_ids[0, span[0]:span[1]].tolist()
                    raw_span = self.tokenizer.decode(tok_slice, errors="replace").strip()
                    # Detect operation from query
                    q_low = clean_q.lower()
                    op = "uppercase" if "uppercase" in q_low else ("lowercase" if "lowercase" in q_low else ("char_count" if "char" in q_low else "word_count"))
                    clean_text = raw_span.strip("'\" ")
                    args["operation"] = op
                    args["text"] = clean_text
                else:
                    status = "clarify"
                    args["missing_field"] = "text"
            elif intent_name == "concept_definition":
                if span:
                    tok_slice = input_ids[0, span[0]:span[1]].tolist()
                    args["concept"] = self.tokenizer.decode(tok_slice, errors="replace").strip().lower()
            elif intent_name == "knowledge_lookup":
                tool = "retrieval"
                if span:
                    tok_slice = input_ids[0, span[0]:span[1]].tolist()
                    args["query"] = self.tokenizer.decode(tok_slice, errors="replace").strip()
            elif intent_name == "clarification_needed":
                status = "clarify"
                args["missing_field"] = "expression"
            elif intent_name == "unsupported":
                status = "unsupported"
                args["reason"] = "Operation prohibited under system policy"

            res_obj = {
                "schema_version": "v2.0",
                "intent": intent_name,
                "tool": tool,
                "args": args,
                "status": status,
                "confidence": conf,
                "span": span,
            }
            return res_obj, f"<pointer_span intent={intent_name} span={span} conf={conf:.2f}>"

        else:
            # Autoregressive generation fallback
            p_toks = input_ids[0].tolist()
            gen = list(p_toks)
            with torch.no_grad():
                for _ in range(self.max_new_tokens):
                    x = torch.tensor([gen[-256:]], dtype=torch.long, device=self.device)
                    nxt = int(torch.argmax(self.model(x)[0, -1, :]).item())
                    if nxt == 1:
                        break
                    gen.append(nxt)

            raw_gen = self.tokenizer.decode(gen[len(p_toks):], errors="replace").strip()
            first_brace = raw_gen.find("{")
            last_brace = raw_gen.rfind("}")
            if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                try:
                    obj = json.loads(raw_gen[first_brace:last_brace+1])
                    return obj, raw_gen
                except Exception:
                    pass

            return {
                "schema_version": "v1.0",
                "intent": "unparseable_output",
                "tool": None,
                "args": {},
                "status": "unsupported",
            }, raw_gen

    def process_turn(
        self,
        user_query: str,
        session_id: str = "default_session",
    ) -> GroundedResponse:
        t0 = time.perf_counter()
        clean_q = user_query.strip()

        # 1. Pre-flight verification with Evidence Gate
        gate_res = self.evidence_gate.process_query(clean_q)
        if gate_res.decision == GateDecision.REFUTE:
            return GroundedResponse(
                query=clean_q,
                answer=gate_res.content,
                route_decision=RouteDecision(
                    route=RouteType.SAFE_FALLBACK,
                    reason=gate_res.reasoning,
                    confidence=1.0,
                    decision_method="evidence_gate_refutation",
                    requires_retrieval=False,
                    requires_slot_extraction=False,
                ),
                component_origin=ComponentOrigin.FALLBACK_POLICY,
                is_supported=False,
                support_explanation=gate_res.reasoning,
                latency_ms=(time.perf_counter() - t0) * 1000.0,
            )

        # 2. Neural Intent & Span Prediction
        pred, raw_gen = self.predict_structured_intent(clean_q)
        intent = pred.get("intent")
        tool_name = pred.get("tool")
        args = pred.get("args", {})
        status = pred.get("status", "ready")

        metadata = {
            "neural_prediction": pred,
            "raw_neural_summary": raw_gen,
        }

        # 3. Handle Clarification
        if status == "clarify" or intent == "clarification_needed":
            missing = args.get("missing_field", "required parameter")
            msg = f"Clarification required: Please specify the {missing}."
            return GroundedResponse(
                query=clean_q,
                answer=msg,
                route_decision=RouteDecision(
                    route=RouteType.DIRECT_CHAT,
                    reason=f"Neural model identified missing parameter: {missing}",
                    confidence=0.9,
                    decision_method="neural_intent_prediction",
                    requires_retrieval=False,
                    requires_slot_extraction=False,
                    metadata=metadata,
                ),
                component_origin=ComponentOrigin.NEURAL_POINTER if self.is_span_pointer else ComponentOrigin.NEURAL_CORE,
                is_supported=True,
                support_explanation=f"Neural intent identified clarification requirement ({missing})",
                latency_ms=(time.perf_counter() - t0) * 1000.0,
            )

        # 4. Handle Unsupported
        if status == "unsupported" or intent == "unsupported":
            reason = args.get("reason", "This action is disallowed by system security policy.")
            return GroundedResponse(
                query=clean_q,
                answer=f"I cannot perform this operation: {reason}",
                route_decision=RouteDecision(
                    route=RouteType.SAFE_FALLBACK,
                    reason=f"Neural prediction flagged unsupported request: {reason}",
                    confidence=1.0,
                    decision_method="neural_intent_prediction",
                    requires_retrieval=False,
                    requires_slot_extraction=False,
                    metadata=metadata,
                ),
                component_origin=ComponentOrigin.FALLBACK_POLICY,
                is_supported=False,
                support_explanation="Rejected under security boundary policy",
                latency_ms=(time.perf_counter() - t0) * 1000.0,
            )

        # 5. Handle Governed Tool Delegation
        if tool_name and self.tool_registry.has_tool(tool_name):
            tool_res = self.tool_executor.execute(tool_name, args)
            if tool_res.success:
                return GroundedResponse(
                    query=clean_q,
                    answer=str(tool_res.output),
                    route_decision=RouteDecision(
                        route=RouteType.CALCULATOR_TOOL if tool_name == "calculator" else RouteType.SLOT_EXTRACTION,
                        reason=f"Executed allowlisted tool '{tool_name}' with neural pointer extracted arguments",
                        confidence=0.95,
                        decision_method="neural_tool_dispatch",
                        requires_retrieval=False,
                        requires_slot_extraction=True,
                        metadata=metadata,
                    ),
                    component_origin=ComponentOrigin.CALCULATOR_TOOL if tool_name == "calculator" else ComponentOrigin.NEURAL_CORE,
                    extracted_slots=args,
                    is_supported=True,
                    support_explanation=f"Computed deterministically via allowlisted tool '{tool_name}'",
                    latency_ms=(time.perf_counter() - t0) * 1000.0,
                )
            else:
                return GroundedResponse(
                    query=clean_q,
                    answer=f"Tool execution failed: {tool_res.error}",
                    route_decision=RouteDecision(
                        route=RouteType.SAFE_FALLBACK,
                        reason=f"Tool '{tool_name}' reported error: {tool_res.error}",
                        confidence=0.5,
                        decision_method="neural_tool_dispatch_failure",
                        requires_retrieval=False,
                        requires_slot_extraction=False,
                        metadata=metadata,
                    ),
                    component_origin=ComponentOrigin.CALCULATOR_TOOL,
                    is_supported=False,
                    support_explanation=tool_res.error or "Tool error",
                    latency_ms=(time.perf_counter() - t0) * 1000.0,
                )

        # 6. Direct Conversational / Concept
        if intent == "concept_definition":
            concept = args.get("concept", "").lower()
            return GroundedResponse(
                query=clean_q,
                answer=f"Identified concept query for '{concept}'.",
                route_decision=RouteDecision(
                    route=RouteType.CONCEPT_DEFINITION,
                    reason=f"Neural model classified concept definition intent for '{concept}'",
                    confidence=0.85,
                    decision_method="neural_intent_prediction",
                    requires_retrieval=False,
                    requires_slot_extraction=False,
                    metadata=metadata,
                ),
                component_origin=ComponentOrigin.NEURAL_POINTER if self.is_span_pointer else ComponentOrigin.NEURAL_CORE,
                is_supported=True,
                support_explanation=f"Classified concept query for '{concept}'",
                latency_ms=(time.perf_counter() - t0) * 1000.0,
            )

        return GroundedResponse(
            query=clean_q,
            answer="Hello! I am ChakrView, your indigenous AI assistant.",
            route_decision=RouteDecision(
                route=RouteType.DIRECT_CHAT,
                reason="Neural core classified general direct chat intent",
                confidence=0.8,
                decision_method="neural_intent_prediction",
                requires_retrieval=False,
                requires_slot_extraction=False,
                metadata=metadata,
            ),
            component_origin=ComponentOrigin.NEURAL_POINTER if self.is_span_pointer else ComponentOrigin.NEURAL_CORE,
            is_supported=True,
            support_explanation="Direct dialogue interaction",
            latency_ms=(time.perf_counter() - t0) * 1000.0,
        )
