"""
Integrated Assistant Runtime Adapter for ChakrView.
Location: experiments/batch2221_2260/integrated_assistant_adapter.py

Connects:
1. AdaptiveExecutionPolicy (hardware discovery, CPU/GPU selection, CPU fallback)
2. VerifiedToolRouter (safe AST calculator for exact arithmetic)
3. ConversationMemoryStore & FeedbackExperienceLedger (user-consented memory & feedback logging)
4. LocalKnowledgeAssistant (citation-grounded document QA & anaphora resolution)

Preserves:
- Zero modifications to chakrview/runtime/session.py
- Zero eval() / exec()
- Full CPU fallback and graceful degradation
"""

from __future__ import annotations

import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chakrview.runtime.local_knowledge_assistant import LocalKnowledgeAssistant, AssistantResponse
from chakrview.runtime.conversation_memory import EphemeralConversationMemory, ConversationFact
from experiments.batch2181_2220.adaptive_runtime_engine import AdaptiveExecutionPolicy
from experiments.batch2181_2220.verified_tool_router import VerifiedToolRouter, ToolExecutionTrace
from experiments.batch2141_2180.feedback_learning_engine import (
    ConversationMemoryStore,
    FeedbackExperienceLedger
)


@dataclass
class IntegratedAssistantResponse:
    query: str
    answer: str
    decision: str  # "CALCULATED", "ANSWER", "ABSTAIN", "REFUTE", "STATEMENT", "CONVERSATIONAL", "WORKING_MEMORY"
    confidence: float
    latency_ms: float
    device_used: str
    execution_breakdown: Dict[str, str] = field(default_factory=dict)
    tool_trace: Optional[Dict[str, Any]] = None
    citations: List[Dict[str, Any]] = field(default_factory=list)
    memory_logged: bool = False
    feedback_id: Optional[str] = None


class IntegratedAssistantAdapter:
    """
    Unified runtime adapter that coordinates hardware selection, tool routing,
    knowledge QA, and feedback capture with truthful subsystem execution attribution.
    """
    def __init__(
        self,
        forced_device: Optional[str] = None,
        memory_path: Optional[Path] = None,
        feedback_path: Optional[Path] = None,
        enable_consent_default: bool = False,
        enable_neural_generation: bool = False,
        neural_checkpoint: Optional[Union[str, Path]] = None,
        neural_max_new_tokens: int = 32,
        neural_prompt_mode: str = "raw_completion",
        neural_decoding: str = "greedy",
        neural_temperature: float = 0.0,
        neural_repetition_penalty: float = 1.0,
        neural_seed: Optional[int] = 42,
    ):
        self.policy = AdaptiveExecutionPolicy(forced_device=forced_device)
        self.tool_router = VerifiedToolRouter()
        self.qa_assistant = LocalKnowledgeAssistant()
        self.working_memory = EphemeralConversationMemory()
        
        mem_file = memory_path or (REPO_ROOT / "experiments" / "batch2221_2260" / "user_memory.json")
        fb_file = feedback_path or (REPO_ROOT / "experiments" / "batch2221_2260" / "user_feedback.json")
        
        self.memory_store = ConversationMemoryStore(storage_path=mem_file)
        self.feedback_ledger = FeedbackExperienceLedger(ledger_path=fb_file)
        self.user_consent = enable_consent_default
        self.enable_neural_generation = enable_neural_generation
        self.neural_checkpoint = neural_checkpoint
        self.neural_max_new_tokens = neural_max_new_tokens
        self.neural_prompt_mode = neural_prompt_mode
        self.neural_decoding = neural_decoding
        self.neural_temperature = neural_temperature
        self.neural_repetition_penalty = neural_repetition_penalty
        self.neural_seed = neural_seed
        
        # Lazy neural runtime adapter
        self.neural_adapter: Optional[Any] = None
        if self.enable_neural_generation:
            from chakrview.runtime.neural_runtime_adapter import NeuralRuntimeAdapter
            self.neural_adapter = NeuralRuntimeAdapter(
                checkpoint_path=self.neural_checkpoint,
                prompt_mode=self.neural_prompt_mode,
                max_new_tokens=self.neural_max_new_tokens,
                decoding=self.neural_decoding,
                temperature=self.neural_temperature,
                repetition_penalty=self.neural_repetition_penalty,
                seed=self.neural_seed,
            )

    def reset_session(self) -> None:
        """Resets ephemeral working memory and session context for a fresh session."""
        self.working_memory.clear()
        self.qa_assistant.reset_conversation()

    def set_user_consent(self, consent: bool) -> None:
        """Explicitly toggles user consent for memory persistence and training-eligible logging."""
        self.user_consent = consent

    def get_system_status(self) -> Dict[str, Any]:
        """Returns comprehensive status of hardware, active device, and memory/feedback stores."""
        hw = self.policy.get_hardware_status()
        selected_device = self.policy.select_device(batch_size=1, seq_len=128)
        return {
            "platform": hw["platform"],
            "cpu_cores": hw["cpu_cores"],
            "system_ram_gb": hw["system_ram_gb"],
            "cuda_available": hw["cuda_available"],
            "gpu_device_name": hw["device_name"],
            "active_selected_device": str(selected_device),
            "user_consent_granted": self.user_consent,
            "memory_records_count": len(self.memory_store.records),
            "feedback_records_count": len(self.feedback_ledger.entries),
            "qa_indexed_documents": len(self.qa_assistant.list_indexed_documents()),
            "working_memory_facts_count": len(self.working_memory.facts),
            "working_memory_turns_count": len(self.working_memory.turns)
        }

    def ingest_directory(self, doc_dir: Path) -> int:
        return self.qa_assistant.ingest_directory(doc_dir)

    def ask(self, query: str, log_interaction: bool = True) -> IntegratedAssistantResponse:
        t0 = time.perf_counter()
        clean_q = query.strip()
        selected_dev = self.policy.select_device(batch_size=1, seq_len=len(clean_q.split()))

        # 1. First Priority: Check if query is an arithmetic / calculation request
        tool_trace: Optional[ToolExecutionTrace] = self.tool_router.route_and_execute(clean_q)
        if tool_trace:
            elapsed = (time.perf_counter() - t0) * 1000.0
            if tool_trace.success:
                ans = f"{tool_trace.tool_output} (calculated via {tool_trace.tool_name})"
                decision_str = "CALCULATED"
                confidence = 1.0
            else:
                ans = f"Incomplete or ambiguous arithmetic expression: {tool_trace.error}. Please provide the complete calculation."
                decision_str = "ABSTAIN"
                confidence = 0.0

            # Record turn in working memory
            self.working_memory.add_turn("user", clean_q, intent="query_calculator")
            self.working_memory.add_turn("assistant", ans, intent="query_calculator")

            # Log to feedback ledger if user consented
            f_id = None
            if log_interaction and self.user_consent:
                fb_entry = self.feedback_ledger.record_interaction(
                    prompt=clean_q,
                    model_output=ans,
                    task_domain="arithmetic",
                    user_consent=True
                )
                f_id = fb_entry["feedback_id"]

            return IntegratedAssistantResponse(
                query=clean_q,
                answer=ans,
                decision=decision_str,
                confidence=confidence,
                latency_ms=round(elapsed, 2),
                device_used=str(selected_dev),
                execution_breakdown={
                    "subsystem": "safe_calculator",
                    "actual_backend": "CPU (AST in-memory)",
                    "neural_inference": "not invoked"
                },
                tool_trace=tool_trace.to_dict(),
                citations=[],
                memory_logged=False,
                feedback_id=f_id
            )

        # 2. Second Priority: Multi-turn arithmetic follow-up referring to prior calculation
        followup = self.tool_router.extract_arithmetic_followup(clean_q)
        if followup:
            op_sym, operand_str, op_verb = followup
            elapsed = (time.perf_counter() - t0) * 1000.0

            # Inspect working memory for the immediate prior calculation turn
            prior_calc_val = None
            prior_calc_abstained = False

            for turn in reversed(self.working_memory.turns):
                if turn.intent == "query_calculator" and turn.role == "assistant":
                    if "Incomplete or ambiguous" in turn.content or "Which number should I" in turn.content:
                        prior_calc_abstained = True
                        break
                    # Extract calculated value from format: "<number> (calculated via safe_calculator)"
                    m_calc = re.match(r"^(\-?\d+(?:\.\d+)?)\s+\(calculated via", turn.content)
                    if m_calc:
                        prior_calc_val = m_calc.group(1)
                        break

            if prior_calc_val is not None and not prior_calc_abstained:
                # Valid, unambiguous prior numeric result exists: safely compute AST expression
                expr = f"{prior_calc_val} {op_sym} {operand_str}"
                res = self.tool_router.calculator.execute(expression=expr)
                if res.success:
                    ans = f"{res.output} (calculated via {self.tool_router.calculator.tool_id})"
                    decision_str = "CALCULATED"
                    confidence = 1.0
                    tool_trace_dict = {
                        "query": clean_q,
                        "tool_name": self.tool_router.calculator.tool_id,
                        "tool_input": {"expression": expr, "prior_result": prior_calc_val},
                        "tool_output": res.output,
                        "success": True,
                        "error": None,
                        "execution_time_ms": elapsed
                    }
                else:
                    ans = f"Incomplete or ambiguous arithmetic expression: {res.error}. Please provide the complete calculation."
                    decision_str = "ABSTAIN"
                    confidence = 0.0
                    tool_trace_dict = None
            else:
                # Ambiguous reference or previous calculation abstained/incomplete:
                # Ask a concise clarification without guessing missing operands
                ans = f"Which number should I {op_verb} by {operand_str}?"
                decision_str = "CLARIFY"
                confidence = 0.0
                tool_trace_dict = None

            # Record turn in working memory
            self.working_memory.add_turn("user", clean_q, intent="query_calculator")
            self.working_memory.add_turn("assistant", ans, intent="query_calculator")

            # Log to feedback ledger if user consented
            f_id = None
            if log_interaction and self.user_consent:
                fb_entry = self.feedback_ledger.record_interaction(
                    prompt=clean_q,
                    model_output=ans,
                    task_domain="arithmetic",
                    user_consent=True
                )
                f_id = fb_entry["feedback_id"]

            return IntegratedAssistantResponse(
                query=clean_q,
                answer=ans,
                decision=decision_str,
                confidence=confidence,
                latency_ms=round(elapsed, 2),
                device_used=str(selected_dev),
                execution_breakdown={
                    "subsystem": "safe_calculator",
                    "actual_backend": "CPU (AST in-memory)",
                    "neural_inference": "not invoked"
                },
                tool_trace=tool_trace_dict,
                citations=[],
                memory_logged=False,
                feedback_id=f_id
            )

        # 3. Third Priority: Conversational intent (greetings, acknowledgements)
        conv_reply = self.working_memory.classify_conversational_intent(clean_q)
        if conv_reply:
            elapsed = (time.perf_counter() - t0) * 1000.0
            self.working_memory.add_turn("user", clean_q, intent="conversational")
            self.working_memory.add_turn("assistant", conv_reply, intent="conversational")
            return IntegratedAssistantResponse(
                query=clean_q,
                answer=conv_reply,
                decision="CONVERSATIONAL",
                confidence=1.0,
                latency_ms=round(elapsed, 2),
                device_used=str(selected_dev),
                execution_breakdown={
                    "subsystem": "working_memory",
                    "actual_backend": "In-Memory Conversational Router",
                    "neural_inference": "not invoked"
                },
                tool_trace=None,
                citations=[],
                memory_logged=False,
                feedback_id=None
            )

        # 3. Third Priority: Recall from ephemeral session facts
        mem_match = self.working_memory.query_fact(clean_q)
        if mem_match:
            fact, ans = mem_match
            elapsed = (time.perf_counter() - t0) * 1000.0
            self.working_memory.add_turn("user", clean_q, intent="query_memory")
            self.working_memory.add_turn("assistant", ans, intent="query_memory")
            return IntegratedAssistantResponse(
                query=clean_q,
                answer=ans,
                decision="WORKING_MEMORY",
                confidence=1.0,
                latency_ms=round(elapsed, 2),
                device_used=str(selected_dev),
                execution_breakdown={
                    "subsystem": "working_memory",
                    "actual_backend": "In-Memory Session Context Store",
                    "neural_inference": "not invoked"
                },
                tool_trace=None,
                citations=[],
                memory_logged=False,
                feedback_id=None
            )

        # 4. Fourth Priority: Declarative fact assertion / statement
        new_fact = self.working_memory.extract_and_store_fact(clean_q)
        if new_fact:
            elapsed = (time.perf_counter() - t0) * 1000.0
            ack_msg = f"I have noted that {new_fact.key} is {new_fact.value} for this session."
            
            # Also persist to ConversationMemoryStore only if user consent is active
            if self.user_consent:
                try:
                    self.memory_store.add_memory(
                        content=clean_q,
                        source="interactive_session",
                        user_consent=True,
                        tags=[new_fact.key]
                    )
                    ack_msg += " (Persisted to local user memory under active consent)"
                except Exception:
                    pass

            self.working_memory.add_turn("user", clean_q, intent="statement_fact")
            self.working_memory.add_turn("assistant", ack_msg, intent="statement_fact")
            return IntegratedAssistantResponse(
                query=clean_q,
                answer=ack_msg,
                decision="STATEMENT",
                confidence=1.0,
                latency_ms=round(elapsed, 2),
                device_used=str(selected_dev),
                execution_breakdown={
                    "subsystem": "working_memory",
                    "actual_backend": "In-Memory Fact Extractor",
                    "neural_inference": "not invoked"
                },
                tool_trace=None,
                citations=[],
                memory_logged=self.user_consent,
                feedback_id=None
            )

        # 5. Fifth Priority: Knowledge Assistant QA with citation verification
        qa_resp: AssistantResponse = self.qa_assistant.ask(clean_q)
        elapsed = (time.perf_counter() - t0) * 1000.0

        # Optional Neural Generation Fallback when QA abstains and neural generation is explicitly enabled
        if qa_resp.decision == "ABSTAIN" and self.enable_neural_generation and self.neural_adapter is not None:
            try:
                gen_res = self.neural_adapter.generate(clean_q)
                neural_elapsed = (time.perf_counter() - t0) * 1000.0
                gen_text = gen_res["generated_text"].strip()
                warning_label = "[EXPERIMENTAL CHAKRVIEW GENERATION - NOT VERIFIED]"
                formatted_answer = f"{warning_label} {gen_text}" if gen_text else f"{warning_label} (model emitted empty completion)"

                # Note: Neural output is recorded under intent='neural_generation' so it is NOT stored as a verified fact
                self.working_memory.add_turn("user", clean_q, intent="neural_generation")
                self.working_memory.add_turn("assistant", formatted_answer, intent="neural_generation")

                return IntegratedAssistantResponse(
                    query=clean_q,
                    answer=formatted_answer,
                    decision="GENERATED_NEURAL",
                    confidence=0.0,
                    latency_ms=round(neural_elapsed, 2),
                    device_used="cpu",
                    execution_breakdown={
                        "subsystem": "neural_model",
                        "actual_backend": "CPU (ChakrMicro v0.1)",
                        "neural_inference": "invoked",
                        "verification_status": "unverified_generated_text",
                        "checkpoint_name": str(gen_res.get("checkpoint_name")),
                        "checkpoint_label": str(gen_res.get("checkpoint_label")),
                        "prompt_mode": str(gen_res.get("prompt_mode")),
                        "decoding_mode": str(gen_res.get("decoding_mode")),
                        "repetition_penalty": str(gen_res.get("repetition_penalty")),
                        "temperature": str(gen_res.get("temperature")),
                        "weight_hash_verified": str(gen_res.get("weight_hash_verified")),
                        "generated_tokens": str(len(gen_res.get("generated_tokens", []))),
                        "stop_reason": str(gen_res.get("stop_reason")),
                    },
                    tool_trace=None,
                    citations=[],
                    memory_logged=False,
                    feedback_id=None
                )
            except Exception as e:
                # If neural generation fails, fail closed and preserve safe deterministic QA abstention
                pass

        self.working_memory.add_turn("user", clean_q, intent="query_document")
        self.working_memory.add_turn("assistant", qa_resp.answer, intent="query_document")

        f_id = None
        if log_interaction and self.user_consent:
            fb_entry = self.feedback_ledger.record_interaction(
                prompt=clean_q,
                model_output=qa_resp.answer,
                task_domain="general_qa",
                user_consent=True
            )
            f_id = fb_entry["feedback_id"]

        return IntegratedAssistantResponse(
            query=clean_q,
            answer=qa_resp.answer,
            decision=qa_resp.decision,
            confidence=qa_resp.confidence,
            latency_ms=round(elapsed, 2),
            device_used=str(selected_dev),
            execution_breakdown={
                "subsystem": "local_knowledge_assistant",
                "actual_backend": "CPU (BM25 + Hybrid lexical/semantic index)",
                "neural_inference": "not invoked"
            },
            tool_trace=None,
            citations=qa_resp.citations,
            memory_logged=False,
            feedback_id=f_id
        )
