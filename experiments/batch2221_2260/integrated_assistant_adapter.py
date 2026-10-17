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
    decision: str  # "CALCULATED", "ANSWER", "ABSTAIN", "REFUTE"
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
        enable_consent_default: bool = False
    ):
        self.policy = AdaptiveExecutionPolicy(forced_device=forced_device)
        self.tool_router = VerifiedToolRouter()
        self.qa_assistant = LocalKnowledgeAssistant()
        
        mem_file = memory_path or (REPO_ROOT / "experiments" / "batch2221_2260" / "user_memory.json")
        fb_file = feedback_path or (REPO_ROOT / "experiments" / "batch2221_2260" / "user_feedback.json")
        
        self.memory_store = ConversationMemoryStore(storage_path=mem_file)
        self.feedback_ledger = FeedbackExperienceLedger(ledger_path=fb_file)
        self.user_consent = enable_consent_default

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
            "qa_indexed_documents": len(self.qa_assistant.list_indexed_documents())
        }

    def ingest_directory(self, doc_dir: Path) -> int:
        return self.qa_assistant.ingest_directory(doc_dir)

    def ask(self, query: str, log_interaction: bool = True) -> IntegratedAssistantResponse:
        t0 = time.perf_counter()
        clean_q = query.strip()
        selected_dev = self.policy.select_device(batch_size=1, seq_len=len(clean_q.split()))

        # 1. First Priority: Check if query is an arithmetic / calculation request
        tool_trace: Optional[ToolExecutionTrace] = self.tool_router.route_and_execute(clean_q)
        if tool_trace and tool_trace.success:
            elapsed = (time.perf_counter() - t0) * 1000.0
            ans = f"{tool_trace.tool_output} (calculated via {tool_trace.tool_name})"
            
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
                decision="CALCULATED",
                confidence=1.0,
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

        # 2. Second Priority: Knowledge Assistant QA with citation verification
        qa_resp: AssistantResponse = self.qa_assistant.ask(clean_q)
        elapsed = (time.perf_counter() - t0) * 1000.0

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
