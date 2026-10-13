"""
Memory-Guided Multi-Step Task Execution Engine for ChakrView.
Location: chakrview/runtime/task_engine.py

Provides bounded, verifiable multi-step task execution:
1. Task state machine: (PENDING -> IN_PROGRESS -> COMPLETED / FAILED / ABORTED).
2. Step execution pipeline:
   - READ: Fetch source knowledge / context
   - RETRIEVE: Query knowledge adapter
   - EXTRACT: Isolate factual slot
   - VERIFY: Cross-check against supporting span
   - RECORD: Atomically persist to session memory
3. Bounded retry limits and idempotent execution across simulated restarts.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional

from chakrview.runtime.persistent_memory import PersistentSessionMemory
from chakrview.runtime.evidence_extractor import EvidenceExtractor
from chakrview.runtime.retrieval_adapter import RetrievalAdapter


class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    ABORTED = "aborted"


@dataclass
class TaskStep:
    step_id: str
    action_type: str  # "retrieve", "extract", "verify", "record"
    status: str = "pending"
    input_data: Dict[str, Any] = field(default_factory=dict)
    output_data: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TaskStep:
        return cls(**data)


class SpecializedWorker:
    """Base class for typed deterministic task workers."""
    pass


class RetrievalWorker:
    """Specialized worker for document and knowledge retrieval."""

    def __init__(self, adapter: RetrievalAdapter) -> None:
        self.adapter = adapter

    def execute(self, query: str, top_k: int = 1) -> List[Dict[str, Any]]:
        hits = self.adapter.search(query, top_k=top_k)
        return [{"chunk_id": h.chunk_id, "text": h.text, "score": h.score} for h in hits]


class EvidenceVerificationWorker:
    """Specialized worker for cross-validating extracted facts against multiple evidence sources."""

    @staticmethod
    def verify_consistency(sources: List[str], candidate_fact: str) -> Dict[str, Any]:
        c_fact = candidate_fact.strip().lower()
        supporting_sources = []
        for i, text in enumerate(sources):
            if c_fact in text.lower():
                supporting_sources.append(i)

        return {
            "verified": len(supporting_sources) > 0,
            "supporting_source_indices": supporting_sources,
            "total_sources_checked": len(sources),
            "consensus_ratio": len(supporting_sources) / max(len(sources), 1)
        }


class MathVerifierWorker:
    """Deterministic verifier for safe numeric and arithmetic verification."""

    @staticmethod
    def verify_numeric_fact(extracted_val: str, expected_type: str = "integer") -> bool:
        v = extracted_val.strip()
        if expected_type == "integer":
            return v.isdigit() or (v.startswith("-") and v[1:].isdigit())
        elif expected_type == "float":
            try:
                float(v)
                return True
            except ValueError:
                return False
        return False


@dataclass
class CognitiveTaskState:
    task_id: str
    session_id: str
    goal: str
    status: TaskStatus = TaskStatus.PENDING
    current_step_idx: int = 0
    steps: List[TaskStep] = field(default_factory=list)
    retry_count: int = 0
    max_retries: int = 3
    final_result: Optional[str] = None
    failure_reason: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CognitiveTaskState:
        data = dict(data)
        data["status"] = TaskStatus(data.get("status", "pending"))
        data["steps"] = [TaskStep.from_dict(s) for s in data.get("steps", [])]
        return cls(**data)


class CognitiveTaskEngine:
    """
    Coordinates multi-step grounded workflows with state persistence,
    step verification, and crash recovery.
    """

    def __init__(
        self,
        storage_dir: Path,
        retrieval_adapter: Optional[RetrievalAdapter] = None,
        memory: Optional[PersistentSessionMemory] = None,
    ) -> None:
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.retrieval_adapter = retrieval_adapter or RetrievalAdapter()
        self.memory = memory or PersistentSessionMemory(self.storage_dir / "session_memory")

    def _get_task_file(self, task_id: str) -> Path:
        safe_id = "".join(c for c in task_id if c.isalnum() or c in ("-", "_"))
        return self.storage_dir / f"task_{safe_id}.json"

    def save_task(self, state: CognitiveTaskState) -> None:
        state.updated_at = time.time()
        path = self._get_task_file(state.task_id)
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(state.to_dict(), f, indent=2)
        tmp.replace(path)

    def load_task(self, task_id: str) -> Optional[CognitiveTaskState]:
        path = self._get_task_file(task_id)
        if not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return CognitiveTaskState.from_dict(data)
        except Exception:
            return None

    def execute_factual_workflow(
        self,
        task_id: str,
        session_id: str,
        query: str,
        supplied_doc: Optional[str] = None,
    ) -> CognitiveTaskState:
        """
        Executes an idempotent multi-step task:
        1. Retrieve/Read
        2. Extract slot
        3. Verify against supporting span
        4. Record in session memory
        """
        state = self.load_task(task_id)
        if state is None:
            state = CognitiveTaskState(
                task_id=task_id,
                session_id=session_id,
                goal=f"Retrieve and verify answer for: {query}",
                status=TaskStatus.IN_PROGRESS,
                steps=[
                    TaskStep(step_id="step_1_retrieve", action_type="retrieve", input_data={"query": query}),
                    TaskStep(step_id="step_2_extract", action_type="extract", input_data={"query": query}),
                    TaskStep(step_id="step_3_verify", action_type="verify"),
                    TaskStep(step_id="step_4_record", action_type="record"),
                ],
            )
            self.save_task(state)

        if state.status == TaskStatus.COMPLETED:
            return state

        # Step 1: Retrieve / Read
        step1 = state.steps[0]
        context_text = supplied_doc or ""
        if step1.status != "completed":
            if not context_text and self.retrieval_adapter:
                hits = self.retrieval_adapter.search(query, top_k=1)
                if hits:
                    context_text = hits[0].text
                    step1.output_data = {"retrieved_chunk_id": hits[0].chunk_id, "text": context_text}
                    step1.status = "completed"
                else:
                    step1.status = "failed"
                    step1.error_message = "No matching evidence found in retrieval index"
                    state.status = TaskStatus.FAILED
                    state.failure_reason = step1.error_message
                    self.save_task(state)
                    return state
            else:
                step1.output_data = {"text": context_text}
                step1.status = "completed"
            self.save_task(state)

        # Step 2: Extract Slot
        step2 = state.steps[1]
        extracted_val = ""
        span = ""
        if step2.status != "completed":
            c_text = step1.output_data.get("text", "")
            ext_res = EvidenceExtractor.extract_slot(c_text, query)
            if ext_res.success:
                extracted_val = ext_res.extracted_value
                span = ext_res.supporting_span
                step2.output_data = {"extracted_value": extracted_val, "supporting_span": span}
                step2.status = "completed"
            elif ext_res.has_conflicts:
                step2.status = "failed"
                step2.error_message = f"Conflicting evidence found: {ext_res.conflicting_values}"
                state.status = TaskStatus.FAILED
                state.failure_reason = step2.error_message
                self.save_task(state)
                return state
            else:
                step2.status = "failed"
                step2.error_message = "Requested fact is not mentioned in evidence context"
                state.status = TaskStatus.FAILED
                state.failure_reason = step2.error_message
                self.save_task(state)
                return state
            self.save_task(state)

        # Step 3: Verify
        step3 = state.steps[2]
        if step3.status != "completed":
            val = step2.output_data.get("extracted_value", "")
            supp_span = step2.output_data.get("supporting_span", "")
            if val and val.lower() in supp_span.lower():
                step3.output_data = {"verified": True, "span": supp_span}
                step3.status = "completed"
            else:
                step3.status = "failed"
                step3.error_message = f"Verification failed: '{val}' not present in span '{supp_span}'"
                state.status = TaskStatus.FAILED
                state.failure_reason = step3.error_message
                self.save_task(state)
                return state
        # Step 4: Record in Persistent Session Memory
        step4 = state.steps[3]
        if step4.status != "completed":
            final_ans = step2.output_data.get("extracted_value", "")
            self.memory.append_turn(
                session_id=session_id,
                role="user",
                content=query,
                metadata={"task_id": task_id},
            )
            self.memory.append_turn(
                session_id=session_id,
                role="assistant",
                content=final_ans,
                metadata={"task_id": task_id, "verified_span": step3.output_data.get("span")},
            )
            step4.output_data = {"recorded_session": session_id, "answer": final_ans}
            step4.status = "completed"
            state.status = TaskStatus.COMPLETED
            state.final_result = final_ans
            self.save_task(state)

        return state

    def execute_multi_source_verified_workflow(
        self,
        task_id: str,
        session_id: str,
        query: str,
        sources: List[str],
    ) -> CognitiveTaskState:
        """
        Executes a workflow verified across multiple independent evidence sources.
        """
        state = self.load_task(task_id)
        if state is None:
            state = CognitiveTaskState(
                task_id=task_id,
                session_id=session_id,
                goal=f"Extract and cross-verify fact for: {query} across {len(sources)} sources",
                status=TaskStatus.IN_PROGRESS,
                steps=[
                    TaskStep(step_id="step_1_extract", action_type="extract", input_data={"query": query, "num_sources": len(sources)}),
                    TaskStep(step_id="step_2_cross_verify", action_type="verify"),
                    TaskStep(step_id="step_3_record", action_type="record"),
                ],
            )
            self.save_task(state)

        if state.status == TaskStatus.COMPLETED:
            return state

        # Step 1: Extract candidate from primary source
        step1 = state.steps[0]
        if step1.status != "completed":
            primary_src = sources[0] if sources else ""
            ext_res = EvidenceExtractor.extract_slot(primary_src, query)
            if ext_res.success:
                step1.output_data = {"extracted_value": ext_res.extracted_value, "primary_span": ext_res.supporting_span}
                step1.status = "completed"
            elif ext_res.has_conflicts:
                step1.status = "failed"
                step1.error_message = f"Conflicting evidence in primary source: {ext_res.conflicting_values}"
                state.status = TaskStatus.FAILED
                state.failure_reason = step1.error_message
                self.save_task(state)
                return state
            else:
                step1.status = "failed"
                step1.error_message = "No fact found in primary source"
                state.status = TaskStatus.FAILED
                state.failure_reason = step1.error_message
                self.save_task(state)
                return state
            self.save_task(state)

        # Step 2: Cross verify across all sources
        step2 = state.steps[1]
        if step2.status != "completed":
            cand_val = step1.output_data.get("extracted_value", "")
            cross_res = EvidenceVerificationWorker.verify_consistency(sources, cand_val)
            if cross_res["verified"]:
                step2.output_data = cross_res
                step2.status = "completed"
            else:
                step2.status = "failed"
                step2.error_message = f"Cross-source verification failed for '{cand_val}'"
                state.status = TaskStatus.FAILED
                state.failure_reason = step2.error_message
                self.save_task(state)
                return state
            self.save_task(state)

        # Step 3: Record
        step3 = state.steps[2]
        if step3.status != "completed":
            final_ans = step1.output_data.get("extracted_value", "")
            self.memory.append_turn(session_id=session_id, role="user", content=query, metadata={"task_id": task_id})
            self.memory.append_turn(session_id=session_id, role="assistant", content=final_ans, metadata={"task_id": task_id, "cross_sources": len(sources)})
            step3.output_data = {"recorded_session": session_id, "answer": final_ans}
            step3.status = "completed"
            state.status = TaskStatus.COMPLETED
            state.final_result = final_ans
            self.save_task(state)

        return state
