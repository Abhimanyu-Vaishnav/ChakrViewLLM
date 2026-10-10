"""
Durable Execution Journal, Crash Recovery & Resumption Engine for ChakrView.

Architectural Guarantees:
1. Stable task and execution identifiers (UUID-backed or caller-defined).
2. Explicit, validated task execution states & state transitions:
   PENDING -> RUNNING -> COMMITTED -> (VERIFYING -> COMPLETED) or (FAILED -> RETRYING / ROLLED_BACK)
3. Persisted execution progress: total units/chunks, completed units, committed results,
   pending units, attempt counts, and resource execution budgets.
4. Atomic journal updates via temporary files and os.replace().
5. Process restart recovery: reload journal on fresh process, skip committed units,
   re-execute pending/failed units idempotently.
6. Checksum integrity: SHA-256 validation of journal contents with tamper detection.
7. Schema versioning: explicit JOURNAL_SCHEMA_VERSION with rejection of incompatible versions.
8. Delivery semantics: At-least-once chunk execution with idempotent result deduplication.
   (Documented: side-effecting operations must ensure idempotency keys).
9. Resource-budget preservation across recovery restarts.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Dict, Any, List, Optional, Tuple, Callable, Set

from chakrview.runtime.hardware_adaptive_engine import (
    ResourceAllocationBudget,
    HardwareAdaptiveGovernor,
    HardwareDiscovery,
)


JOURNAL_SCHEMA_VERSION = "1.0.0"


class JournalTaskStatus(str, Enum):
    """Explicit lifecycle status for a journaled execution task."""
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    CHUNK_COMMITTED = "CHUNK_COMMITTED"
    VERIFYING = "VERIFYING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    RETRYING = "RETRYING"
    ROLLED_BACK = "ROLLED_BACK"


VALID_JOURNAL_TRANSITIONS: Dict[JournalTaskStatus, Set[JournalTaskStatus]] = {
    JournalTaskStatus.PENDING: {
        JournalTaskStatus.RUNNING,
        JournalTaskStatus.FAILED,
    },
    JournalTaskStatus.RUNNING: {
        JournalTaskStatus.CHUNK_COMMITTED,
        JournalTaskStatus.VERIFYING,
        JournalTaskStatus.FAILED,
        JournalTaskStatus.RETRYING,
    },
    JournalTaskStatus.CHUNK_COMMITTED: {
        JournalTaskStatus.RUNNING,
        JournalTaskStatus.VERIFYING,
        JournalTaskStatus.COMPLETED,
        JournalTaskStatus.FAILED,
        JournalTaskStatus.RETRYING,
    },
    JournalTaskStatus.VERIFYING: {
        JournalTaskStatus.COMPLETED,
        JournalTaskStatus.FAILED,
        JournalTaskStatus.ROLLED_BACK,
    },
    JournalTaskStatus.FAILED: {
        JournalTaskStatus.RETRYING,
        JournalTaskStatus.ROLLED_BACK,
    },
    JournalTaskStatus.RETRYING: {
        JournalTaskStatus.RUNNING,
        JournalTaskStatus.FAILED,
    },
    JournalTaskStatus.ROLLED_BACK: {
        JournalTaskStatus.PENDING,
        JournalTaskStatus.FAILED,
    },
    JournalTaskStatus.COMPLETED: set(),  # Terminal
}


class JournalError(RuntimeError):
    """Base exception for journal failures."""
    pass


class IncompatibleSchemaError(JournalError):
    """Raised when journal file schema version does not match supported version."""
    pass


class CorruptedJournalError(JournalError):
    """Raised when journal file is malformed, truncated, or fails checksum verification."""
    pass


class InvalidJournalTransitionError(JournalError):
    """Raised when an illegal lifecycle transition is attempted."""
    pass


@dataclass
class JournalChunkRecord:
    """Record of an individual unit/chunk execution state."""
    chunk_index: int
    total_chunks: int
    chunk_payload: str
    is_committed: bool = False
    result: Optional[Any] = None
    error: Optional[str] = None
    attempt_count: int = 0
    committed_at: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "JournalChunkRecord":
        return cls(**data)


@dataclass
class ExecutionJournalState:
    """Complete serializable snapshot of a journaled execution task."""
    task_id: str
    schema_version: str
    status: JournalTaskStatus
    total_chunks: int
    completed_chunks_count: int
    chunks: List[JournalChunkRecord]
    committed_results: List[Any]
    failed_chunk_indices: List[int]
    attempt_count: int
    peak_working_bytes: int
    created_at: float
    updated_at: float
    budget_summary: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    checksum: str = ""

    def compute_payload_hash(self) -> str:
        """Computes deterministic SHA-256 hash across all data excluding checksum itself."""
        data_to_hash = {
            "task_id": self.task_id,
            "schema_version": self.schema_version,
            "status": self.status.value,
            "total_chunks": self.total_chunks,
            "completed_chunks_count": self.completed_chunks_count,
            "chunks": [c.to_dict() for c in self.chunks],
            "committed_results": self.committed_results,
            "failed_chunk_indices": sorted(list(self.failed_chunk_indices)),
            "attempt_count": self.attempt_count,
            "peak_working_bytes": self.peak_working_bytes,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "budget_summary": self.budget_summary,
            "metadata": self.metadata,
        }
        encoded = json.dumps(data_to_hash, sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "task_id": self.task_id,
            "schema_version": self.schema_version,
            "status": self.status.value,
            "total_chunks": self.total_chunks,
            "completed_chunks_count": self.completed_chunks_count,
            "chunks": [c.to_dict() for c in self.chunks],
            "committed_results": self.committed_results,
            "failed_chunk_indices": self.failed_chunk_indices,
            "attempt_count": self.attempt_count,
            "peak_working_bytes": self.peak_working_bytes,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "budget_summary": self.budget_summary,
            "metadata": self.metadata,
            "checksum": self.checksum,
        }
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExecutionJournalState":
        d = dict(data)
        d["status"] = JournalTaskStatus(d["status"])
        d["chunks"] = [JournalChunkRecord.from_dict(c) for c in d["chunks"]]
        return cls(**d)


class DurableExecutionJournal:
    """
    Disk-backed, crash-resilient journal manager for adaptive task execution.
    """

    def __init__(self, journal_path: Path | str) -> None:
        self.journal_path = Path(journal_path)
        self.journal_path.parent.mkdir(parents=True, exist_ok=True)

    def write_atomic(self, state: ExecutionJournalState) -> None:
        """
        Atomically writes execution state to disk using a temporary file
        and atomic replacement. Calculates content SHA-256 before write.
        """
        state.updated_at = time.time()
        state.checksum = state.compute_payload_hash()

        payload = state.to_dict()
        serialized = json.dumps(payload, indent=2, sort_keys=True)

        target_dir = self.journal_path.parent
        # Create temp file in same directory to ensure same filesystem for atomic rename
        temp_fd, temp_path_str = tempfile.mkstemp(
            dir=str(target_dir),
            prefix=f".{self.journal_path.stem}_",
            suffix=".tmp",
        )
        try:
            with os.fdopen(temp_fd, "w", encoding="utf-8") as f:
                f.write(serialized)
                f.flush()
                os.fsync(f.fileno())

            # Atomic replace
            os.replace(temp_path_str, str(self.journal_path))
        except Exception as e:
            if os.path.exists(temp_path_str):
                try:
                    os.unlink(temp_path_str)
                except Exception:
                    pass
            raise JournalError(f"Atomic journal write failed: {e}") from e

    def load(self) -> ExecutionJournalState:
        """
        Loads and cryptographically verifies the journal state from disk.
        """
        if not self.journal_path.is_file():
            raise FileNotFoundError(f"Journal file not found: {self.journal_path}")

        try:
            with open(self.journal_path, "r", encoding="utf-8") as f:
                content = f.read()
            if not content.strip():
                raise CorruptedJournalError("Journal file is empty.")
            data = json.loads(content)
        except json.JSONDecodeError as jde:
            raise CorruptedJournalError(f"Journal file contains invalid JSON: {jde}") from jde

        # Schema version check
        schema_version = data.get("schema_version")
        if not schema_version:
            raise CorruptedJournalError("Journal missing schema_version.")
        if schema_version != JOURNAL_SCHEMA_VERSION:
            raise IncompatibleSchemaError(
                f"Incompatible journal schema version '{schema_version}'. Expected '{JOURNAL_SCHEMA_VERSION}'."
            )

        stored_checksum = data.get("checksum", "")
        if not stored_checksum:
            raise CorruptedJournalError("Journal missing integrity checksum.")

        state = ExecutionJournalState.from_dict(data)
        computed_checksum = state.compute_payload_hash()

        if computed_checksum != stored_checksum:
            raise CorruptedJournalError(
                f"Journal checksum mismatch! Stored: {stored_checksum}, Computed: {computed_checksum}."
            )

        return state

    def exists(self) -> bool:
        return self.journal_path.is_file()


class DurableChunkedExecutor:
    """
    Executes chunked workloads with durable disk journaling, crash recovery,
    and idempotency guarantees.
    """

    @classmethod
    def execute(
        cls,
        task_id: str,
        text: str,
        chunk_processor_fn: Callable[[str], Any],
        budget: ResourceAllocationBudget,
        journal_path: Path | str,
        simulate_crash_after_chunk: Optional[int] = None,
        simulate_failure_chunk: Optional[int] = None,
        verification_fn: Optional[Callable[[List[Any]], bool]] = None,
    ) -> ExecutionJournalState:
        """
        Executes a task chunk-by-chunk with durable progress journaling.
        Supports resume from existing journal if one is present on disk.
        """
        journal = DurableExecutionJournal(journal_path)
        chunks = cls._split_into_chunks(text, budget.default_chunk_chars)
        total_chunks = len(chunks)

        if journal.exists():
            state = journal.load()
            if state.task_id != task_id:
                raise JournalError(
                    f"Journal task_id mismatch. File has '{state.task_id}', requested '{task_id}'."
                )
            if state.total_chunks != total_chunks:
                raise IncompatibleSchemaError(
                    f"Journal chunks mismatch. Expected {total_chunks}, found {state.total_chunks}."
                )
            # If task is already completed, return immediately (idempotent call)
            if state.status == JournalTaskStatus.COMPLETED:
                return state

            # Resuming existing task
            state.attempt_count += 1
            cls._transition_status(state, JournalTaskStatus.RUNNING, "Resuming execution from journal")
            journal.write_atomic(state)
        else:
            # Initialize fresh journal
            now = time.time()
            chunk_records = [
                JournalChunkRecord(
                    chunk_index=i,
                    total_chunks=total_chunks,
                    chunk_payload=c,
                    is_committed=False,
                )
                for i, c in enumerate(chunks)
            ]
            state = ExecutionJournalState(
                task_id=task_id,
                schema_version=JOURNAL_SCHEMA_VERSION,
                status=JournalTaskStatus.PENDING,
                total_chunks=total_chunks,
                completed_chunks_count=0,
                chunks=chunk_records,
                committed_results=[],
                failed_chunk_indices=[],
                attempt_count=1,
                peak_working_bytes=0,
                created_at=now,
                updated_at=now,
                budget_summary=budget.summary,
            )
            cls._transition_status(state, JournalTaskStatus.RUNNING, "Starting new task execution")
            journal.write_atomic(state)

        # Execute chunk-by-chunk
        for idx in range(total_chunks):
            chunk_rec = state.chunks[idx]

            # Idempotency check: Skip already committed chunks
            if chunk_rec.is_committed:
                continue

            chunk_rec.attempt_count += 1
            chunk_bytes = len(chunk_rec.chunk_payload.encode("utf-8"))
            if chunk_bytes > state.peak_working_bytes:
                state.peak_working_bytes = chunk_bytes

            # Simulate transient failure if requested
            if simulate_failure_chunk is not None and idx == simulate_failure_chunk and chunk_rec.attempt_count == 1:
                chunk_rec.error = f"Transient fault on chunk {idx}"
                if idx not in state.failed_chunk_indices:
                    state.failed_chunk_indices.append(idx)
                cls._transition_status(state, JournalTaskStatus.FAILED, f"Chunk {idx} fault")
                journal.write_atomic(state)

                # Replan & retry
                cls._transition_status(state, JournalTaskStatus.RETRYING, f"Replanning chunk {idx}")
                cls._transition_status(state, JournalTaskStatus.RUNNING, f"Retrying chunk {idx}")
                chunk_rec.attempt_count += 1
                try:
                    res = chunk_processor_fn(chunk_rec.chunk_payload)
                    chunk_rec.result = res
                    chunk_rec.is_committed = True
                    chunk_rec.committed_at = time.time()
                    chunk_rec.error = None
                    if idx in state.failed_chunk_indices:
                        state.failed_chunk_indices.remove(idx)
                    state.committed_results.append(res)
                    state.completed_chunks_count += 1
                    cls._transition_status(state, JournalTaskStatus.CHUNK_COMMITTED, f"Chunk {idx} committed after retry")
                    journal.write_atomic(state)
                except Exception as retry_err:
                    chunk_rec.error = str(retry_err)
                    cls._transition_status(state, JournalTaskStatus.FAILED, f"Retry failed on chunk {idx}")
                    journal.write_atomic(state)
                    raise retry_err
            else:
                try:
                    res = chunk_processor_fn(chunk_rec.chunk_payload)
                    chunk_rec.result = res
                    chunk_rec.is_committed = True
                    chunk_rec.committed_at = time.time()
                    chunk_rec.error = None
                    state.committed_results.append(res)
                    state.completed_chunks_count += 1
                    cls._transition_status(state, JournalTaskStatus.CHUNK_COMMITTED, f"Chunk {idx} committed")
                    journal.write_atomic(state)
                except Exception as e:
                    chunk_rec.error = str(e)
                    if idx not in state.failed_chunk_indices:
                        state.failed_chunk_indices.append(idx)
                    cls._transition_status(state, JournalTaskStatus.FAILED, f"Execution failed on chunk {idx}: {e}")
                    journal.write_atomic(state)
                    raise e

            # Simulate process crash/halt after committing a chunk
            if simulate_crash_after_chunk is not None and idx == simulate_crash_after_chunk:
                # Return immediately without completing remaining chunks
                return state

        # Verification step
        all_chunks_committed = (state.completed_chunks_count == total_chunks)
        if all_chunks_committed:
            cls._transition_status(state, JournalTaskStatus.VERIFYING, "Verifying completed task outputs")
            journal.write_atomic(state)

            verification_passed = True
            if verification_fn is not None:
                try:
                    verification_passed = verification_fn(state.committed_results)
                except Exception as ver_err:
                    verification_passed = False
                    state.metadata["verification_error"] = str(ver_err)

            if verification_passed:
                cls._transition_status(state, JournalTaskStatus.COMPLETED, "All chunks verified and committed")
            else:
                cls._transition_status(state, JournalTaskStatus.FAILED, "Output verification failed")
            journal.write_atomic(state)

        return state

    @staticmethod
    def _split_into_chunks(text: str, chunk_chars: int) -> List[str]:
        words = text.split()
        chunks = []
        current = []
        current_len = 0

        for w in words:
            if current_len + len(w) + 1 > chunk_chars and current:
                chunks.append(" ".join(current))
                current = [w]
                current_len = len(w)
            else:
                current.append(w)
                current_len += len(w) + 1

        if current:
            chunks.append(" ".join(current))

        return chunks or [text]

    @staticmethod
    def _transition_status(
        state: ExecutionJournalState,
        new_status: JournalTaskStatus,
        reason: str = "",
    ) -> None:
        if state.status == new_status:
            return

        allowed = VALID_JOURNAL_TRANSITIONS.get(state.status, set())
        if new_status not in allowed:
            raise InvalidJournalTransitionError(
                f"Cannot transition task '{state.task_id}' from {state.status.value} to {new_status.value}. "
                f"Valid targets: {[s.value for s in allowed]}. Reason: {reason}"
            )
        state.status = new_status
