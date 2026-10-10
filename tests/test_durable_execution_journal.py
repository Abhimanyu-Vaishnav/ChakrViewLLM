"""
Comprehensive Deterministic Failure-Injection & Recovery Tests for ChakrView.

Covers:
1. Normal execution and persisted completion.
2. Process restart after partial progress (multi-process integration test).
3. Recovery of pending work without losing completed progress.
4. Interruption during journal writing (simulated atomic crash with leftover .tmp file).
5. Corrupt or incompatible journal data (bad JSON, tampered checksum, mismatched schema version).
6. Duplicate retry attempts and idempotency (already committed chunks are not re-executed).
7. Task failure followed by successful replanning/recovery.
8. Resource-limit enforcement during resumed execution.
9. Invalid state transitions and attempts to mark incomplete work as complete.
10. Backward compatibility with existing in-memory recovery behavior.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import pytest

from chakrview.runtime.hardware_adaptive_engine import (
    HardwareDiscovery,
    HardwareAdaptiveGovernor,
    ChunkedAdaptiveExecutor,
    HardwareExecutionProfile,
    ExecutionBudgetMode,
)
from chakrview.runtime.recovery import (
    DurableExecutionJournal,
    DurableChunkedExecutor,
    ExecutionJournalState,
    JournalTaskStatus,
    CorruptedJournalError,
    IncompatibleSchemaError,
    InvalidJournalTransitionError,
    JOURNAL_SCHEMA_VERSION,
)


def test_01_normal_execution_and_persisted_completion(tmp_path: Path):
    """Test 1: Normal execution completes all chunks and persists valid journal."""
    journal_file = tmp_path / "journal_01.json"
    env = HardwareDiscovery.probe()
    budget = HardwareAdaptiveGovernor.select_profile(env)
    budget.default_chunk_chars = 35

    payload = "Chunk alpha payload. Chunk beta payload. Chunk gamma payload."
    executed_chunks = []

    def mock_processor(chunk: str) -> str:
        executed_chunks.append(chunk)
        return chunk.upper()

    state = DurableChunkedExecutor.execute(
        task_id="task_normal_01",
        text=payload,
        chunk_processor_fn=mock_processor,
        budget=budget,
        journal_path=journal_file,
    )

    assert state.status == JournalTaskStatus.COMPLETED
    assert state.completed_chunks_count == state.total_chunks
    assert len(state.committed_results) == state.total_chunks
    assert journal_file.is_file()

    # Verify journal reloads cleanly from disk
    loaded_state = DurableExecutionJournal(journal_file).load()
    assert loaded_state.task_id == "task_normal_01"
    assert loaded_state.status == JournalTaskStatus.COMPLETED
    assert loaded_state.completed_chunks_count == state.total_chunks
    assert loaded_state.committed_results == [c.upper() for c in executed_chunks]


def test_02_recovery_of_pending_work_without_losing_completed_progress(tmp_path: Path):
    """Test 2 & 3: Simulate mid-task interruption and verify resume recovers without re-executing committed work."""
    journal_file = tmp_path / "journal_resume.json"
    env = HardwareDiscovery.probe()
    budget = HardwareAdaptiveGovernor.select_profile(env)
    budget.default_chunk_chars = 20

    payload = "First section data. Second section data. Third section data. Fourth section data."
    executed_run_1 = []

    def processor_run_1(chunk: str) -> str:
        executed_run_1.append(chunk)
        return f"PROCESSED: {chunk}"

    # Crash after chunk index 1 (so chunks 0 and 1 are committed, 2 and 3 remain pending)
    interrupted_state = DurableChunkedExecutor.execute(
        task_id="task_resumable_01",
        text=payload,
        chunk_processor_fn=processor_run_1,
        budget=budget,
        journal_path=journal_file,
        simulate_crash_after_chunk=1,
    )

    assert interrupted_state.status == JournalTaskStatus.CHUNK_COMMITTED
    assert interrupted_state.completed_chunks_count == 2
    assert len(executed_run_1) == 2

    # Now simulate process recovery: resume execution on same journal
    executed_run_2 = []

    def processor_run_2(chunk: str) -> str:
        executed_run_2.append(chunk)
        return f"PROCESSED: {chunk}"

    resumed_state = DurableChunkedExecutor.execute(
        task_id="task_resumable_01",
        text=payload,
        chunk_processor_fn=processor_run_2,
        budget=budget,
        journal_path=journal_file,
    )

    assert resumed_state.status == JournalTaskStatus.COMPLETED
    assert resumed_state.completed_chunks_count == 4
    assert resumed_state.attempt_count == 2  # Incremented attempt counter
    # Idempotency check: processor_run_2 was ONLY called for pending chunks (2 and 3), NOT 0 and 1
    assert len(executed_run_2) == 2
    assert len(resumed_state.committed_results) == 4


def test_03_real_separate_process_restart_recovery(tmp_path: Path):
    """Test 2 (Real subprocess): Run partial execution in child process, crash it, and resume in second child process."""
    journal_file = tmp_path / "subproc_journal.json"
    payload_file = tmp_path / "payload.txt"
    payload = "Alpha section content. Beta section content. Gamma section content. Delta section content."
    payload_file.write_text(payload, encoding="utf-8")

    # Script for Subprocess A: execute with crash after chunk 1
    subproc_script_a = tmp_path / "run_subproc_a.py"
    subproc_script_a.write_text(f"""
import sys
from pathlib import Path
REPO_ROOT = Path(r"{Path(__file__).resolve().parent.parent}")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chakrview.runtime.hardware_adaptive_engine import HardwareDiscovery, HardwareAdaptiveGovernor
from chakrview.runtime.recovery import DurableChunkedExecutor

env = HardwareDiscovery.probe()
budget = HardwareAdaptiveGovernor.select_profile(env)
budget.default_chunk_chars = 25

text = Path(r"{payload_file}").read_text(encoding="utf-8")
res = DurableChunkedExecutor.execute(
    task_id="task_subproc_01",
    text=text,
    chunk_processor_fn=lambda c: f"OUT: {{c}}",
    budget=budget,
    journal_path=r"{journal_file}",
    simulate_crash_after_chunk=1,
)
print("SUBPROC_A_COMPLETED_CHUNKS:", res.completed_chunks_count)
""", encoding="utf-8")

    # Run Subprocess A
    proc_a = subprocess.run(
        [sys.executable, str(subproc_script_a)],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert proc_a.returncode == 0
    assert "SUBPROC_A_COMPLETED_CHUNKS: 2" in proc_a.stdout
    assert journal_file.is_file()

    # Script for Subprocess B: resume and complete task
    subproc_script_b = tmp_path / "run_subproc_b.py"
    subproc_script_b.write_text(f"""
import sys
from pathlib import Path
REPO_ROOT = Path(r"{Path(__file__).resolve().parent.parent}")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chakrview.runtime.hardware_adaptive_engine import HardwareDiscovery, HardwareAdaptiveGovernor
from chakrview.runtime.recovery import DurableChunkedExecutor

env = HardwareDiscovery.probe()
budget = HardwareAdaptiveGovernor.select_profile(env)
budget.default_chunk_chars = 25

text = Path(r"{payload_file}").read_text(encoding="utf-8")
res = DurableChunkedExecutor.execute(
    task_id="task_subproc_01",
    text=text,
    chunk_processor_fn=lambda c: f"OUT: {{c}}",
    budget=budget,
    journal_path=r"{journal_file}",
)
print("SUBPROC_B_FINAL_STATUS:", res.status.value)
print("SUBPROC_B_COMPLETED_CHUNKS:", res.completed_chunks_count)
""", encoding="utf-8")

    # Run Subprocess B
    proc_b = subprocess.run(
        [sys.executable, str(subproc_script_b)],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert proc_b.returncode == 0
    assert "SUBPROC_B_FINAL_STATUS: COMPLETED" in proc_b.stdout
    assert "SUBPROC_B_COMPLETED_CHUNKS: 4" in proc_b.stdout

    # Verify journal file in parent process
    final_journal = DurableExecutionJournal(journal_file).load()
    assert final_journal.status == JournalTaskStatus.COMPLETED
    assert final_journal.completed_chunks_count == 4


def test_04_interruption_during_journal_writing_leaves_valid_prior_state(tmp_path: Path):
    """Test 4: Interruption during atomic write (simulated leftover temp file) does not corrupt active journal."""
    journal_file = tmp_path / "journal_atomic.json"
    env = HardwareDiscovery.probe()
    budget = HardwareAdaptiveGovernor.select_profile(env)
    budget.default_chunk_chars = 40

    payload = "Short data block for atomic test."
    state = DurableChunkedExecutor.execute(
        task_id="task_atomic_01",
        text=payload,
        chunk_processor_fn=lambda c: c.lower(),
        budget=budget,
        journal_path=journal_file,
    )
    assert journal_file.is_file()
    valid_content = journal_file.read_text(encoding="utf-8")

    # Simulate an abandoned partial/corrupt temp file created by interrupted write
    orphan_tmp = tmp_path / ".journal_atomic_orphan.tmp"
    orphan_tmp.write_text("CORRUPTED_INCOMPLETE_WRITE_DATA", encoding="utf-8")

    # Journal loader must still cleanly load the valid journal_atomic.json without failing
    journal = DurableExecutionJournal(journal_file)
    loaded = journal.load()
    assert loaded.task_id == "task_atomic_01"
    assert loaded.status == JournalTaskStatus.COMPLETED


def test_05_corrupt_or_incompatible_journal_data_handling(tmp_path: Path):
    """Test 5: Handle corrupted JSON, tampered checksum, and incompatible schema version."""
    journal_file = tmp_path / "journal_corrupt.json"

    # Case A: Corrupted JSON syntax
    journal_file.write_text("{malformed: json, [unclosed", encoding="utf-8")
    journal = DurableExecutionJournal(journal_file)
    with pytest.raises(CorruptedJournalError, match="invalid JSON"):
        journal.load()

    # Case B: Incompatible schema version
    bad_schema_payload = {
        "task_id": "bad_task",
        "schema_version": "99.0.0",  # Incompatible
        "status": "COMPLETED",
        "total_chunks": 1,
        "completed_chunks_count": 1,
        "chunks": [],
        "committed_results": [],
        "failed_chunk_indices": [],
        "attempt_count": 1,
        "peak_working_bytes": 10,
        "created_at": 1000.0,
        "updated_at": 1000.0,
        "budget_summary": "test",
        "metadata": {},
        "checksum": "fake",
    }
    journal_file.write_text(json.dumps(bad_schema_payload), encoding="utf-8")
    with pytest.raises(IncompatibleSchemaError, match="Incompatible journal schema version"):
        journal.load()

    # Case C: Tampered checksum / integrity violation
    tampered_payload = dict(bad_schema_payload)
    tampered_payload["schema_version"] = JOURNAL_SCHEMA_VERSION
    tampered_payload["checksum"] = "bad_checksum_hash_0000000000000000"
    journal_file.write_text(json.dumps(tampered_payload), encoding="utf-8")
    with pytest.raises(CorruptedJournalError, match="Journal checksum mismatch"):
        journal.load()


def test_06_idempotent_retry_and_duplicate_deduplication(tmp_path: Path):
    """Test 6: Idempotent execution: calling executor on completed task does not duplicate results."""
    journal_file = tmp_path / "journal_idempotent.json"
    env = HardwareDiscovery.probe()
    budget = HardwareAdaptiveGovernor.select_profile(env)
    budget.default_chunk_chars = 30

    payload = "Data piece one. Data piece two."
    call_counts = {"count": 0}

    def counting_processor(c: str) -> str:
        call_counts["count"] += 1
        return c.upper()

    # Run 1: completes 2 chunks
    state1 = DurableChunkedExecutor.execute(
        task_id="task_idempotent_01",
        text=payload,
        chunk_processor_fn=counting_processor,
        budget=budget,
        journal_path=journal_file,
    )
    assert call_counts["count"] == 2
    assert len(state1.committed_results) == 2

    # Run 2: execute again on already completed journal
    state2 = DurableChunkedExecutor.execute(
        task_id="task_idempotent_01",
        text=payload,
        chunk_processor_fn=counting_processor,
        budget=budget,
        journal_path=journal_file,
    )

    # Counting processor must NOT be invoked again
    assert call_counts["count"] == 2
    assert len(state2.committed_results) == 2
    assert state2.status == JournalTaskStatus.COMPLETED


def test_07_task_failure_followed_by_successful_recovery(tmp_path: Path):
    """Test 7: Chunk transient failure records failure, replans, and completes successfully."""
    journal_file = tmp_path / "journal_fail_recover.json"
    env = HardwareDiscovery.probe()
    budget = HardwareAdaptiveGovernor.select_profile(env)
    budget.default_chunk_chars = 15

    payload = "First section. Second section. Third section."

    state = DurableChunkedExecutor.execute(
        task_id="task_fault_01",
        text=payload,
        chunk_processor_fn=lambda c: f"VERIFIED: {c}",
        budget=budget,
        journal_path=journal_file,
        simulate_failure_chunk=1,  # Fault on chunk index 1
    )

    assert state.status == JournalTaskStatus.COMPLETED
    assert state.completed_chunks_count == 3
    # Chunk 1 attempt count must be 2 due to retry
    assert state.chunks[1].attempt_count == 2
    assert state.chunks[1].error is None


def test_08_resource_budget_governance_in_recovery(tmp_path: Path):
    """Test 8: Ensure resource governor boundaries are preserved during resumed execution."""
    journal_file = tmp_path / "journal_budget.json"
    env = HardwareDiscovery.probe()

    # User active workload triggers LOW_RESOURCE throttled budget
    throttled_budget = HardwareAdaptiveGovernor.select_profile(env, user_workload_active=True)
    assert throttled_budget.profile == HardwareExecutionProfile.LOW_RESOURCE
    assert throttled_budget.allocated_threads == 1
    assert throttled_budget.yield_priority_to_user is True

    payload = "Short throttled payload chunk."
    state = DurableChunkedExecutor.execute(
        task_id="task_throttled_01",
        text=payload,
        chunk_processor_fn=lambda c: c,
        budget=throttled_budget,
        journal_path=journal_file,
    )

    assert state.status == JournalTaskStatus.COMPLETED
    assert "LOW_RESOURCE" in state.budget_summary or "throttling" in state.budget_summary.lower()


def test_09_invalid_state_transitions_and_verification_rejection(tmp_path: Path):
    """Test 9: Invalid state transition raises error, and failing output verification rejects completion."""
    journal_file = tmp_path / "journal_invalid.json"
    env = HardwareDiscovery.probe()
    budget = HardwareAdaptiveGovernor.select_profile(env)

    # Subtest A: Failed verification prevents task completion
    def failing_verifier(results):
        return False  # Verification assertion fails

    state = DurableChunkedExecutor.execute(
        task_id="task_fail_verif_01",
        text="Payload for verification failure test.",
        chunk_processor_fn=lambda c: c,
        budget=budget,
        journal_path=journal_file,
        verification_fn=failing_verifier,
    )
    # Status must NOT be COMPLETED because verification failed
    assert state.status == JournalTaskStatus.FAILED

    # Subtest B: Illegal state transition directly on state
    with pytest.raises(InvalidJournalTransitionError, match="Cannot transition task"):
        # FAILED directly to COMPLETED is illegal (must go through RETRYING or ROLLED_BACK)
        DurableChunkedExecutor._transition_status(state, JournalTaskStatus.COMPLETED)


def test_10_backward_compatibility_with_in_memory_chunked_executor(tmp_path: Path):
    """Test 10: In-memory ChunkedAdaptiveExecutor continues to function without journal_path."""
    env = HardwareDiscovery.probe()
    budget = HardwareAdaptiveGovernor.select_profile(env)
    budget.default_chunk_chars = 12

    payload = "Part one. Part two. Part three."
    record = ChunkedAdaptiveExecutor.execute_with_checkpointing(
        task_id="in_memory_compat_01",
        text=payload,
        chunk_processor_fn=lambda c: c.upper(),
        budget=budget,
        simulate_failure_chunk=1,
    )

    assert record.is_complete is True
    assert record.total_chunks == 3
    assert record.completed_chunks == 3
    assert len(record.committed_results) == 3
    assert record.replanned_count == 1
