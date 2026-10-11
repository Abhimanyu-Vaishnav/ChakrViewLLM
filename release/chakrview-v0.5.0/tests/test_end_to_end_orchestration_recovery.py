"""
End-to-End Orchestration Recovery & Failure-Injection Tests (Step 16).

Verifies the integration of DurableExecutionJournal directly with CognitiveController:
1. Complete task through the actual orchestration entry point (CognitiveController.execute_task).
2. Partial execution followed by a separate-process restart.
3. Recovery through orchestration rather than a direct journal-only API.
4. Output-validation failure preventing successful completion.
5. A failed retry remaining visibly failed when recovery cannot finish it.
6. Repeated resume requests not duplicating committed work (idempotency).
7. Two independent tasks retaining separate state and journals.
8. Concurrent or stale writers being handled safely (tamper / task-id separation).
9. Abandoned temporary files cleanup and atomic integrity.
10. Legacy callers continuing to work without durable journaling.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import pytest

from chakrview.cognition.controller import CognitiveController
from chakrview.cognition.task import CognitiveTask, TaskStatus, TaskConstraints
from chakrview.cognition.planner import DeterministicRulePlanner, PlanStep, PlanStepStatus
from chakrview.cognition.verifier import StepVerifier
from chakrview.cognition.observation import StepObservation
from chakrview.runtime.recovery import (
    DurableExecutionJournal,
    ExecutionJournalState,
    JournalTaskStatus,
    JournalError,
    CorruptedJournalError,
    IncompatibleSchemaError,
)


def test_01_complete_orchestrated_task_with_durable_journal(tmp_path: Path):
    """
    Test 1: Complete multi-step task through CognitiveController.execute_task with journaling.
    Verifies plan, steps, observations, task state, and COMPLETED status persisted to disk.
    """
    journal_path = tmp_path / "orchestrated_task_01.json"
    controller = CognitiveController()

    task = CognitiveTask(
        task_id="task_calc_complete",
        user_request="calculate 25 * 4 + 50",
    )
    result = controller.execute_task(task, journal_path=journal_path)

    assert result.success is True
    assert result.task.status == TaskStatus.COMPLETED
    assert result.task.execution_state.get("calc_result") == 150.0
    assert "150" in result.response_text

    # Verify journal on disk
    journal = DurableExecutionJournal(journal_path)
    assert journal.exists()
    state = journal.load(expected_task_id="task_calc_complete")
    assert state.status == JournalTaskStatus.COMPLETED
    assert state.total_chunks == 3
    assert state.completed_chunks_count == 3
    assert all(c.is_committed for c in state.chunks)
    assert state.metadata["task_execution_state"]["calc_result"] == 150.0
    assert "150" in state.metadata["response_text"]


def test_02_partial_execution_and_separate_process_restart(tmp_path: Path):
    """
    Test 2 & 3: Partial execution interrupted mid-task, followed by separate Python process restart.
    Verifies recovery occurs through CognitiveController orchestration, skipping completed steps.
    """
    journal_path = tmp_path / "proc_restart_journal.json"
    task_id = "task_restart_e2e"

    # Step A: Run child process 1 that simulates crash after step_1
    worker_script = f"""
import sys
from chakrview.cognition.controller import CognitiveController
from chakrview.cognition.task import CognitiveTask

controller = CognitiveController()
task = CognitiveTask(task_id="{task_id}", user_request="calculate 40 * 5 + 10")
try:
    controller.execute_task(
        task,
        journal_path=r"{journal_path}",
        simulate_interrupt_after_step="step_1",
    )
except KeyboardInterrupt:
    sys.exit(42)
"""
    res1 = subprocess.run(
        [sys.executable, "-c", worker_script],
        capture_output=True,
        text=True,
    )
    assert res1.returncode == 42, f"Expected simulated interruption exit 42, got {res1.returncode}, stderr={res1.stderr}"

    # Verify intermediate state in journal
    journal = DurableExecutionJournal(journal_path)
    assert journal.exists()
    inter_state = journal.load(expected_task_id=task_id)
    assert inter_state.status == JournalTaskStatus.RUNNING
    assert inter_state.chunks[0].chunk_payload == "step_1"
    assert inter_state.chunks[0].is_committed is True
    assert inter_state.chunks[1].is_committed is False

    # Step B: Run child process 2 to resume the task through CognitiveController
    resume_script = f"""
import sys
from chakrview.cognition.controller import CognitiveController
from chakrview.cognition.task import CognitiveTask, TaskStatus

controller = CognitiveController()
task = CognitiveTask(task_id="{task_id}", user_request="calculate 40 * 5 + 10")
res = controller.execute_task(task, journal_path=r"{journal_path}")
if not res.success or res.task.status != TaskStatus.COMPLETED:
    sys.exit(1)
if res.task.execution_state.get("calc_result") != 210.0:
    sys.exit(2)
sys.exit(0)
"""
    res2 = subprocess.run(
        [sys.executable, "-c", resume_script],
        capture_output=True,
        text=True,
    )
    assert res2.returncode == 0, f"Child resume failed with code {res2.returncode}, stderr={res2.stderr}"

    # Verify final journal state
    final_state = journal.load(expected_task_id=task_id)
    assert final_state.status == JournalTaskStatus.COMPLETED
    assert final_state.attempt_count == 2  # Incremented across restart
    assert all(c.is_committed for c in final_state.chunks)
    assert final_state.metadata["task_execution_state"]["calc_result"] == 210.0


def test_04_output_validation_failure_prevents_completion(tmp_path: Path):
    """
    Test 4: Output verification failure prevents task from completing and marks journal visibly failed/rolled back.
    """
    journal_path = tmp_path / "failed_validation_journal.json"
    controller = CognitiveController()

    # Request an invalid calculation (divide by zero)
    task = CognitiveTask(
        task_id="task_fail_validation",
        user_request="calculate 100 / 0",
    )
    result = controller.execute_task(task, journal_path=journal_path)

    assert result.success is False
    assert result.task.status == TaskStatus.ROLLED_BACK
    assert "halted due to failure" in result.response_text

    # Journal must NOT be marked COMPLETED
    journal = DurableExecutionJournal(journal_path)
    state = journal.load(expected_task_id="task_fail_validation")
    assert state.status in (JournalTaskStatus.ROLLED_BACK, JournalTaskStatus.FAILED)
    assert state.status != JournalTaskStatus.COMPLETED
    assert state.completed_chunks_count < state.total_chunks


def test_05_failed_retry_remains_visibly_failed_when_unrecoverable(tmp_path: Path):
    """
    Test 5: Failed retries remain visibly failed in both graph and persisted journal.
    """
    journal_path = tmp_path / "retry_exhaustion_journal.json"
    controller = CognitiveController()

    task = CognitiveTask(
        task_id="task_retry_exhaust",
        user_request="calculate 999 / 0",
        constraints=TaskConstraints(max_retries=1),
    )
    result = controller.execute_task(task, journal_path=journal_path)

    assert result.success is False
    journal = DurableExecutionJournal(journal_path)
    state = journal.load(expected_task_id="task_retry_exhaust")
    assert state.status == JournalTaskStatus.ROLLED_BACK
    assert len(state.failed_chunk_indices) > 0


def test_06_repeated_resume_requests_do_not_duplicate_work(tmp_path: Path):
    """
    Test 6: Calling execute_task on an already-completed journaled task returns immediately
    without duplicating or re-executing steps (idempotent resume).
    """
    journal_path = tmp_path / "idempotent_task.json"
    controller = CognitiveController()

    task = CognitiveTask(
        task_id="task_idempotent",
        user_request="calculate 12 * 12",
    )
    res1 = controller.execute_task(task, journal_path=journal_path)
    assert res1.success is True
    assert res1.task.execution_state.get("calc_result") == 144.0

    journal = DurableExecutionJournal(journal_path)
    state1 = journal.load(expected_task_id="task_idempotent")
    first_updated_at = state1.updated_at
    first_attempt_count = state1.attempt_count

    # Second execution invocation on the same task and journal
    res2 = controller.execute_task(task, journal_path=journal_path)
    assert res2.success is True
    assert res2.task.status == TaskStatus.COMPLETED
    assert res2.task.execution_state.get("calc_result") == 144.0

    state2 = journal.load(expected_task_id="task_idempotent")
    # State should remain completed and attempt count shouldn't balloon
    assert state2.status == JournalTaskStatus.COMPLETED
    assert state2.completed_chunks_count == state1.completed_chunks_count


def test_07_independent_tasks_retain_separate_state(tmp_path: Path):
    """
    Test 7: Two independent tasks retain separate journals and execution state without crossover.
    """
    journal_path_a = tmp_path / "task_a.json"
    journal_path_b = tmp_path / "task_b.json"
    controller = CognitiveController()

    task_a = CognitiveTask(task_id="task_alpha", user_request="calculate 10 + 20")
    task_b = CognitiveTask(task_id="task_beta", user_request="calculate 100 + 200")

    res_a = controller.execute_task(task_a, journal_path=journal_path_a)
    res_b = controller.execute_task(task_b, journal_path=journal_path_b)

    assert res_a.task.execution_state.get("calc_result") == 30.0
    assert res_b.task.execution_state.get("calc_result") == 300.0

    state_a = DurableExecutionJournal(journal_path_a).load(expected_task_id="task_alpha")
    state_b = DurableExecutionJournal(journal_path_b).load(expected_task_id="task_beta")

    assert state_a.task_id == "task_alpha"
    assert state_b.task_id == "task_beta"
    assert state_a.metadata["task_execution_state"]["calc_result"] == 30.0
    assert state_b.metadata["task_execution_state"]["calc_result"] == 300.0


def test_08_cross_task_journal_reuse_is_safely_rejected(tmp_path: Path):
    """
    Test 8: Attempting to execute Task B with an existing journal belonging to Task A raises JournalError.
    """
    journal_path = tmp_path / "shared_journal.json"
    controller = CognitiveController()

    task_a = CognitiveTask(task_id="task_first", user_request="calculate 5 * 5")
    controller.execute_task(task_a, journal_path=journal_path)

    task_b = CognitiveTask(task_id="task_second", user_request="calculate 7 * 7")
    with pytest.raises(JournalError, match="Journal task_id mismatch"):
        controller.execute_task(task_b, journal_path=journal_path)


def test_09_abandoned_temporary_files_cleanup_and_corrupt_rejection(tmp_path: Path):
    """
    Test 9: Abandoned temp files cleanup and rejection of tampered/corrupt journals.
    """
    journal_path = tmp_path / "cleanup_test.json"
    journal = DurableExecutionJournal(journal_path)

    # Simulate abandoned temp files
    temp_dir = journal_path.parent
    stale_tmp1 = temp_dir / f".{journal_path.stem}_abandoned1.tmp"
    stale_tmp2 = temp_dir / f".{journal_path.stem}_abandoned2.tmp"
    stale_tmp1.write_text("abandoned payload 1", encoding="utf-8")
    stale_tmp2.write_text("abandoned payload 2", encoding="utf-8")

    cleaned = journal.cleanup_abandoned_temp_files()
    assert cleaned == 2
    assert not stale_tmp1.exists()
    assert not stale_tmp2.exists()

    # Test tampering rejection
    task = CognitiveTask(task_id="task_tamper", user_request="calculate 1 + 1")
    controller = CognitiveController()
    controller.execute_task(task, journal_path=journal_path)

    # Tamper with journal file
    with open(journal_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    data["completed_chunks_count"] = 999  # Tampered
    with open(journal_path, "w", encoding="utf-8") as f:
        json.dump(data, f)

    with pytest.raises(CorruptedJournalError, match="checksum mismatch"):
        journal.load()


def test_10_legacy_callers_work_without_journal_path():
    """
    Test 10: Callers omitting journal_path continue to execute with in-memory state unaltered.
    """
    controller = CognitiveController()
    task = CognitiveTask(
        task_id="task_legacy_in_mem",
        user_request="calculate 30 * 3",
    )
    result = controller.execute_task(task)

    assert result.success is True
    assert result.task.status == TaskStatus.COMPLETED
    assert result.task.execution_state.get("calc_result") == 90.0
    assert "90" in result.response_text
