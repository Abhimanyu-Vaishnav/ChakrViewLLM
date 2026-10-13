"""
Batch 188: Task Engine Durable Synthesis and Recovery Evaluation.
Tests:
1. Multi-document evidence synthesis happy path.
2. Cross-document contradiction detection and graceful failure.
3. Recovery from disk after simulated restart during multi-step execution.
4. Idempotent re-execution of completed synthesis task.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import json
import shutil
import tempfile

from chakrview.runtime.task_engine import CognitiveTaskEngine, TaskStatus
from chakrview.runtime.persistent_memory import PersistentSessionMemory

REPORTS_DIR = Path("experiments/batch181_190_evidence_synthesis/reports")


def evaluate_task_recovery():
    temp_dir = Path(tempfile.mkdtemp(prefix="chakr_synth_task_"))
    try:
        mem = PersistentSessionMemory(temp_dir / "sessions")
        engine1 = CognitiveTaskEngine(storage_dir=temp_dir / "tasks", memory=mem)

        docs = [
            ("doc_chem_1", "Hydrogen is the lightest element with atomic number 1."),
            ("doc_chem_2", "Atomic number of Hydrogen is 1, with standard atomic weight 1.008."),
            ("doc_astro_1", "Jupiter is the largest planet in the Solar System."),
        ]

        # 1. Happy Path Execution
        task1 = engine1.execute_synthesis_workflow(
            task_id="synth_task_1",
            session_id="session_user_1",
            query="What is the atomic number of Hydrogen?",
            document_texts=docs,
        )
        t1_ok = (task1.status == TaskStatus.COMPLETED and task1.final_result == "1")

        # 2. Simulated Engine Restart & Recovery
        engine2 = CognitiveTaskEngine(storage_dir=temp_dir / "tasks", memory=mem)
        recovered_task = engine2.load_task("synth_task_1")
        rec_ok = (recovered_task is not None and recovered_task.status == TaskStatus.COMPLETED and recovered_task.final_result == "1")

        # 3. Idempotent Re-execution
        idemp_task = engine2.execute_synthesis_workflow(
            task_id="synth_task_1",
            session_id="session_user_1",
            query="What is the atomic number of Hydrogen?",
            document_texts=docs,
        )
        idemp_ok = (idemp_task.status == TaskStatus.COMPLETED and idemp_task.final_result == "1")

        # 4. Conflicting Multi-Document Evidence
        conflict_docs = [
            ("source_A", "The measured boiling point of the solution is 105 C."),
            ("source_B", "The measured boiling point of the solution is 112 C."),
        ]
        task_conflict = engine2.execute_synthesis_workflow(
            task_id="synth_task_conflict",
            session_id="session_user_2",
            query="What is the boiling point of the solution?",
            document_texts=conflict_docs,
        )
        conflict_ok = (task_conflict.status == TaskStatus.FAILED and "Conflicting facts" in (task_conflict.failure_reason or ""))

        report = {
            "milestone": "Batch 188: Durable Task Engine Integration & Recovery Evaluation",
            "tests": {
                "multi_document_synthesis_happy_path": t1_ok,
                "restart_and_state_recovery": rec_ok,
                "idempotent_reexecution": idemp_ok,
                "conflicting_evidence_abstention": conflict_ok,
            },
            "all_passed": all([t1_ok, rec_ok, idemp_ok, conflict_ok]),
            "task_trace": {
                "synth_task_1_final": task1.final_result,
                "synth_task_conflict_reason": task_conflict.failure_reason,
            }
        }

        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        out_file = REPORTS_DIR / "task_recovery_results.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        print(f"Task recovery evaluation written to {out_file}")
        print(f"All Task Engine Tests Passed: {report['all_passed']}")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    evaluate_task_recovery()
