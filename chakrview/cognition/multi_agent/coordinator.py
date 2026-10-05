"""
ChakrView Step 113: Governed Multi-Agent Coordinator and Protocol.

Orchestrates multi-worker task execution over PersistentTaskGraph:
- Shared PPB persistence (SQLite)
- DAG-based dependency scheduling (independent tasks can run in parallel)
- Contract enforcement and isolation
- Reviewer critique handling & revision routing
- Worker failure rerouting
- Final synthesis
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from chakrview.cognition.multi_agent.contracts import (
    WorkerRole,
    WorkerContract,
    WorkerPackage,
    ReviewerVerdict,
)
from chakrview.cognition.multi_agent.result import (
    WorkerResult,
    WorkerExecutionStatus,
    SynthesisResult,
)
from chakrview.cognition.multi_agent.worker import (
    BaseCognitiveWorker,
    ProjectAnalystWorker,
    PlannerWorker,
    ImplementerWorker,
    TestEngineerWorker,
    ReviewerWorker,
)
from chakrview.cognition.ppb.task_models import (
    PersistentTaskGraph,
    PersistentTaskNode,
    TaskNodeStatus,
    TaskResourceType,
)
from chakrview.cognition.tool_gate import (
    GovernedToolGate,
    ToolAuthorizationError,
    ArgumentValidationError,
)


class MultiAgentCoordinator:
    """
    Coordinates multi-agent workflows using PersistentTaskGraph and SQLite PPB.
    Guarantees no worker receives full repository or unassigned authority.
    """

    def __init__(
        self,
        db_path: Path,
        workspace_root: Path,
        tool_gate: GovernedToolGate,
    ) -> None:
        self.db_path = Path(db_path)
        self.workspace_root = Path(workspace_root)
        self.tool_gate = tool_gate
        self.workers: Dict[str, BaseCognitiveWorker] = {}
        self._init_ppb_tables()

    def _init_ppb_tables(self) -> None:
        """Initialize tables for multi-agent coordination in SQLite PPB."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS worker_executions (
                    task_id TEXT PRIMARY KEY,
                    worker_id TEXT,
                    role TEXT,
                    status TEXT,
                    context_tokens_used INTEGER,
                    evidence_json TEXT,
                    error TEXT,
                    timestamp REAL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS multi_agent_audit (
                    audit_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type TEXT,
                    task_id TEXT,
                    worker_id TEXT,
                    details_json TEXT,
                    timestamp REAL
                )
            """)
            conn.commit()

    def register_worker(self, worker: BaseCognitiveWorker) -> None:
        """Register a worker instance."""
        self.workers[worker.worker_id] = worker

    def record_audit(self, event_type: str, task_id: str, worker_id: str, details: Dict[str, Any]) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO multi_agent_audit (event_type, task_id, worker_id, details_json, timestamp) VALUES (?, ?, ?, ?, ?)",
                (event_type, task_id, worker_id, json.dumps(details), time.time()),
            )
            conn.commit()

    def persist_worker_result(self, result: WorkerResult) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO worker_executions 
                (task_id, worker_id, role, status, context_tokens_used, evidence_json, error, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                result.task_id,
                result.worker_id,
                result.role.value,
                result.status.value,
                result.context_tokens_used,
                json.dumps(result.evidence),
                result.error,
                time.time(),
            ))
            conn.commit()

    def get_persisted_result(self, task_id: str) -> Optional[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT worker_id, role, status, context_tokens_used, evidence_json, error FROM worker_executions WHERE task_id = ?", (task_id,))
            row = cursor.fetchone()
            if row:
                return {
                    "task_id": task_id,
                    "worker_id": row[0],
                    "role": row[1],
                    "status": row[2],
                    "context_tokens_used": row[3],
                    "evidence": json.loads(row[4]),
                    "error": row[5],
                }
            return None

    def execute_worker_task(
        self,
        worker_id: str,
        contract: WorkerContract,
        package: WorkerPackage,
        fallback_worker_id: Optional[str] = None,
    ) -> WorkerResult:
        """
        Executes a task under an explicit contract. Supports worker rerouting if primary is unavailable.
        """
        # Validate contract first
        try:
            contract.validate()
        except Exception as e:
            self.record_audit("CONTRACT_VIOLATION", contract.task_id, worker_id, {"error": str(e)})
            return WorkerResult(
                worker_id=worker_id,
                task_id=contract.task_id,
                role=contract.role,
                status=WorkerExecutionStatus.FAILED,
                error=f"Contract validation rejected: {str(e)}",
            )

        worker = self.workers.get(worker_id)
        if worker is None:
            if fallback_worker_id and fallback_worker_id in self.workers:
                self.record_audit("WORKER_REROUTE", contract.task_id, worker_id, {"rerouted_to": fallback_worker_id})
                worker = self.workers[fallback_worker_id]
                contract.worker_id = fallback_worker_id
            else:
                return WorkerResult(
                    worker_id=worker_id,
                    task_id=contract.task_id,
                    role=contract.role,
                    status=WorkerExecutionStatus.FAILED,
                    error=f"Worker {worker_id} not available and no fallback",
                )

        # Enforce context limits
        if package.measured_context_tokens > contract.context_budget:
            return WorkerResult(
                worker_id=worker.worker_id,
                task_id=contract.task_id,
                role=contract.role,
                status=WorkerExecutionStatus.FAILED,
                error=f"Context budget exceeded: {package.measured_context_tokens} > {contract.context_budget}",
            )

        # Execute
        result = worker.execute(contract, package)

        # Validate schema if required
        try:
            result.validate_against_schema(contract.expected_output_schema)
        except Exception as e:
            result.status = WorkerExecutionStatus.MALFORMED_OUTPUT
            result.error = f"Malformed worker output schema: {str(e)}"

        # Persist result to PPB
        self.persist_worker_result(result)
        self.record_audit("WORKER_COMPLETED", contract.task_id, result.worker_id, {"status": result.status.value})
        return result
