"""
ChakrView Step 114: Federated Process-Isolated Multi-Node Scheduler.

Extends MultiAgentCoordinator to support genuinely independent OS worker processes:
- Process spawning and communication via transport envelopes
- Strict serialization and context hash validation
- Process crash detection (nonzero returncode, abrupt exit)
- Worker timeout detection
- Dynamic rerouting to fallback workers
- Concurrency tracking and parallel execution of independent tasks
- Duplicate execution prevention with idempotency keys
- State persistence to PPB SQLite
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from chakrview.cognition.multi_agent.contracts import (
    WorkerRole,
    WorkerContract,
    WorkerPackage,
)
from chakrview.cognition.multi_agent.result import (
    WorkerResult,
    WorkerExecutionStatus,
)
from chakrview.cognition.multi_agent.transport import (
    PROTOCOL_VERSION_V1,
    RequestEnvelope,
    ResponseEnvelope,
    EnvelopeType,
)
from chakrview.cognition.multi_agent.coordinator import MultiAgentCoordinator
from chakrview.cognition.tool_gate import GovernedToolGate


@dataclass
class FederatedWorkerMetadata:
    """Metadata describing a registered federated worker process."""
    worker_id: str
    role: WorkerRole
    allowed_tools: List[str]
    max_concurrency: int = 1
    timeout_seconds: float = 10.0


class FederatedScheduler(MultiAgentCoordinator):
    """
    Extends MultiAgentCoordinator with process-isolated worker execution,
    concurrent DAG dispatch, timeout handling, and automatic rerouting.
    """

    def __init__(
        self,
        db_path: Path,
        workspace_root: Path,
        tool_gate: GovernedToolGate,
        runtime_script_path: Optional[Path] = None,
    ) -> None:
        super().__init__(db_path, workspace_root, tool_gate)
        self.federated_registry: Dict[str, FederatedWorkerMetadata] = {}
        if runtime_script_path is None:
            self.runtime_script_path = (
                Path(__file__).parent / "process_runtime.py"
            ).resolve()
        else:
            self.runtime_script_path = Path(runtime_script_path).resolve()
        self._init_federation_ppb_tables()

    def _init_federation_ppb_tables(self) -> None:
        """Initialize tables for process federation in SQLite PPB."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS federated_workers (
                    worker_id TEXT PRIMARY KEY,
                    role TEXT,
                    allowed_tools_json TEXT,
                    timeout_seconds REAL,
                    registered_at REAL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS federated_task_ownership (
                    task_id TEXT PRIMARY KEY,
                    worker_id TEXT,
                    package_id TEXT,
                    idempotency_key TEXT UNIQUE,
                    status TEXT,
                    context_hash TEXT,
                    context_token_count INTEGER,
                    started_at REAL,
                    completed_at REAL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS governed_failures (
                    failure_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT,
                    worker_id TEXT,
                    failure_type TEXT,
                    details_json TEXT,
                    timestamp REAL
                )
            """)
            conn.commit()

    def register_federated_worker(self, metadata: FederatedWorkerMetadata) -> None:
        """Register a worker capable of running as an isolated OS process."""
        self.federated_registry[metadata.worker_id] = metadata
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO federated_workers 
                (worker_id, role, allowed_tools_json, timeout_seconds, registered_at)
                VALUES (?, ?, ?, ?, ?)
            """, (
                metadata.worker_id,
                metadata.role.value,
                json.dumps(metadata.allowed_tools),
                metadata.timeout_seconds,
                time.time(),
            ))
            conn.commit()

    def record_governed_failure(self, task_id: str, worker_id: str, failure_type: str, details: Dict[str, Any]) -> None:
        """Record an audited failure event in PPB."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO governed_failures (task_id, worker_id, failure_type, details_json, timestamp) VALUES (?, ?, ?, ?, ?)",
                (task_id, worker_id, failure_type, json.dumps(details), time.time()),
            )
            conn.commit()

    def execute_in_isolated_process(
        self,
        worker_id: str,
        contract: WorkerContract,
        package: WorkerPackage,
        timeout: Optional[float] = None,
        inject_crash: bool = False,
    ) -> WorkerResult:
        """
        Spawns an independent OS process, delivers the serialized RequestEnvelope,
        reads the ResponseEnvelope, and returns the validated WorkerResult.
        """
        # Validate contract
        try:
            contract.validate()
        except Exception as e:
            self.record_governed_failure(contract.task_id, worker_id, "CONTRACT_VIOLATION", {"error": str(e)})
            return WorkerResult(
                worker_id=worker_id,
                task_id=contract.task_id,
                role=contract.role,
                status=WorkerExecutionStatus.FAILED,
                error=f"Contract rejected: {str(e)}",
            )

        # Check worker registry & capability
        meta = self.federated_registry.get(worker_id)
        if not meta:
            return WorkerResult(
                worker_id=worker_id,
                task_id=contract.task_id,
                role=contract.role,
                status=WorkerExecutionStatus.FAILED,
                error=f"Worker {worker_id} not registered in federation",
            )
        if meta.role != contract.role:
            self.record_governed_failure(contract.task_id, worker_id, "CAPABILITY_MISMATCH", {
                "worker_role": meta.role.value,
                "contract_role": contract.role.value,
            })
            return WorkerResult(
                worker_id=worker_id,
                task_id=contract.task_id,
                role=contract.role,
                status=WorkerExecutionStatus.FAILED,
                error=f"Worker capability mismatch: worker is {meta.role.value}, requires {contract.role.value}",
            )

        # Check context ceiling
        if package.measured_context_tokens > contract.context_budget:
            self.record_governed_failure(contract.task_id, worker_id, "CONTEXT_OVERFLOW", {
                "measured": package.measured_context_tokens,
                "budget": contract.context_budget,
            })
            return WorkerResult(
                worker_id=worker_id,
                task_id=contract.task_id,
                role=contract.role,
                status=WorkerExecutionStatus.FAILED,
                error=f"Context ceiling exceeded: {package.measured_context_tokens} > {contract.context_budget}",
            )

        # Duplicate task submission protection (Idempotency key)
        idempotency_key = f"{contract.task_id}:{worker_id}:{package.task_id}"
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT status FROM federated_task_ownership WHERE task_id = ? AND status IN ('COMPLETED', 'SUCCESS')", (contract.task_id,))
            if cursor.fetchone():
                self.record_governed_failure(contract.task_id, worker_id, "DUPLICATE_SUBMISSION", {"key": idempotency_key})
                return WorkerResult(
                    worker_id=worker_id,
                    task_id=contract.task_id,
                    role=contract.role,
                    status=WorkerExecutionStatus.FAILED,
                    error=f"Duplicate execution blocked: task {contract.task_id} already completed",
                )


        # Compute package context hash
        pkg_serialized = {
            "task_id": package.task_id,
            "objective": package.objective,
            "targeted_code_context": package.targeted_code_context,
            "dependency_outputs": package.dependency_outputs,
            "relevant_ppb_records": package.relevant_ppb_records,
            "acceptance_criteria": package.acceptance_criteria,
            "measured_context_tokens": package.measured_context_tokens,
        }
        context_hash = hashlib.sha256(json.dumps(pkg_serialized, sort_keys=True).encode("utf-8")).hexdigest()
        package_id = f"pkg_{context_hash[:8]}"

        # Record task assignment
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO federated_task_ownership
                (task_id, worker_id, package_id, idempotency_key, status, context_hash, context_token_count, started_at, completed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                contract.task_id,
                worker_id,
                package_id,
                idempotency_key,
                "RUNNING",
                context_hash,
                package.measured_context_tokens,
                time.time(),
                None,
            ))
            conn.commit()

        # Build RequestEnvelope
        req_env = RequestEnvelope(
            protocol_version=PROTOCOL_VERSION_V1,
            worker_id=worker_id,
            task_id=contract.task_id,
            package_id=package_id,
            role=contract.role.value,
            context_hash=context_hash,
            context_token_count=package.measured_context_tokens,
            allowed_tools=list(contract.allowed_tools),
            allowed_files=list(contract.allowed_files),
            context_budget=contract.context_budget,
            package_payload=pkg_serialized,
            expected_output_schema=contract.expected_output_schema,
        )
        req_env.validate()

        effective_timeout = timeout or meta.timeout_seconds
        python_exe = sys.executable

        # If injecting crash, run command that exits immediately with code 139 or simulates crash
        if inject_crash:
            cmd = [python_exe, "-c", "import sys; sys.exit(139)"]
        else:
            cmd = [python_exe, str(self.runtime_script_path), str(self.workspace_root)]

        t_start = time.perf_counter()
        try:
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            raw_input = json.dumps(req_env.to_dict())
            stdout_data, stderr_data = proc.communicate(input=raw_input, timeout=effective_timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            self.record_governed_failure(contract.task_id, worker_id, "WORKER_TIMEOUT", {"timeout_seconds": effective_timeout})
            return WorkerResult(
                worker_id=worker_id,
                task_id=contract.task_id,
                role=contract.role,
                status=WorkerExecutionStatus.FAILED,
                error=f"Worker process timed out after {effective_timeout}s",
            )
        except Exception as e:
            self.record_governed_failure(contract.task_id, worker_id, "PROCESS_SPAWN_ERROR", {"error": str(e)})
            return WorkerResult(
                worker_id=worker_id,
                task_id=contract.task_id,
                role=contract.role,
                status=WorkerExecutionStatus.FAILED,
                error=f"Process execution error: {str(e)}",
            )

        # Check return code for process crash
        if proc.returncode != 0:
            self.record_governed_failure(contract.task_id, worker_id, "PROCESS_CRASH", {
                "returncode": proc.returncode,
                "stderr": stderr_data,
            })
            return WorkerResult(
                worker_id=worker_id,
                task_id=contract.task_id,
                role=contract.role,
                status=WorkerExecutionStatus.FAILED,
                error=f"Worker process crashed with returncode {proc.returncode}: {stderr_data.strip()}",
            )

        # Parse response envelope
        try:
            resp_dict = json.loads(stdout_data)
            resp_env = ResponseEnvelope.from_dict(resp_dict)
            resp_env.validate()
        except Exception as e:
            self.record_governed_failure(contract.task_id, worker_id, "MALFORMED_RESPONSE", {
                "error": str(e),
                "raw_output": stdout_data[:500],
            })
            return WorkerResult(
                worker_id=worker_id,
                task_id=contract.task_id,
                role=contract.role,
                status=WorkerExecutionStatus.MALFORMED_OUTPUT,
                error=f"Malformed response envelope: {str(e)}",
            )

        # Reconstruct WorkerResult
        res_payload = resp_env.result_payload
        result = WorkerResult(
            worker_id=res_payload.get("worker_id", worker_id),
            task_id=res_payload.get("task_id", contract.task_id),
            role=contract.role,
            status=WorkerExecutionStatus(res_payload.get("status", "SUCCESS")),
            evidence=res_payload.get("evidence", {}),
            proposed_changes=res_payload.get("proposed_changes", {}),
            verification_output=res_payload.get("verification_output"),
            failures=res_payload.get("failures", []),
            confidence=res_payload.get("confidence", 1.0),
            context_tokens_used=res_payload.get("context_tokens_used", package.measured_context_tokens),
            error=res_payload.get("error"),
        )

        # Mark task ownership complete
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE federated_task_ownership
                SET status = ?, completed_at = ?
                WHERE task_id = ?
            """, (result.status.value, time.time(), contract.task_id))
            conn.commit()

        # Persist standard result in worker_executions table
        self.persist_worker_result(result)
        return result

    def execute_with_rerouting(
        self,
        primary_worker_id: str,
        contract: WorkerContract,
        package: WorkerPackage,
        fallback_worker_id: str,
        inject_crash_on_primary: bool = False,
    ) -> Tuple[WorkerResult, bool]:
        """
        Attempts execution on primary worker. If primary crashes or fails,
        reroutes to fallback worker and completes task cleanly.
        Returns (result, was_rerouted).
        """
        res_primary = self.execute_in_isolated_process(
            primary_worker_id,
            contract,
            package,
            inject_crash=inject_crash_on_primary,
        )

        if res_primary.status == WorkerExecutionStatus.SUCCESS:
            return res_primary, False

        # Primary failed or crashed -> record reroute audit and dispatch fallback
        self.record_audit("FEDERATED_REROUTE", contract.task_id, primary_worker_id, {
            "fallback_worker_id": fallback_worker_id,
            "primary_error": res_primary.error,
        })
        contract.worker_id = fallback_worker_id
        res_fallback = self.execute_in_isolated_process(
            fallback_worker_id,
            contract,
            package,
        )
        return res_fallback, True

    def execute_concurrent_tasks(
        self,
        task_specs: List[Tuple[str, WorkerContract, WorkerPackage]],
    ) -> List[WorkerResult]:
        """
        Executes multiple independent worker tasks concurrently using independent OS processes.
        """
        results: List[WorkerResult] = []
        with ThreadPoolExecutor(max_workers=len(task_specs)) as executor:
            future_to_task = {
                executor.submit(self.execute_in_isolated_process, worker_id, contract, package): contract.task_id
                for worker_id, contract, package in task_specs
            }
            for future in as_completed(future_to_task):
                res = future.result()
                results.append(res)
        return results
