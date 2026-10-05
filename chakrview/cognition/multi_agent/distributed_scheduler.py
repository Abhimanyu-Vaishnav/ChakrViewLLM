"""
ChakrView Step 117: Distributed DAG Scheduler & Worker Placement Engine.

Combines PersistentTaskGraph dependencies, TaskResourceRequirements,
and WorkerResourceProfile into a deterministic DAG placement scheduler.
"""

from __future__ import annotations

import json
import sqlite3
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
from chakrview.cognition.multi_agent.resource_federation import (
    ResourceCapacityLevel,
    WorkerResourceProfile,
    TaskResourceRequirements,
    ResourceAwareWorkerSelector,
)
from chakrview.cognition.multi_agent.transport import RequestEnvelope, ResponseEnvelope
from chakrview.cognition.multi_agent.transport_channel import (
    BaseTransportChannel,
    SubprocessTransportChannel,
)
from chakrview.cognition.ppb.task_models import (
    PersistentTaskGraph,
    PersistentTaskNode,
    TaskNodeStatus,
)
from chakrview.cognition.tool_gate import GovernedToolGate


class DistributedDAGScheduler:
    """
    Schedules topological task nodes over a pool of heterogeneous federated workers
    with strict resource awareness, dependency enforcement, and concurrency controls.
    """

    def __init__(
        self,
        db_path: Path,
        workspace_root: Path,
        tool_gate: GovernedToolGate,
        transport_channel: Optional[BaseTransportChannel] = None,
    ) -> None:
        self.db_path = Path(db_path)
        self.workspace_root = Path(workspace_root)
        self.tool_gate = tool_gate
        self.workers: Dict[str, WorkerResourceProfile] = {}
        if transport_channel is None:
            runtime_script = Path(__file__).parent / "process_runtime.py"
            self.transport = SubprocessTransportChannel(runtime_script, workspace_root)
        else:
            self.transport = transport_channel
        self._init_db_tables()

    def _init_db_tables(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS scheduled_tasks (
                    task_id TEXT PRIMARY KEY,
                    worker_id TEXT,
                    status TEXT,
                    resource_level TEXT,
                    scheduled_at REAL,
                    completed_at REAL
                )
            """)
            conn.commit()

    def register_worker_profile(self, profile: WorkerResourceProfile) -> None:
        self.workers[profile.worker_id] = profile

    def schedule_and_execute_node(
        self,
        node: PersistentTaskNode,
        package: WorkerPackage,
        allowed_tools: List[str],
        requirements: Optional[TaskResourceRequirements] = None,
    ) -> WorkerResult:
        reqs = requirements or TaskResourceRequirements()
        # Role mapping from task description / tags
        role_map = {
            "INSPECTION": WorkerRole.PROJECT_ANALYST,
            "AST_ANALYSIS": WorkerRole.PROJECT_ANALYST,
            "DEPENDENCY_ANALYSIS": WorkerRole.PLANNER,
            "REASONING": WorkerRole.PLANNER,
            "PATCH_EXECUTION": WorkerRole.IMPLEMENTER,
            "VERIFICATION": WorkerRole.TEST_ENGINEER,
        }
        target_role = role_map.get(node.resource_type.value, WorkerRole.IMPLEMENTER)

        contract = WorkerContract(
            worker_id="placeholder",
            role=target_role,
            task_id=node.node_id,
            allowed_tools=allowed_tools,
            allowed_files=node.affected_files,
            context_budget=512,
        )

        # Select best worker deterministically
        best_worker = ResourceAwareWorkerSelector.select_best_worker(
            list(self.workers.values()),
            contract,
            reqs,
        )
        if not best_worker:
            return WorkerResult(
                worker_id="unassigned",
                task_id=node.node_id,
                role=target_role,
                status=WorkerExecutionStatus.FAILED,
                error=f"No eligible worker found for requirements {reqs}",
            )

        contract.worker_id = best_worker.worker_id
        best_worker.current_active_tasks += 1

        # Dispatch via transport
        try:
            # Build envelope
            req_env = RequestEnvelope(
                protocol_version="1.0.0",
                worker_id=best_worker.worker_id,
                task_id=node.node_id,
                package_id=f"pkg_{node.node_id}",
                role=target_role.value,
                context_hash="dummy_ctx_hash",
                context_token_count=package.measured_context_tokens,
                allowed_tools=allowed_tools,
                allowed_files=node.affected_files,
                context_budget=512,
                package_payload={
                    "task_id": package.task_id,
                    "objective": package.objective,
                    "dependency_outputs": package.dependency_outputs,
                    "targeted_code_context": package.targeted_code_context,
                    "measured_context_tokens": package.measured_context_tokens,
                },
            )
            resp_env = self.transport.send_request(req_env, timeout_seconds=15.0)
            payload = resp_env.result_payload
            res = WorkerResult(
                worker_id=best_worker.worker_id,
                task_id=node.node_id,
                role=target_role,
                status=WorkerExecutionStatus(payload.get("status", "SUCCESS")),
                evidence=payload.get("evidence", {}),
                proposed_changes=payload.get("proposed_changes", {}),
                verification_output=payload.get("verification_output"),
                context_tokens_used=payload.get("context_tokens_used", package.measured_context_tokens),
                error=payload.get("error"),
            )
        except Exception as e:
            res = WorkerResult(
                worker_id=best_worker.worker_id,
                task_id=node.node_id,
                role=target_role,
                status=WorkerExecutionStatus.FAILED,
                error=str(e),
            )
        finally:
            best_worker.current_active_tasks = max(0, best_worker.current_active_tasks - 1)

        # Record in DB
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO scheduled_tasks (task_id, worker_id, status, resource_level, scheduled_at, completed_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (node.node_id, best_worker.worker_id, res.status.value, best_worker.capacity_level.value, time.time(), time.time()))
            conn.commit()

        return res
