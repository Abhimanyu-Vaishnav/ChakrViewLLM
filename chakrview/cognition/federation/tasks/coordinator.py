"""
Federation Task Coordinator & Fault-Tolerant Orchestration Engine (Step 38).

CRITICAL AXIOMS:
- LOCAL_TASK_AUTHORITY > REMOTE_WORKER_STATE
- TASK RESILIENCE: A task must survive the failure or disappearance of any single worker.
- DURABLE CONTINUITY: Checkpoints preserve progress so unfinished work can resume on alternate nodes.
- AT-LEAST-ONCE + DEDUPLICATION: Duplicate or racing results are strictly rejected.
- JOURNAL INTEGRITY: State mutations are recorded in the Step 34 security journal.
- ΔW = 0: Zero neural weight mutation.
"""

import logging
import threading
import time
import uuid
from typing import Dict, List, Optional, Any, Set, Tuple, Callable

from chakrview.cognition.peering.models import (
    AuditEventType,
)
from chakrview.cognition.federation.persistence.models import (
    JournalEntryType,
)
from chakrview.cognition.federation.tasks.models import (
    DistributedTask,
    WorkUnit,
    TaskState,
    WorkUnitState,
    TaskCheckpoint,
    TaskResultEnvelope,
    SchedulingDecision,
    ResourceRequirements,
    AggregationStrategy,
    DistributedExecutionPlan,
)
from chakrview.cognition.federation.tasks.scheduler import (
    DeterministicTaskScheduler,
)
from chakrview.cognition.federation.tasks.checkpoint import (
    TaskCheckpointManager,
)
from chakrview.cognition.federation.tasks.validator import (
    TaskResultValidator,
)
from chakrview.cognition.federation.tasks.aggregator import (
    TaskResultAggregator,
)
from chakrview.cognition.federation.tasks.executor import (
    FederationTaskExecutor,
)
from chakrview.cognition.federation.tasks.errors import (
    FederationTaskError,
    InvalidTaskDefinitionError,
    NoEligibleWorkerError,
    DuplicateResultError,
    StaleResultError,
    TaskCancelledError,
    WorkerUnavailableError,
)

logger = logging.getLogger(__name__)


class FederationTaskCoordinator:
    """
    Coordinates end-to-end distributed task lifecycles, resource scheduling,
    failure recovery, work continuity, and result aggregation across federation peers.
    """

    def __init__(
        self,
        local_node_id: str,
        scheduler: DeterministicTaskScheduler,
        checkpoint_manager: TaskCheckpointManager,
        local_executor: Optional[FederationTaskExecutor] = None,
        transport_client: Optional[Any] = None,
        journal: Optional[Any] = None,
        audit_logger: Optional[Any] = None,
    ) -> None:
        self.local_node_id = local_node_id
        self.scheduler = scheduler
        self.checkpoint_manager = checkpoint_manager
        self.local_executor = local_executor
        self.transport_client = transport_client
        self.journal = journal
        self.audit_logger = audit_logger
        self.aggregator = TaskResultAggregator()
        self.validator = TaskResultValidator()

        self._lock = threading.RLock()
        self._tasks: Dict[str, DistributedTask] = {}
        # Worker assignment reverse mapping: worker_node_id -> Set[(task_id, unit_id)]
        self._worker_assignments: Dict[str, Set[Tuple[str, str]]] = {}

    # ========================================================================
    # Task Creation & Decomposition
    # ========================================================================

    def create_task(
        self,
        name: str,
        tenant_id: str = "default",
        description: str = "",
        priority: int = 5,
        timeout_seconds: float = 300.0,
        aggregation_strategy: AggregationStrategy = AggregationStrategy.CONCATENATE,
        requirements: Optional[ResourceRequirements] = None,
    ) -> DistributedTask:
        """Create and register a new DistributedTask."""
        with self._lock:
            task_id = f"task_{uuid.uuid4().hex[:12]}"
            task = DistributedTask(
                task_id=task_id,
                name=name,
                tenant_id=tenant_id,
                description=description,
                created_by_node_id=self.local_node_id,
                priority=priority,
                deadline=time.time() + timeout_seconds,
                state=TaskState.CREATED,
                requirements=requirements,
                aggregation_strategy=aggregation_strategy,
            )
            self._tasks[task_id] = task

            self._log_audit(AuditEventType.TASK_CREATED, {"task_id": task_id, "name": name})
            self._journal_append(JournalEntryType.TASK_CREATED, {"task_id": task_id, "tenant_id": tenant_id})
            return task

    def decompose_task(
        self,
        task_id: str,
        units_spec: List[Dict[str, Any]],
    ) -> List[WorkUnit]:
        """
        Decompose a task into discrete WorkUnits.
        Each spec item should contain: capability_id, input_payload, and optional requirements.
        """
        with self._lock:
            task = self._get_task_or_raise(task_id)
            if task.state != TaskState.CREATED:
                raise InvalidTaskDefinitionError(
                    f"Task {task_id} must be in CREATED state to decompose, currently {task.state.value}"
                )

            task.transition_to(TaskState.VALIDATING)
            task.transition_to(TaskState.PLANNING)

            work_units: List[WorkUnit] = []
            for i, spec in enumerate(units_spec):
                unit_id = f"unit_{task_id}_{i}"
                cap_id = spec["capability_id"]
                payload = spec.get("input_payload", {})
                req = spec.get("requirements")
                if req is None:
                    req = task.requirements or ResourceRequirements(
                        capability_id=cap_id,
                        tenant_id=task.tenant_id,
                    )
                elif isinstance(req, dict):
                    req = ResourceRequirements.from_dict(req)

                unit = WorkUnit(
                    unit_id=unit_id,
                    task_id=task_id,
                    sequence=i,
                    capability_id=cap_id,
                    input_payload=payload,
                    requirements=req,
                    state=WorkUnitState.PENDING,
                )
                work_units.append(unit)

            task.work_units = work_units
            task.transition_to(TaskState.QUEUED)

            self._log_audit(
                AuditEventType.TASK_DECOMPOSED,
                {"task_id": task_id, "unit_count": len(work_units)},
            )
            self._journal_append(
                JournalEntryType.TASK_PLANNED,
                {"task_id": task_id, "unit_count": len(work_units)},
            )
            return work_units

    # ========================================================================
    # Scheduling & Dispatch
    # ========================================================================

    def schedule_and_dispatch_task(self, task_id: str) -> None:
        """
        Schedule all pending work units and dispatch them to eligible worker nodes.
        """
        with self._lock:
            task = self._get_task_or_raise(task_id)
            if task.state not in (TaskState.QUEUED, TaskState.RUNNING, TaskState.PARTIALLY_COMPLETED, TaskState.RECOVERING):
                raise InvalidTaskDefinitionError(
                    f"Task {task_id} cannot be dispatched in state {task.state.value}"
                )

            if task.state == TaskState.QUEUED:
                task.transition_to(TaskState.RUNNING)

            for unit in task.work_units:
                if unit.state == WorkUnitState.PENDING:
                    self._schedule_and_dispatch_unit(task, unit)

    def _schedule_and_dispatch_unit(
        self,
        task: DistributedTask,
        unit: WorkUnit,
        excluded_nodes: Optional[Set[str]] = None,
    ) -> None:
        """Schedule a single work unit and dispatch it."""
        try:
            decision = self.scheduler.schedule_unit(unit, excluded_nodes=excluded_nodes)
            task.scheduling_decisions.append(decision)

            unit.assigned_node_id = decision.selected_node
            unit.attempt += 1
            unit.transition_to(WorkUnitState.ASSIGNED)

            # Record worker assignment mapping
            worker = decision.selected_node
            if worker not in self._worker_assignments:
                self._worker_assignments[worker] = set()
            self._worker_assignments[worker].add((unit.task_id, unit.unit_id))

            self._log_audit(
                AuditEventType.TASK_ASSIGNED,
                {"task_id": unit.task_id, "unit_id": unit.unit_id, "worker_id": worker},
            )
            self._journal_append(
                JournalEntryType.TASK_ASSIGNED,
                {"task_id": unit.task_id, "unit_id": unit.unit_id, "worker_id": worker, "attempt": unit.attempt},
            )

            # Dispatch execution
            self._execute_assigned_unit(task, unit)

        except NoEligibleWorkerError as e:
            logger.warning("Scheduling failed for unit %s: %s", unit.unit_id, str(e))
            unit.transition_to(WorkUnitState.FAILED)
            task.transition_to(TaskState.FAILED)
            task.error_message = f"No eligible worker for unit {unit.unit_id}: {str(e)}"
            self._log_audit(AuditEventType.TASK_FAILED, {"task_id": task.task_id, "error": task.error_message})
            self._journal_append(JournalEntryType.TASK_FAILED, {"task_id": task.task_id, "reason": task.error_message})

    def _execute_assigned_unit(self, task: DistributedTask, unit: WorkUnit) -> None:
        """Execute unit locally if assigned to self, or dispatch over transport channel."""
        worker = unit.assigned_node_id

        if worker == self.local_node_id and self.local_executor:
            # Local Execution
            unit.transition_to(WorkUnitState.RUNNING)
            try:
                result = self.local_executor.execute_unit(unit, originating_node_id=self.local_node_id)
                self.record_result(result)
            except Exception as e:
                logger.warning("Local execution failed on unit %s: %s", unit.unit_id, str(e))
                self.handle_unit_failure(unit.task_id, unit.unit_id, str(e))
        else:
            # Remote Execution over transport client
            if self.transport_client is not None:
                try:
                    # Mark unit as running once successfully handed to transport
                    unit.transition_to(WorkUnitState.RUNNING)
                    self.transport_client.send_task_assignment(worker, unit)
                except Exception as e:
                    logger.warning("Transport dispatch failed to worker %s: %s", worker, str(e))
                    self.handle_worker_failure(worker)
            else:
                # In mock/test environments without transport client, mark as running
                unit.transition_to(WorkUnitState.RUNNING)

    # ========================================================================
    # Checkpointing & Result Ingestion
    # ========================================================================

    def record_checkpoint(self, checkpoint: TaskCheckpoint) -> None:
        """Ingest a checkpoint emitted by a worker."""
        with self._lock:
            task = self._tasks.get(checkpoint.task_id)
            if not task:
                return

            unit = task.get_unit(checkpoint.unit_id)
            if not unit:
                return

            # Verify integrity & save
            self.checkpoint_manager.save_checkpoint(checkpoint)
            unit.latest_checkpoint = checkpoint
            unit.checkpoint_reference = checkpoint.checkpoint_id

            if unit.state == WorkUnitState.RUNNING:
                unit.transition_to(WorkUnitState.CHECKPOINTED)
                unit.transition_to(WorkUnitState.RUNNING)

            self._log_audit(
                AuditEventType.TASK_CHECKPOINTED,
                {
                    "task_id": checkpoint.task_id,
                    "unit_id": checkpoint.unit_id,
                    "progress": checkpoint.progress,
                    "worker_id": checkpoint.worker_id,
                },
            )

    def record_result(self, result: TaskResultEnvelope) -> None:
        """
        Validate and ingest a worker result envelope.
        Deduplicates racing results and triggers aggregation when all units complete.
        """
        with self._lock:
            task = self._get_task_or_raise(result.task_id)
            unit = task.get_unit(result.unit_id)
            if not unit:
                raise FederationTaskError(f"Result refers to unknown unit {result.unit_id} in task {task.task_id}")

            # Validate result (enforces deduplication, staleness, and integrity)
            self.validator.validate_result(unit, result)

            if result.status == "SUCCESS":
                unit.result = result
                unit.result_reference = result.result_digest
                unit.transition_to(WorkUnitState.COMPLETED)
                self.scheduler.release_assignment(result.worker_id)

                self._log_audit(
                    AuditEventType.TASK_RESULT_VALIDATED,
                    {"task_id": task.task_id, "unit_id": unit.unit_id, "worker_id": result.worker_id},
                )
                self._journal_append(
                    JournalEntryType.TASK_RESULT_ACCEPTED,
                    {"task_id": task.task_id, "unit_id": unit.unit_id, "worker_id": result.worker_id},
                )

                # Check if all units have completed
                if task.are_all_units_completed():
                    self._complete_task(task)
                else:
                    if task.state == TaskState.RUNNING:
                        task.transition_to(TaskState.PARTIALLY_COMPLETED)
                        task.transition_to(TaskState.RUNNING)
            else:
                logger.warning("Unit %s failed on worker %s: %s", unit.unit_id, result.worker_id, result.error_message)
                self.handle_unit_failure(task.task_id, unit.unit_id, result.error_message or "Worker failure")

    def _complete_task(self, task: DistributedTask) -> None:
        """Aggregate results and transition task to COMPLETED."""
        try:
            final_res = self.aggregator.aggregate_results(task)
            task.final_result = final_res
            task.transition_to(TaskState.COMPLETED)

            self._log_audit(AuditEventType.TASK_COMPLETED, {"task_id": task.task_id})
            self._journal_append(JournalEntryType.TASK_COMPLETED, {"task_id": task.task_id})
        except Exception as e:
            logger.error("Result aggregation failed for task %s: %s", task.task_id, str(e))
            task.transition_to(TaskState.FAILED)
            task.error_message = f"Result aggregation failed: {str(e)}"
            self._log_audit(AuditEventType.TASK_FAILED, {"task_id": task.task_id, "error": task.error_message})
            self._journal_append(JournalEntryType.TASK_FAILED, {"task_id": task.task_id, "reason": task.error_message})

    # ========================================================================
    # Failure Recovery & Work Continuity
    # ========================================================================

    def handle_worker_failure(self, failed_worker_node_id: str) -> List[WorkUnit]:
        """
        Handle a worker node disconnect, crash, or quarantine.
        Reassigns all unfinished work units from the failed worker to alternate nodes.
        Preserves completed units and resumes from checkpoints where available.
        """
        reassigned_units: List[WorkUnit] = []
        with self._lock:
            assigned_pairs = list(self._worker_assignments.get(failed_worker_node_id, set()))
            self._worker_assignments.pop(failed_worker_node_id, None)

            self._log_audit(
                AuditEventType.TASK_WORKER_FAILED,
                {"worker_id": failed_worker_node_id, "affected_units": len(assigned_pairs)},
            )

            for task_id, unit_id in assigned_pairs:
                task = self._tasks.get(task_id)
                if not task:
                    continue

                unit = task.get_unit(unit_id)
                if not unit or unit.state in (WorkUnitState.COMPLETED, WorkUnitState.ABANDONED):
                    continue

                logger.info(
                    "Recovering unit %s of task %s from failed worker %s",
                    unit_id, task_id, failed_worker_node_id,
                )

                # Release scheduler load for failed worker
                self.scheduler.release_assignment(failed_worker_node_id)

                # Transition task to RECOVERING if running
                if task.state in (TaskState.RUNNING, TaskState.PARTIALLY_COMPLETED):
                    task.transition_to(TaskState.RECOVERING)

                # Transition unit to RETRYABLE then PENDING
                if unit.state != WorkUnitState.RETRYABLE:
                    unit.transition_to(WorkUnitState.RETRYABLE)
                unit.transition_to(WorkUnitState.PENDING)

                # Extract latest durable checkpoint if available to preserve progress
                latest_cp = self.checkpoint_manager.get_latest_checkpoint(task_id, unit_id)
                if latest_cp:
                    unit.latest_checkpoint = latest_cp
                    unit.checkpoint_reference = latest_cp.checkpoint_id
                    # Merge partial results into payload for next worker
                    if latest_cp.partial_result:
                        unit.input_payload["resume_from_checkpoint"] = latest_cp.partial_result

                # Reassign to an alternate node (excluding failed worker)
                self._journal_append(
                    JournalEntryType.TASK_REASSIGNED,
                    {
                        "task_id": task_id,
                        "unit_id": unit_id,
                        "previous_worker": failed_worker_node_id,
                        "resumed_checkpoint": latest_cp.checkpoint_id if latest_cp else None,
                    },
                )

                task.transition_to(TaskState.RUNNING)
                self._schedule_and_dispatch_unit(task, unit, excluded_nodes={failed_worker_node_id})
                reassigned_units.append(unit)

        return reassigned_units

    def handle_unit_failure(self, task_id: str, unit_id: str, error_message: str) -> None:
        """Handle execution failure of a specific work unit with retry limit enforcement."""
        with self._lock:
            task = self._get_task_or_raise(task_id)
            unit = task.get_unit(unit_id)
            if not unit:
                return

            if unit.assigned_node_id:
                self.scheduler.release_assignment(unit.assigned_node_id)

            if unit.attempt < unit.requirements.max_retries:
                # Retry on alternate worker
                failed_worker = unit.assigned_node_id
                unit.transition_to(WorkUnitState.RETRYABLE)
                unit.transition_to(WorkUnitState.PENDING)
                excluded = {failed_worker} if failed_worker else set()
                self._schedule_and_dispatch_unit(task, unit, excluded_nodes=excluded)
            else:
                # Max retries exceeded; fail unit and task
                unit.transition_to(WorkUnitState.FAILED)
                task.transition_to(TaskState.FAILED)
                task.error_message = f"Unit {unit_id} exceeded max retries ({unit.requirements.max_retries}): {error_message}"
                self._log_audit(AuditEventType.TASK_FAILED, {"task_id": task_id, "error": task.error_message})
                self._journal_append(JournalEntryType.TASK_FAILED, {"task_id": task_id, "reason": task.error_message})

    def cancel_task(self, task_id: str) -> None:
        """Cancel an ongoing or queued task and release all worker reservations."""
        with self._lock:
            task = self._get_task_or_raise(task_id)
            if task.state in (TaskState.COMPLETED, TaskState.FAILED, TaskState.CANCELLED, TaskState.EXPIRED):
                return

            task.transition_to(TaskState.CANCELLED)

            # Mark all unfinished units as abandoned
            for unit in task.work_units:
                if unit.state not in (WorkUnitState.COMPLETED, WorkUnitState.FAILED, WorkUnitState.ABANDONED):
                    if unit.assigned_node_id:
                        self.scheduler.release_assignment(unit.assigned_node_id)
                    unit.transition_to(WorkUnitState.ABANDONED)

            if self.local_executor:
                self.local_executor.cancel_task(task_id)

            self._log_audit(AuditEventType.TASK_CANCELLED, {"task_id": task_id})
            self._journal_append(JournalEntryType.TASK_CANCELLED, {"task_id": task_id})

    # ========================================================================
    # Task Retrieval & Recovery
    # ========================================================================

    def get_task(self, task_id: str) -> Optional[DistributedTask]:
        """Look up task by ID."""
        with self._lock:
            return self._tasks.get(task_id)

    def list_tasks(self, tenant_id: Optional[str] = None) -> List[DistributedTask]:
        """List tasks, optionally filtered by tenant."""
        with self._lock:
            if tenant_id is None or tenant_id == "*":
                return list(self._tasks.values())
            return [t for t in self._tasks.values() if t.tenant_id == tenant_id]

    def _get_task_or_raise(self, task_id: str) -> DistributedTask:
        task = self._tasks.get(task_id)
        if not task:
            raise FederationTaskError(f"Task '{task_id}' not found")
        return task

    # ========================================================================
    # Audit & WAL Integration
    # ========================================================================

    def _log_audit(self, event_type: AuditEventType, details: Dict[str, Any]) -> None:
        """Emit an audit event if audit_logger is configured."""
        if self.audit_logger and hasattr(self.audit_logger, "log_event"):
            try:
                self.audit_logger.log_event(event_type=event_type, details=details)
            except Exception:
                pass

    def _journal_append(self, entry_type: JournalEntryType, payload: Dict[str, Any]) -> None:
        """Append to durable WAL journal if configured."""
        if self.journal and hasattr(self.journal, "append_entry"):
            try:
                self.journal.append_entry(entry_type=entry_type, payload=payload)
            except Exception:
                pass
