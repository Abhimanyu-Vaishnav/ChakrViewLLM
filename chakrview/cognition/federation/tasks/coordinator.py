"""
Federation Task Coordinator & Fault-Tolerant Orchestration Engine (Steps 38 & 39).

CRITICAL AXIOMS:
- LOCAL_TASK_AUTHORITY > REMOTE_WORKER_STATE
- TASK RESILIENCE: A task must survive the failure or disappearance of any single worker.
- DURABLE CONTINUITY: Checkpoints preserve progress so unfinished work can resume on alternate nodes.
- ATTEMPT FENCING: Stale or partitioned worker writes are rejected immediately.
- AT-LEAST-ONCE + DEDUPLICATION: Duplicate or racing results and duplicate commits are strictly rejected.
- LEASE DEADLINE ENFORCEMENT: Workers only execute under time-bounded leases.
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
    CheckpointManifest,
    CheckpointStatus,
    TaskResultEnvelope,
    SchedulingDecision,
    ResourceRequirements,
    AggregationStrategy,
    DistributedExecutionPlan,
    WorkerLease,
    LeaseState,
    ResumeAction,
    AttemptFenceToken,
)
from chakrview.cognition.federation.tasks.scheduler import (
    DeterministicTaskScheduler,
)
from chakrview.cognition.federation.tasks.checkpoint import (
    TaskCheckpointManager,
    CheckpointStore,
)
from chakrview.cognition.federation.tasks.lease import (
    WorkerLeaseManager,
    AttemptFenceManager,
    DeterministicFailureDetector,
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
    DuplicateCommitError,
    StaleResultError,
    FencedAttemptError,
    LeaseExpiredError,
    TaskCancelledError,
    WorkerUnavailableError,
)

logger = logging.getLogger(__name__)


class FederationTaskCoordinator:
    """
    Coordinates end-to-end distributed task lifecycles, resource scheduling,
    failure recovery, work continuity, lease heartbeats, and result aggregation.
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
        lease_manager: Optional[WorkerLeaseManager] = None,
        fence_manager: Optional[AttemptFenceManager] = None,
        failure_detector: Optional[DeterministicFailureDetector] = None,
    ) -> None:
        self.local_node_id = local_node_id
        self.scheduler = scheduler
        self.checkpoint_manager = checkpoint_manager
        self.checkpoint_store = getattr(checkpoint_manager, "checkpoint_store", CheckpointStore(journal=journal))
        self.local_executor = local_executor
        self.transport_client = transport_client
        self.journal = journal
        self.audit_logger = audit_logger
        self.aggregator = TaskResultAggregator()
        self.validator = TaskResultValidator()

        # Step 39 Continuity & Resilience Managers
        self.failure_detector = failure_detector or DeterministicFailureDetector()
        self.lease_manager = lease_manager or WorkerLeaseManager(failure_detector=self.failure_detector)
        self.fence_manager = fence_manager or AttemptFenceManager()

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
        units_spec: Optional[List[Any]] = None,
        work_units: Optional[List[Any]] = None,
    ) -> List[WorkUnit]:
        """
        Decompose a task into discrete, idempotent, checkpointable WorkUnits.
        Accepts either a list of spec dictionaries (units_spec) or instantiated WorkUnit objects.
        """
        with self._lock:
            task = self._get_task_or_raise(task_id)
            if task.state != TaskState.CREATED:
                raise InvalidTaskDefinitionError(
                    f"Task {task_id} must be in CREATED state to decompose, currently {task.state.value}"
                )

            task.transition_to(TaskState.VALIDATING)
            task.transition_to(TaskState.PLANNING)

            input_list = units_spec if units_spec is not None else (work_units or [])
            built_units: List[WorkUnit] = []

            for i, item in enumerate(input_list):
                if isinstance(item, WorkUnit):
                    unit = item
                    unit.task_id = task_id
                    unit.sequence = i
                    if not unit.requirements:
                        unit.requirements = task.requirements or ResourceRequirements()
                else:
                    spec = dict(item)
                    unit_id = spec.get("unit_id", f"unit_{task_id}_{i}")
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
                built_units.append(unit)

            task.work_units = built_units
            task.transition_to(TaskState.QUEUED)

            self._log_audit(
                AuditEventType.TASK_DECOMPOSED,
                {"task_id": task_id, "unit_count": len(built_units)},
            )
            self._journal_append(
                JournalEntryType.TASK_PLANNED,
                {"task_id": task_id, "unit_count": len(built_units)},
            )
            return built_units

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
        """Schedule a single work unit and dispatch it under lease and fencing token."""
        try:
            decision = self.scheduler.schedule_unit(unit, excluded_nodes=excluded_nodes)
            task.scheduling_decisions.append(decision)

            unit.assigned_node_id = decision.selected_node
            unit.attempt += 1
            worker = decision.selected_node

            # Step 39: Issue monotonic attempt fencing token and worker execution lease
            fence_token = self.fence_manager.issue_fence_token(
                task_id=unit.task_id,
                unit_id=unit.unit_id,
                attempt_number=unit.attempt,
                worker_id=worker,
            )
            unit.fencing_token = fence_token.fencing_token

            lease = self.lease_manager.grant_lease(
                worker_id=worker,
                task_id=unit.task_id,
                unit_id=unit.unit_id,
                attempt_number=unit.attempt,
                fencing_token=unit.fencing_token,
            )
            unit.lease = lease

            unit.transition_to(WorkUnitState.ASSIGNED)

            # Record worker assignment mapping
            if worker not in self._worker_assignments:
                self._worker_assignments[worker] = set()
            self._worker_assignments[worker].add((unit.task_id, unit.unit_id))

            self._log_audit(
                AuditEventType.TASK_ASSIGNED,
                {"task_id": unit.task_id, "unit_id": unit.unit_id, "worker_id": worker, "attempt": unit.attempt},
            )
            self._log_audit(
                AuditEventType.TASK_LEASE_GRANTED,
                {"task_id": unit.task_id, "unit_id": unit.unit_id, "worker_id": worker, "lease_id": lease.lease_id},
            )
            self._journal_append(
                JournalEntryType.TASK_ASSIGNED,
                {"task_id": unit.task_id, "unit_id": unit.unit_id, "worker_id": worker, "attempt": unit.attempt},
            )
            self._journal_append(
                JournalEntryType.TASK_LEASE_GRANTED,
                {"task_id": unit.task_id, "unit_id": unit.unit_id, "worker_id": worker, "lease_id": lease.lease_id},
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
    # Heartbeat, Lease & Checkpoint Ingestion
    # ========================================================================

    def record_heartbeat(
        self,
        task_id: str,
        unit_id: str,
        worker_id: str,
        fencing_token: int,
        current_time: Optional[float] = None,
    ) -> WorkerLease:
        """Process heartbeat from assigned worker and renew execution lease."""
        with self._lock:
            lease = self.lease_manager.renew_lease(
                task_id=task_id,
                unit_id=unit_id,
                worker_id=worker_id,
                fencing_token=fencing_token,
                current_time=current_time,
            )
            self._journal_append(
                JournalEntryType.TASK_LEASE_RENEWED,
                {"task_id": task_id, "unit_id": unit_id, "worker_id": worker_id, "lease_id": lease.lease_id},
            )
            self._log_audit(
                AuditEventType.TASK_LEASE_RENEWED,
                {"task_id": task_id, "unit_id": unit_id, "worker_id": worker_id},
            )
            return lease

    def record_checkpoint(self, checkpoint: TaskCheckpoint) -> None:
        """Ingest a checkpoint emitted by a worker and durably commit it."""
        with self._lock:
            task = self._tasks.get(checkpoint.task_id)
            if not task:
                return

            unit = task.get_unit(checkpoint.unit_id)
            if not unit:
                return

            # Verify integrity & save into checkpoint manager
            self.checkpoint_manager.save_checkpoint(checkpoint)
            unit.latest_checkpoint = checkpoint
            unit.checkpoint_reference = checkpoint.checkpoint_id

            # Also store authoritative committed manifest
            manifest = self.checkpoint_store.get_committed_checkpoint(checkpoint.task_id, checkpoint.unit_id)
            if manifest:
                unit.latest_manifest = manifest

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

    def record_checkpoint_manifest(self, manifest: CheckpointManifest) -> CheckpointManifest:
        """Ingest and commit a full CheckpointManifest."""
        with self._lock:
            created = self.checkpoint_store.create_manifest(manifest)
            committed = self.checkpoint_store.commit_checkpoint(
                manifest.task_id, manifest.work_unit_id, manifest.checkpoint_id
            )

            task = self._tasks.get(manifest.task_id)
            if task:
                unit = task.get_unit(manifest.work_unit_id)
                if unit:
                    unit.latest_manifest = committed
                    unit.checkpoint_reference = committed.checkpoint_id
                    unit.completed_work_range = committed.completed_work_range
                    unit.remaining_work = committed.remaining_work

                    # Sync back to TaskCheckpoint for Step 38 compat
                    progress_val = committed.completed_work_range.get("progress", 0.5)
                    unit.latest_checkpoint = TaskCheckpoint(
                        task_id=manifest.task_id,
                        unit_id=manifest.work_unit_id,
                        checkpoint_id=manifest.checkpoint_id,
                        sequence=manifest.checkpoint_sequence,
                        state=unit.state,
                        progress=progress_val,
                        partial_result=manifest.intermediate_payload,
                        worker_id=manifest.worker_id,
                        attempt=manifest.attempt_id,
                    )
            return committed

    def record_result(self, result: TaskResultEnvelope) -> None:
        """
        Validate and ingest a worker result envelope.
        Enforces attempt fencing, duplicate commit protection, and triggers aggregation.
        """
        with self._lock:
            task = self._get_task_or_raise(result.task_id)
            unit = task.get_unit(result.unit_id)
            if not unit:
                raise FederationTaskError(f"Result refers to unknown unit {result.unit_id} in task {task.task_id}")

            # Validate result (enforces deduplication, staleness, fencing, and integrity)
            self.validator.validate_result(unit, result)

            if result.status == "SUCCESS":
                unit.result = result
                unit.committed_result = result
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

            # Prune active leases for completed task
            self.lease_manager.prune_leases_for_task(task.task_id)

            self._log_audit(AuditEventType.TASK_COMPLETED, {"task_id": task.task_id})
            self._journal_append(JournalEntryType.TASK_COMPLETED, {"task_id": task.task_id})
        except Exception as e:
            logger.error("Result aggregation failed for task %s: %s", task.task_id, str(e))
            task.transition_to(TaskState.FAILED)
            task.error_message = f"Result aggregation failed: {str(e)}"
            self._log_audit(AuditEventType.TASK_FAILED, {"task_id": task.task_id, "error": task.error_message})
            self._journal_append(JournalEntryType.TASK_FAILED, {"task_id": task.task_id, "reason": task.error_message})

    # ========================================================================
    # Failure Recovery & Work Continuity (Step 39)
    # ========================================================================

    def handle_worker_failure(self, failed_worker_node_id: str) -> List[WorkUnit]:
        """
        Handle a worker node disconnect, crash, lease expiry, or quarantine.
        Applies attempt fencing to prevent late writes, recovers from committed checkpoints,
        and reassigns unfinished units to healthy alternate workers.
        """
        reassigned_units: List[WorkUnit] = []
        with self._lock:
            assigned_pairs = list(self._worker_assignments.get(failed_worker_node_id, set()))
            self._worker_assignments.pop(failed_worker_node_id, None)

            # Expire worker leases
            self.lease_manager.expire_worker_leases(failed_worker_node_id)

            self._log_audit(
                AuditEventType.TASK_WORKER_FAILED,
                {"worker_id": failed_worker_node_id, "affected_units": len(assigned_pairs)},
            )
            self._log_audit(
                AuditEventType.TASK_RECOVERY_INITIATED,
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

                # Step 39: Fence previous attempt to reject late writes or racing results
                self.fence_manager.fence_work_unit(task_id, unit_id)
                self._journal_append(
                    JournalEntryType.TASK_ATTEMPT_FENCED,
                    {"task_id": task_id, "unit_id": unit_id, "worker_id": failed_worker_node_id, "attempt": unit.attempt},
                )
                self._log_audit(
                    AuditEventType.TASK_ATTEMPT_FENCED,
                    {"task_id": task_id, "unit_id": unit_id, "worker_id": failed_worker_node_id},
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

                # Step 39 Resume Semantics: Extract latest committed checkpoint
                committed_manifest = self.checkpoint_store.get_committed_checkpoint(task_id, unit_id)
                latest_cp = self.checkpoint_manager.get_latest_checkpoint(task_id, unit_id)

                resume_action = ResumeAction.RESTART_WORK_UNIT
                if committed_manifest is not None and committed_manifest.intermediate_payload is not None:
                    resume_action = ResumeAction.RESUME_FROM_CHECKPOINT
                    unit.latest_manifest = committed_manifest
                    unit.checkpoint_reference = committed_manifest.checkpoint_id
                    unit.input_payload["resume_from_checkpoint"] = committed_manifest.intermediate_payload
                    unit.completed_work_range = committed_manifest.completed_work_range
                    unit.remaining_work = committed_manifest.remaining_work
                elif latest_cp is not None and latest_cp.partial_result is not None:
                    resume_action = ResumeAction.RESUME_FROM_CHECKPOINT
                    unit.latest_checkpoint = latest_cp
                    unit.checkpoint_reference = latest_cp.checkpoint_id
                    unit.input_payload["resume_from_checkpoint"] = latest_cp.partial_result
                else:
                    # Clean restart
                    unit.input_payload.pop("resume_from_checkpoint", None)

                # Reassign to an alternate node (excluding failed worker)
                self._journal_append(
                    JournalEntryType.TASK_WORK_MIGRATED,
                    {
                        "task_id": task_id,
                        "unit_id": unit_id,
                        "previous_worker": failed_worker_node_id,
                        "resume_action": resume_action.value,
                        "checkpoint_id": unit.checkpoint_reference,
                    },
                )
                self._log_audit(
                    AuditEventType.TASK_WORK_MIGRATED,
                    {
                        "task_id": task_id,
                        "unit_id": unit_id,
                        "previous_worker": failed_worker_node_id,
                        "resume_action": resume_action.value,
                    },
                )

                unit.failed_nodes.add(failed_worker_node_id)
                task.transition_to(TaskState.RUNNING)
                self._schedule_and_dispatch_unit(task, unit, excluded_nodes=set(unit.failed_nodes))
                reassigned_units.append(unit)

            self._log_audit(
                AuditEventType.TASK_RECOVERY_COMPLETED,
                {"worker_id": failed_worker_node_id, "reassigned_units": len(reassigned_units)},
            )
            self._journal_append(
                JournalEntryType.TASK_RECOVERY_COMPLETED,
                {"worker_id": failed_worker_node_id, "reassigned_units": len(reassigned_units)},
            )

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
                self.fence_manager.fence_work_unit(task_id, unit_id)

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
                    self.fence_manager.fence_work_unit(task_id, unit.unit_id)
                    unit.transition_to(WorkUnitState.ABANDONED)

            self.lease_manager.prune_leases_for_task(task_id)

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
