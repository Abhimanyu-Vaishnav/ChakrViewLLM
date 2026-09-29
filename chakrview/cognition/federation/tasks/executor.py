"""
Worker-Side Task Execution Engine Bound to Sovereign CapabilityGate (Step 38).

CRITICAL AXIOMS:
- NO ARBITRARY REMOTE CODE EXECUTION:
  Workers execute only pre-registered capabilities via CapabilityGate.
- LOCAL_AUTHORITY > PEER_AUTHORITY:
  Execution must be authorized under an active ResourceExecutionGrant.
- RESOURCE QUOTA ENFORCEMENT:
  CPU and RAM usage are bounded and tracked per grant.
- ISOLATION & FAIL-CLOSED:
  Any execution failure, timeout, or unauthorized call fails cleanly without crashing the node.
- ΔW = 0:
  Neural model weights are strictly immutable.
"""

import logging
import threading
import time
from typing import Dict, List, Optional, Any, Callable

from chakrview.capability.gate import (
    CapabilityGate,
    CapabilityAuthorizationError,
    CapabilityArgumentValidationError,
)
from chakrview.capability.contract import (
    CapabilityRequest,
    CapabilityResult,
    CapabilityContext,
)
from chakrview.capability.registry import CapabilityRegistry
from chakrview.cognition.federation.tasks.models import (
    WorkUnit,
    WorkUnitState,
    TaskCheckpoint,
    TaskResultEnvelope,
    ResourceRequirements,
    ResourceExecutionGrant,
)
from chakrview.cognition.federation.tasks.grant import (
    ExecutionGrantManager,
)
from chakrview.cognition.federation.tasks.errors import (
    UnauthorizedTaskExecutionError,
    ExecutionGrantViolationError,
    WorkerExecutionError,
    TaskCancelledError,
)

logger = logging.getLogger(__name__)


class FederationTaskExecutor:
    """
    Sovereign worker execution engine. Executes dispatched WorkUnits on the local node
    under CapabilityGate mediation and ResourceExecutionGrant policies.
    """

    def __init__(
        self,
        local_node_id: str,
        grant_manager: ExecutionGrantManager,
        capability_gate: Optional[CapabilityGate] = None,
        checkpoint_callback: Optional[Callable[[TaskCheckpoint], None]] = None,
    ) -> None:
        self.local_node_id = local_node_id
        self.grant_manager = grant_manager
        self.capability_gate = capability_gate or CapabilityGate()
        self.checkpoint_callback = checkpoint_callback
        self._lock = threading.RLock()
        self._cancelled_tasks: set = set()
        self._active_executions: Dict[str, WorkUnit] = {}

    def cancel_task(self, task_id: str) -> None:
        """Mark a task as cancelled locally to halt ongoing or queued units."""
        with self._lock:
            self._cancelled_tasks.add(task_id)

    def is_cancelled(self, task_id: str) -> bool:
        """Check if a task was cancelled."""
        with self._lock:
            return task_id in self._cancelled_tasks

    def execute_unit(
        self,
        unit: WorkUnit,
        originating_node_id: str,
    ) -> TaskResultEnvelope:
        """
        Execute an assigned WorkUnit on this node.
        
        Args:
            unit: The WorkUnit to execute.
            originating_node_id: Node ID of the coordinating node.
            
        Returns:
            A validated TaskResultEnvelope.
            
        Raises:
            UnauthorizedTaskExecutionError if grant check fails.
            TaskCancelledError if task was cancelled before or during execution.
            WorkerExecutionError if capability execution fails.
        """
        if self.is_cancelled(unit.task_id):
            raise TaskCancelledError(f"Task {unit.task_id} has been cancelled locally")

        # 1. Sovereign Authorization Check via ResourceExecutionGrant
        grant = self.grant_manager.authorize_execution(
            peer_node_id=originating_node_id,
            tenant_id=unit.requirements.tenant_id,
            capability_id=unit.capability_id,
            requirements=unit.requirements,
        )

        # 2. Reserve resources
        self.grant_manager.allocate_resources(grant.grant_id, unit.requirements)
        start_time = time.time()

        with self._lock:
            self._active_executions[unit.unit_id] = unit

        try:
            # 3. Emit Initial Checkpoint (Progress 0.0)
            checkpoint_0 = TaskCheckpoint(
                task_id=unit.task_id,
                unit_id=unit.unit_id,
                checkpoint_id=f"cp_{unit.unit_id}_start",
                sequence=1,
                state=WorkUnitState.RUNNING,
                progress=0.0,
                partial_result=None,
                worker_id=self.local_node_id,
                attempt=unit.attempt,
            )
            if self.checkpoint_callback:
                self.checkpoint_callback(checkpoint_0)

            # Check cancellation
            if self.is_cancelled(unit.task_id):
                raise TaskCancelledError(f"Task {unit.task_id} cancelled during startup")

            # 4. Capability Execution through CapabilityGate
            cap_request = CapabilityRequest(
                capability_id=unit.capability_id,
                parameters=unit.input_payload,
                caller_id=originating_node_id,
                timeout_seconds=unit.requirements.timeout_seconds,
            )

            # Authorize against CapabilityGate
            context = CapabilityContext(
                user_id=originating_node_id,
                session_id=unit.task_id,
                task_id=unit.task_id,
            )
            is_cap_auth = self.capability_gate.authorize(cap_request, context=context)
            if not is_cap_auth:
                raise UnauthorizedTaskExecutionError(
                    f"CapabilityGate denied execution of capability '{unit.capability_id}' "
                    f"for caller '{originating_node_id}'"
                )

            # 5. Emit Midpoint Checkpoint (Progress 0.5)
            checkpoint_1 = TaskCheckpoint(
                task_id=unit.task_id,
                unit_id=unit.unit_id,
                checkpoint_id=f"cp_{unit.unit_id}_mid",
                sequence=2,
                state=WorkUnitState.RUNNING,
                progress=0.5,
                partial_result={"status": "executing"},
                worker_id=self.local_node_id,
                attempt=unit.attempt,
            )
            if self.checkpoint_callback:
                self.checkpoint_callback(checkpoint_1)

            # Check cancellation before invocation
            # 0. Check worker lease deadline if present
            if unit.lease is not None and unit.lease.is_expired():
                raise WorkerExecutionError(f"Worker lease {unit.lease.lease_id} expired prior to execution")

            if self.is_cancelled(unit.task_id):
                raise TaskCancelledError(f"Task {unit.task_id} cancelled before capability invocation")

            # Execute capability
            cap_result = self.capability_gate.execute(cap_request, context=context)

            # 6. Build Result Envelope
            duration_ms = (time.time() - start_time) * 1000.0
            output_val = getattr(cap_result, "output", getattr(cap_result, "data", None))
            error_val = getattr(cap_result, "error", getattr(cap_result, "error_message", None))

            if cap_result.success:
                result_envelope = TaskResultEnvelope(
                    task_id=unit.task_id,
                    unit_id=unit.unit_id,
                    attempt=unit.attempt,
                    worker_id=self.local_node_id,
                    status="SUCCESS",
                    result_data=output_val,
                    execution_time_ms=duration_ms,
                    error_message=None,
                    fencing_token=unit.fencing_token,
                )
            else:
                result_envelope = TaskResultEnvelope(
                    task_id=unit.task_id,
                    unit_id=unit.unit_id,
                    attempt=unit.attempt,
                    worker_id=self.local_node_id,
                    status="FAILED",
                    result_data=None,
                    execution_time_ms=duration_ms,
                    error_message=error_val or "Capability execution reported failure",
                    fencing_token=unit.fencing_token,
                )

            return result_envelope

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000.0
            if isinstance(e, TaskCancelledError):
                raise
            logger.warning("Task execution error on unit %s: %s", unit.unit_id, str(e))
            return TaskResultEnvelope(
                task_id=unit.task_id,
                unit_id=unit.unit_id,
                attempt=unit.attempt,
                worker_id=self.local_node_id,
                status="FAILED",
                result_data=None,
                execution_time_ms=duration_ms,
                error_message=str(e),
                fencing_token=unit.fencing_token,
            )
        finally:
            # Always release allocated resources
            self.grant_manager.release_resources(grant.grant_id, unit.requirements)
            with self._lock:
                self._active_executions.pop(unit.unit_id, None)
