"""
ChakrView Step 83: Resource-Aware Task Scheduler.

Maps PersistentTaskNode resource demands against current HardwareCapability / ResourcePolicy:
- LOW_RESOURCE: Strictly sequential execution, minimal worker counts, conservative batch sizes, CPU only.
- STANDARD: Moderate chunk sizes, bounded parallelism for independent read-only inspection tasks.
- ACCELERATED: GPU acceleration enabled when beneficial for inference/embedding, larger working sets.
- DISTRIBUTED_READY: Interfaces prepared for future cluster/worker node leases without fake execution.
- Task Type Differentiation: INSPECTION vs REASONING vs PATCH_EXECUTION (exclusive lock).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from chakrview.runtime.resource import (
    HardwareCapability,
    ResourceDetector,
    ResourcePolicy,
    RuntimeStrategy,
)
from chakrview.cognition.ppb.task_models import (
    PersistentTaskGraph,
    PersistentTaskNode,
    TaskNodeStatus,
    TaskResourceType,
)


@dataclass(frozen=True)
class TaskExecutionLease:
    """
    Bounded resource allocation granted to a task node before execution begins.
    """
    node_id: str
    resource_type: TaskResourceType
    max_memory_mb: int
    allow_parallel: bool
    use_accelerator: bool
    worker_id: str = "local_worker_0"
    is_exclusive_write: bool = False


@dataclass
class ScheduleDecision:
    """Telemetry and scheduling outcome for a batch of ready nodes."""
    strategy: RuntimeStrategy
    eligible_nodes_count: int
    allocated_leases: List[TaskExecutionLease]
    deferred_nodes_count: int
    rationale: str


class ResourceAwareTaskScheduler:
    """
    Determines how and when PersistentTaskNodes execute based on hardware constraints.
    """

    def __init__(self, hardware_capability: Optional[HardwareCapability] = None) -> None:
        self.hardware_capability = hardware_capability or ResourceDetector.detect()
        self.resource_policy = ResourceDetector.determine_strategy(self.hardware_capability)

    def schedule_next_batch(self, graph: PersistentTaskGraph) -> ScheduleDecision:
        """
        Evaluate currently READY nodes in the task graph and grant execution leases.
        """
        ready_nodes = graph.get_ready_nodes()
        if not ready_nodes:
            return ScheduleDecision(
                strategy=self.resource_policy.strategy,
                eligible_nodes_count=0,
                allocated_leases=[],
                deferred_nodes_count=0,
                rationale="No nodes are currently ready for scheduling.",
            )

        strategy = self.resource_policy.strategy
        leases: List[TaskExecutionLease] = []

        if strategy == RuntimeStrategy.LOW_RESOURCE:
            # Strictly sequential: Take only the highest priority ready node
            target = ready_nodes[0]
            is_write = (target.resource_type == TaskResourceType.PATCH_EXECUTION)
            lease = TaskExecutionLease(
                node_id=target.node_id,
                resource_type=target.resource_type,
                max_memory_mb=256,
                allow_parallel=False,
                use_accelerator=False,
                is_exclusive_write=is_write,
            )
            leases.append(lease)
            deferred = len(ready_nodes) - 1
            rationale = "LOW_RESOURCE strategy: Allocated strictly 1 sequential node."

        elif strategy == RuntimeStrategy.STANDARD:
            # Can allocate up to 2 independent read-only nodes concurrently, OR 1 exclusive write node
            first = ready_nodes[0]
            if first.resource_type == TaskResourceType.PATCH_EXECUTION:
                # Exclusive lock
                leases.append(TaskExecutionLease(
                    node_id=first.node_id,
                    resource_type=first.resource_type,
                    max_memory_mb=512,
                    allow_parallel=False,
                    use_accelerator=False,
                    is_exclusive_write=True,
                ))
                deferred = len(ready_nodes) - 1
                rationale = "STANDARD strategy: Exclusive lock granted for PATCH_EXECUTION."
            else:
                # Up to 2 independent read nodes
                for n in ready_nodes[:2]:
                    if n.resource_type != TaskResourceType.PATCH_EXECUTION:
                        leases.append(TaskExecutionLease(
                            node_id=n.node_id,
                            resource_type=n.resource_type,
                            max_memory_mb=512,
                            allow_parallel=True,
                            use_accelerator=False,
                            is_exclusive_write=False,
                        ))
                deferred = len(ready_nodes) - len(leases)
                rationale = f"STANDARD strategy: Allocated {len(leases)} parallel read leases."

        else:  # ACCELERATED or DISTRIBUTED_READY
            # Can allocate up to 4 nodes; enable accelerator for REASONING if GPU available
            first = ready_nodes[0]
            if first.resource_type == TaskResourceType.PATCH_EXECUTION:
                leases.append(TaskExecutionLease(
                    node_id=first.node_id,
                    resource_type=first.resource_type,
                    max_memory_mb=1024,
                    allow_parallel=False,
                    use_accelerator=self.resource_policy.use_accelerator,
                    is_exclusive_write=True,
                ))
                deferred = len(ready_nodes) - 1
                rationale = "ACCELERATED strategy: Exclusive lock granted for PATCH_EXECUTION."
            else:
                for idx, n in enumerate(ready_nodes[:4]):
                    if n.resource_type != TaskResourceType.PATCH_EXECUTION:
                        use_acc = self.resource_policy.use_accelerator and (n.resource_type == TaskResourceType.REASONING)
                        leases.append(TaskExecutionLease(
                            node_id=n.node_id,
                            resource_type=n.resource_type,
                            max_memory_mb=1024,
                            allow_parallel=True,
                            use_accelerator=use_acc,
                            worker_id=f"accelerated_worker_{idx}",
                            is_exclusive_write=False,
                        ))
                deferred = len(ready_nodes) - len(leases)
                rationale = f"ACCELERATED strategy: Allocated {len(leases)} parallel leases with accelerator support."

        return ScheduleDecision(
            strategy=strategy,
            eligible_nodes_count=len(ready_nodes),
            allocated_leases=leases,
            deferred_nodes_count=deferred,
            rationale=rationale,
        )
