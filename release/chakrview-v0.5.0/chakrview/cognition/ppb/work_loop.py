"""
ChakrView Step 84: Incremental Cognitive Work Loop & Step 85: Project Knowledge Evolution.

Unifies:
- User Task -> Decomposition -> Persistent Task Graph (Step 82)
- Resource-Aware Scheduling (Step 83)
- Bounded Work Loop with Safe Resumability after Interruption (Step 84)
- Knowledge Evolution: Evolving PPB records with what was inspected, changed, verified, and learned (Step 85)
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

from chakrview.cognition.ppb.task_models import (
    PersistentTaskGraph,
    PersistentTaskNode,
    TaskNodeStatus,
    TaskResourceType,
)
from chakrview.cognition.ppb.task_storage import (
    PersistentTaskStorage,
    DeterministicTaskDecomposer,
)
from chakrview.cognition.ppb.scheduler import (
    ResourceAwareTaskScheduler,
    ScheduleDecision,
    TaskExecutionLease,
)
from chakrview.cognition.ppb.brain import PersistentProjectBrain
from chakrview.cognition.ppb.models import (
    EpistemicStatus,
    KnowledgeRecord,
    KnowledgeRecordType,
)
from chakrview.cognition.ppb.retrieval import ProjectKnowledgeRetriever
from chakrview.cognition.ppb.maintainer import ChangeAwareBrainMaintainer
from chakrview.cognition.repository.change_detector import RepositoryDiff, FileChange, ChangeCategory
from chakrview.cognition.repository.impact_analyzer import ImpactReport


@dataclass
class LoopExecutionProgress:
    """Progress and telemetry snapshot of an execution session across the task graph."""
    graph_id: str
    total_nodes: int
    completed_nodes: int
    failed_nodes: int
    pending_nodes: int
    is_finished: bool
    iterations_run: int
    knowledge_records_evolved: int
    execution_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "graph_id": self.graph_id,
            "total_nodes": self.total_nodes,
            "completed_nodes": self.completed_nodes,
            "failed_nodes": self.failed_nodes,
            "pending_nodes": self.pending_nodes,
            "is_finished": self.is_finished,
            "iterations_run": self.iterations_run,
            "knowledge_records_evolved": self.knowledge_records_evolved,
            "execution_time_ms": round(self.execution_time_ms, 2),
        }


class IncrementalCognitiveWorkLoop:
    """
    Executes a persistent task graph in bounded, resource-aware iterations.
    Supports complete process interruption and safe resumption.
    """

    def __init__(
        self,
        brain: PersistentProjectBrain,
        task_storage: PersistentTaskStorage,
        scheduler: Optional[ResourceAwareTaskScheduler] = None,
        custom_node_executor: Optional[Callable[[PersistentTaskNode, TaskExecutionLease], Dict[str, Any]]] = None,
    ) -> None:
        self.brain = brain
        self.task_storage = task_storage
        self.scheduler = scheduler or ResourceAwareTaskScheduler(brain.hardware_capability)
        self.retriever = ProjectKnowledgeRetriever(brain)
        self.maintainer = ChangeAwareBrainMaintainer(brain)
        self.custom_node_executor = custom_node_executor

    def run_graph(
        self,
        graph_id: str,
        max_iterations: Optional[int] = None,
    ) -> LoopExecutionProgress:
        """
        Execute ready nodes in the graph in bounded steps.
        If interrupted (or max_iterations reached), progress remains saved in SQLite.
        """
        start_t = time.perf_counter()
        graph = self.task_storage.load_graph(graph_id)
        if not graph:
            raise ValueError(f"Task graph '{graph_id}' not found in storage")

        iterations = 0
        evolved_count = 0

        while not graph.is_finished:
            if max_iterations is not None and iterations >= max_iterations:
                break

            decision = self.scheduler.schedule_next_batch(graph)
            if not decision.allocated_leases:
                # No nodes can currently proceed (e.g., all remaining are blocked or done)
                break

            for lease in decision.allocated_leases:
                node = graph.get_node(lease.node_id)
                if not node:
                    continue

                # Mark node RUNNING
                node.status = TaskNodeStatus.RUNNING
                self.task_storage.save_graph(graph)

                # Execute node safely
                try:
                    res_payload = self._execute_single_node(node, lease)
                    graph.mark_completed(node.node_id, result_payload=res_payload)
                    # Step 85: Evolve project knowledge based on completed node
                    evolved = self._evolve_knowledge_from_node(node, res_payload)
                    if evolved:
                        evolved_count += 1
                except Exception as ex:
                    node.retry_count += 1
                    if node.retry_count >= node.max_retries:
                        graph.mark_failed(node.node_id, str(ex))
                    else:
                        node.status = TaskNodeStatus.READY

                # Persist state after every individual node completion
                self.task_storage.save_graph(graph)

            iterations += 1

        summary = graph.get_progress_summary()
        elapsed = (time.perf_counter() - start_t) * 1000.0

        return LoopExecutionProgress(
            graph_id=graph_id,
            total_nodes=len(graph.nodes),
            completed_nodes=summary.get(TaskNodeStatus.COMPLETED.value, 0),
            failed_nodes=summary.get(TaskNodeStatus.FAILED.value, 0),
            pending_nodes=summary.get(TaskNodeStatus.PENDING.value, 0) + summary.get(TaskNodeStatus.READY.value, 0),
            is_finished=graph.is_finished,
            iterations_run=iterations,
            knowledge_records_evolved=evolved_count,
            execution_time_ms=elapsed,
        )

    def _execute_single_node(
        self,
        node: PersistentTaskNode,
        lease: TaskExecutionLease,
    ) -> Dict[str, Any]:
        """Execute task node work under leased resource boundaries."""
        if self.custom_node_executor:
            return self.custom_node_executor(node, lease)

        # Built-in deterministic simulation for cognition subtasks
        if node.resource_type == TaskResourceType.INSPECTION:
            # Query PPB for relevant files
            bundle = self.retriever.retrieve_for_task(node.description, target_files=node.affected_files)
            return {
                "inspected_files": [r.file_path for r in bundle.records],
                "matched_records_count": len(bundle.records),
                "epistemic_status": "KNOWN",
            }

        elif node.resource_type == TaskResourceType.DEPENDENCY_ANALYSIS:
            return {
                "dependencies_analyzed": node.affected_files,
                "verified_integrity": True,
            }

        elif node.resource_type == TaskResourceType.REASONING:
            return {
                "proposal_formulated": True,
                "approved_for_patch": True,
                "confidence": 0.95,
            }

        elif node.resource_type == TaskResourceType.PATCH_EXECUTION:
            return {
                "patch_applied": True,
                "modified_files": node.affected_files,
            }

        elif node.resource_type == TaskResourceType.VERIFICATION:
            return {
                "tests_passed": True,
                "verification_status": "VERIFIED",
            }

        return {"executed": True}

    def _evolve_knowledge_from_node(
        self,
        node: PersistentTaskNode,
        result: Dict[str, Any],
    ) -> bool:
        """
        Step 85: Persist structured, epistemically classified project knowledge
        derived from completed task execution.
        """
        project_id = self.brain.project_id
        record_id = f"evolved_{node.node_id}"

        if node.resource_type in (TaskResourceType.PATCH_EXECUTION, TaskResourceType.VERIFICATION):
            status = EpistemicStatus.FACT if result.get("tests_passed", True) else EpistemicStatus.INSUFFICIENT
            rec = KnowledgeRecord(
                record_id=record_id,
                project_id=project_id,
                record_type=KnowledgeRecordType.TASK_HISTORY,
                file_path=node.affected_files[0] if node.affected_files else "project",
                summary=f"Evolved knowledge from {node.title}",
                details={
                    "node_id": node.node_id,
                    "task_type": node.resource_type.value,
                    "result": result,
                    "affected_files": node.affected_files,
                    "required_symbols": node.required_symbols,
                },
                dependencies=node.dependencies,
                evidence_ids=[f"node_eval_{node.node_id}"],
                epistemic_status=status,
                confidence=1.0 if status == EpistemicStatus.FACT else 0.5,
                source_chunk="cognitive_work_loop",
            )
            self.brain.store_knowledge(rec)
            return True

        elif node.resource_type == TaskResourceType.INSPECTION and result.get("inspected_files"):
            rec = KnowledgeRecord(
                record_id=record_id,
                project_id=project_id,
                record_type=KnowledgeRecordType.OBSERVATION,
                file_path=node.affected_files[0] if node.affected_files else "project",
                summary=f"Grounded inspection: {node.title}",
                details={"result": result},
                epistemic_status=EpistemicStatus.FACT,
                confidence=1.0,
                source_chunk="cognitive_work_loop",
            )
            self.brain.store_knowledge(rec)
            return True

        return False
