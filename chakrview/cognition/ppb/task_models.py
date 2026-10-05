"""
ChakrView Step 82: Persistent Task Decomposition Models & Lifecycle.

Defines:
- TaskNodeStatus: Lifecycle states (PENDING, READY, RUNNING, COMPLETED, FAILED, BLOCKED, SKIPPED).
- TaskResourceType: Categorization of execution resource demands (INSPECTION, AST_ANALYSIS, DEPENDENCY_ANALYSIS, REASONING, PATCH_EXECUTION, VERIFICATION).
- PersistentTaskNode: Unit of decomposed work with deterministic ID, parent links, dependencies, resource profile, and completion records.
- PersistentTaskGraph: Directed Acyclic Graph (DAG) of task nodes with cycle validation, topological ordering, and progress metrics.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple


class TaskNodeStatus(str, Enum):
    """Lifecycle states for individual task nodes in the persistent task graph."""
    PENDING = "PENDING"       # Waiting for prerequisite dependencies to complete
    READY = "READY"           # All dependencies satisfied, eligible for scheduling
    RUNNING = "RUNNING"       # Actively executing under an allocated resource lease
    COMPLETED = "COMPLETED"   # Execution and local verification succeeded
    FAILED = "FAILED"         # Execution or verification failed
    BLOCKED = "BLOCKED"       # Prerequisite dependency failed or blocked
    SKIPPED = "SKIPPED"       # Intentionally bypassed or unnecessary


class TaskResourceType(str, Enum):
    """Resource demand classification for task nodes."""
    INSPECTION = "INSPECTION"                   # Tiny file / read-only inspection (lowest footprint)
    AST_ANALYSIS = "AST_ANALYSIS"               # AST parse and symbol analysis (moderate CPU)
    DEPENDENCY_ANALYSIS = "DEPENDENCY_ANALYSIS" # Graph / topology analysis (moderate memory)
    REASONING = "REASONING"                     # Structured reasoning & proposal generation (high context)
    PATCH_EXECUTION = "PATCH_EXECUTION"         # Atomic file modification (exclusive write lease)
    VERIFICATION = "VERIFICATION"               # Test execution / multi-level verification (isolated runner)


@dataclass
class PersistentTaskNode:
    """
    A single node in the persistent task decomposition graph.
    All IDs and fingerprints are deterministic and independent of volatile timestamps.
    """
    node_id: str
    graph_id: str
    title: str
    description: str
    resource_type: TaskResourceType
    parent_id: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)  # IDs of prerequisite nodes
    affected_files: List[str] = field(default_factory=list)
    required_symbols: List[str] = field(default_factory=list)
    priority: int = 100  # Lower number = higher priority
    status: TaskNodeStatus = TaskNodeStatus.PENDING
    retry_count: int = 0
    max_retries: int = 2
    completion_evidence_ids: List[str] = field(default_factory=list)
    result_payload: Dict[str, Any] = field(default_factory=dict)
    failure_reason: Optional[str] = None
    created_at_utc: str = ""
    updated_at_utc: str = ""

    def compute_fingerprint(self) -> str:
        """Deterministic fingerprint of this node's task specification."""
        hasher = hashlib.sha256()
        hasher.update(self.graph_id.encode("utf-8"))
        hasher.update((self.parent_id or "").encode("utf-8"))
        hasher.update(self.title.encode("utf-8"))
        hasher.update(self.resource_type.value.encode("utf-8"))
        hasher.update(",".join(sorted(self.dependencies)).encode("utf-8"))
        hasher.update(",".join(sorted(self.affected_files)).encode("utf-8"))
        return hasher.hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "graph_id": self.graph_id,
            "title": self.title,
            "description": self.description,
            "resource_type": self.resource_type.value,
            "parent_id": self.parent_id,
            "dependencies": list(self.dependencies),
            "affected_files": list(self.affected_files),
            "required_symbols": list(self.required_symbols),
            "priority": self.priority,
            "status": self.status.value,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "completion_evidence_ids": list(self.completion_evidence_ids),
            "result_payload": self.result_payload,
            "failure_reason": self.failure_reason,
            "created_at_utc": self.created_at_utc,
            "updated_at_utc": self.updated_at_utc,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PersistentTaskNode:
        d = dict(data)
        d["resource_type"] = TaskResourceType(d["resource_type"])
        d["status"] = TaskNodeStatus(d["status"])
        return cls(**d)


class PersistentTaskGraph:
    """
    DAG of PersistentTaskNodes representing an end-to-end user task decomposition.
    Guarantees deterministic dependency order and detects cycles.
    """

    def __init__(self, graph_id: str, project_id: str, root_task_description: str) -> None:
        self.graph_id = graph_id
        self.project_id = project_id
        self.root_task_description = root_task_description
        self.nodes: Dict[str, PersistentTaskNode] = {}

    def add_node(self, node: PersistentTaskNode) -> None:
        if node.graph_id != self.graph_id:
            raise ValueError(f"Node graph_id {node.graph_id} does not match {self.graph_id}")
        self.nodes[node.node_id] = node
        self.validate_no_cycles()

    def get_node(self, node_id: str) -> Optional[PersistentTaskNode]:
        return self.nodes.get(node_id)

    def get_ready_nodes(self) -> List[PersistentTaskNode]:
        """
        Returns all nodes currently eligible for execution:
        - status is PENDING or READY
        - all prerequisite dependencies are in COMPLETED status
        Sorted deterministically by priority, then node_id.
        """
        ready: List[PersistentTaskNode] = []
        for node in self.nodes.values():
            if node.status in (TaskNodeStatus.PENDING, TaskNodeStatus.READY):
                deps_met = True
                for dep_id in node.dependencies:
                    dep = self.nodes.get(dep_id)
                    if not dep or dep.status != TaskNodeStatus.COMPLETED:
                        deps_met = False
                        break
                if deps_met:
                    ready.append(node)

        # Deterministic sorting: priority ascending, then node_id
        ready.sort(key=lambda n: (n.priority, n.node_id))
        return ready

    def mark_completed(
        self,
        node_id: str,
        evidence_ids: Optional[List[str]] = None,
        result_payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        node = self.nodes[node_id]
        node.status = TaskNodeStatus.COMPLETED
        if evidence_ids:
            node.completion_evidence_ids.extend(evidence_ids)
        if result_payload:
            node.result_payload.update(result_payload)

    def mark_failed(self, node_id: str, reason: str) -> None:
        node = self.nodes[node_id]
        node.status = TaskNodeStatus.FAILED
        node.failure_reason = reason
        # Propagate BLOCKED to downstream dependents
        self._propagate_blocked(node_id)

    def _propagate_blocked(self, failed_node_id: str) -> None:
        for node in self.nodes.values():
            if failed_node_id in node.dependencies and node.status in (TaskNodeStatus.PENDING, TaskNodeStatus.READY):
                node.status = TaskNodeStatus.BLOCKED
                node.failure_reason = f"Prerequisite dependency '{failed_node_id}' failed"
                self._propagate_blocked(node.node_id)

    def validate_no_cycles(self) -> None:
        """Kahn's algorithm cycle check across task dependencies."""
        in_degree: Dict[str, int] = {nid: 0 for nid in self.nodes}
        adj: Dict[str, List[str]] = {nid: [] for nid in self.nodes}

        for nid, node in self.nodes.items():
            for dep in node.dependencies:
                if dep in self.nodes:
                    adj[dep].append(nid)
                    in_degree[nid] += 1

        queue = [nid for nid, deg in in_degree.items() if deg == 0]
        visited_count = 0

        while queue:
            curr = queue.pop(0)
            visited_count += 1
            for nxt in adj[curr]:
                in_degree[nxt] -= 1
                if in_degree[nxt] == 0:
                    queue.append(nxt)

        if visited_count < len(self.nodes):
            raise ValueError(f"Cycle detected in task graph {self.graph_id}")

    @property
    def is_finished(self) -> bool:
        """True if all nodes are in terminal states (COMPLETED, FAILED, BLOCKED, SKIPPED)."""
        terminal = {TaskNodeStatus.COMPLETED, TaskNodeStatus.FAILED, TaskNodeStatus.BLOCKED, TaskNodeStatus.SKIPPED}
        return all(n.status in terminal for n in self.nodes.values())

    @property
    def is_all_completed(self) -> bool:
        """True if all nodes completed successfully."""
        return len(self.nodes) > 0 and all(n.status == TaskNodeStatus.COMPLETED for n in self.nodes.values())

    def get_progress_summary(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for n in self.nodes.values():
            st = n.status.value
            counts[st] = counts.get(st, 0) + 1
        return counts

    def to_dict(self) -> Dict[str, Any]:
        return {
            "graph_id": self.graph_id,
            "project_id": self.project_id,
            "root_task_description": self.root_task_description,
            "nodes": {nid: n.to_dict() for nid, n in sorted(self.nodes.items())},
            "progress": self.get_progress_summary(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PersistentTaskGraph:
        graph = cls(
            graph_id=data["graph_id"],
            project_id=data["project_id"],
            root_task_description=data["root_task_description"],
        )
        for nid, ndata in data.get("nodes", {}).items():
            graph.add_node(PersistentTaskNode.from_dict(ndata))
        return graph
