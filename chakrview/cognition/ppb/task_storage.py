"""
ChakrView Step 82: Persistent Task Storage & Deterministic Task Decomposition.

Provides:
- PersistentTaskStorage: SQLite persistence layer for PersistentTaskGraph and PersistentTaskNode.
- DeterministicTaskDecomposer: Decomposes a high-level user request into a structured PersistentTaskGraph
  derived from project knowledge, symbols, and dependency topology without hallucinations.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from chakrview.cognition.ppb.task_models import (
    PersistentTaskGraph,
    PersistentTaskNode,
    TaskNodeStatus,
    TaskResourceType,
)
from chakrview.cognition.ppb.models import KnowledgeRecordType, EpistemicStatus
from chakrview.cognition.ppb.brain import PersistentProjectBrain
from chakrview.cognition.ppb.retrieval import ProjectKnowledgeRetriever, PPBRetrievalBudget


class PersistentTaskStorage:
    """
    Durable, thread-safe SQLite persistence for decomposed task graphs.
    Survives process kills, restarts, and sessions.
    """

    def __init__(self, db_path: str) -> None:
        self.db_path = Path(db_path).resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._init_tables()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _init_tables(self) -> None:
        with self._lock:
            with self._get_connection() as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS task_graphs (
                        graph_id TEXT PRIMARY KEY,
                        project_id TEXT NOT NULL,
                        root_task_description TEXT NOT NULL,
                        created_at_utc TEXT NOT NULL,
                        updated_at_utc TEXT NOT NULL
                    );
                """)

                conn.execute("""
                    CREATE TABLE IF NOT EXISTS task_nodes (
                        node_id TEXT PRIMARY KEY,
                        graph_id TEXT NOT NULL,
                        title TEXT NOT NULL,
                        description TEXT NOT NULL,
                        resource_type TEXT NOT NULL,
                        parent_id TEXT,
                        dependencies_json TEXT NOT NULL,
                        affected_files_json TEXT NOT NULL,
                        required_symbols_json TEXT NOT NULL,
                        priority INTEGER NOT NULL,
                        status TEXT NOT NULL,
                        retry_count INTEGER NOT NULL DEFAULT 0,
                        max_retries INTEGER NOT NULL DEFAULT 2,
                        completion_evidence_ids_json TEXT NOT NULL,
                        result_payload_json TEXT NOT NULL,
                        failure_reason TEXT,
                        created_at_utc TEXT NOT NULL,
                        updated_at_utc TEXT NOT NULL,
                        FOREIGN KEY (graph_id) REFERENCES task_graphs(graph_id) ON DELETE CASCADE
                    );
                """)

                conn.execute("CREATE INDEX IF NOT EXISTS idx_tn_graph ON task_nodes(graph_id);")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_tn_status ON task_nodes(status);")

    def save_graph(self, graph: PersistentTaskGraph) -> None:
        """Atomically persist or update an entire task graph."""
        now_str = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        with self._lock:
            with self._get_connection() as conn:
                conn.execute("""
                    INSERT INTO task_graphs (graph_id, project_id, root_task_description, created_at_utc, updated_at_utc)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(graph_id) DO UPDATE SET
                        updated_at_utc = excluded.updated_at_utc;
                """, (graph.graph_id, graph.project_id, graph.root_task_description, now_str, now_str))

                for node in graph.nodes.values():
                    conn.execute("""
                        INSERT INTO task_nodes (
                            node_id, graph_id, title, description, resource_type,
                            parent_id, dependencies_json, affected_files_json, required_symbols_json,
                            priority, status, retry_count, max_retries, completion_evidence_ids_json,
                            result_payload_json, failure_reason, created_at_utc, updated_at_utc
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(node_id) DO UPDATE SET
                            status = excluded.status,
                            retry_count = excluded.retry_count,
                            completion_evidence_ids_json = excluded.completion_evidence_ids_json,
                            result_payload_json = excluded.result_payload_json,
                            failure_reason = excluded.failure_reason,
                            updated_at_utc = excluded.updated_at_utc;
                    """, (
                        node.node_id,
                        node.graph_id,
                        node.title,
                        node.description,
                        node.resource_type.value,
                        node.parent_id,
                        json.dumps(node.dependencies, sort_keys=True),
                        json.dumps(node.affected_files, sort_keys=True),
                        json.dumps(node.required_symbols, sort_keys=True),
                        node.priority,
                        node.status.value,
                        node.retry_count,
                        node.max_retries,
                        json.dumps(node.completion_evidence_ids, sort_keys=True),
                        json.dumps(node.result_payload, sort_keys=True),
                        node.failure_reason,
                        node.created_at_utc or now_str,
                        now_str,
                    ))

    def load_graph(self, graph_id: str) -> Optional[PersistentTaskGraph]:
        """Reconstruct a PersistentTaskGraph from persistent SQLite storage."""
        with self._lock:
            with self._get_connection() as conn:
                cur = conn.execute("SELECT * FROM task_graphs WHERE graph_id = ? LIMIT 1;", (graph_id,))
                grow = cur.fetchone()
                if not grow:
                    return None

                graph = PersistentTaskGraph(
                    graph_id=grow["graph_id"],
                    project_id=grow["project_id"],
                    root_task_description=grow["root_task_description"],
                )

                cur_nodes = conn.execute(
                    "SELECT * FROM task_nodes WHERE graph_id = ? ORDER BY priority ASC, node_id ASC;",
                    (graph_id,),
                )
                for nrow in cur_nodes.fetchall():
                    node = PersistentTaskNode(
                        node_id=nrow["node_id"],
                        graph_id=nrow["graph_id"],
                        title=nrow["title"],
                        description=nrow["description"],
                        resource_type=TaskResourceType(nrow["resource_type"]),
                        parent_id=nrow["parent_id"],
                        dependencies=json.loads(nrow["dependencies_json"]),
                        affected_files=json.loads(nrow["affected_files_json"]),
                        required_symbols=json.loads(nrow["required_symbols_json"]),
                        priority=nrow["priority"],
                        status=TaskNodeStatus(nrow["status"]),
                        retry_count=nrow["retry_count"],
                        max_retries=nrow["max_retries"],
                        completion_evidence_ids=json.loads(nrow["completion_evidence_ids_json"]),
                        result_payload=json.loads(nrow["result_payload_json"]),
                        failure_reason=nrow["failure_reason"],
                        created_at_utc=nrow["created_at_utc"],
                        updated_at_utc=nrow["updated_at_utc"],
                    )
                    graph.nodes[node.node_id] = node

                return graph

    def update_node_status(
        self,
        node_id: str,
        status: TaskNodeStatus,
        result_payload: Optional[Dict[str, Any]] = None,
        failure_reason: Optional[str] = None,
    ) -> bool:
        now_str = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        with self._lock:
            with self._get_connection() as conn:
                params: List[Any] = [status.value, now_str]
                set_clauses = ["status = ?", "updated_at_utc = ?"]

                if result_payload is not None:
                    set_clauses.append("result_payload_json = ?")
                    params.append(json.dumps(result_payload, sort_keys=True))
                if failure_reason is not None:
                    set_clauses.append("failure_reason = ?")
                    params.append(failure_reason)

                params.append(node_id)
                sql = f"UPDATE task_nodes SET {', '.join(set_clauses)} WHERE node_id = ?;"
                cur = conn.execute(sql, tuple(params))
                return cur.rowcount > 0


class DeterministicTaskDecomposer:
    """
    Decomposes a user task into a structured, dependency-ordered PersistentTaskGraph.
    Grounded in PersistentProjectBrain knowledge.
    """

    def __init__(self, brain: PersistentProjectBrain, storage: PersistentTaskStorage) -> None:
        self.brain = brain
        self.storage = storage
        self.retriever = ProjectKnowledgeRetriever(brain)

    def decompose_task(
        self,
        user_task: str,
        graph_id: Optional[str] = None,
        target_files: Optional[Sequence[str]] = None,
        target_symbols: Optional[Sequence[str]] = None,
    ) -> PersistentTaskGraph:
        """
        Produce a durable, multi-step execution graph for a complex goal.
        """
        gid = graph_id or f"graph_{self.brain.project_id}_{hashlib.sha256(user_task.encode('utf-8')).hexdigest()[:12]}"
        graph = PersistentTaskGraph(
            graph_id=gid,
            project_id=self.brain.project_id,
            root_task_description=user_task,
        )

        # 1. Retrieve project context to ground the decomposition
        retrieved = self.retriever.retrieve_for_task(
            task_query=user_task,
            budget=PPBRetrievalBudget(max_records=10),
            target_files=target_files,
            target_symbols=target_symbols,
        )

        primary_files = retrieved.matched_files or (list(target_files) if target_files else ["core"])
        primary_symbols = retrieved.matched_symbols or (list(target_symbols) if target_symbols else [])

        # Subtask 1: Understand Architecture
        node_arch = PersistentTaskNode(
            node_id=f"{gid}_step_1_arch",
            graph_id=gid,
            title="Understand Architecture & Relevant Contracts",
            description=f"Inspect system structure and contracts for '{user_task}'",
            resource_type=TaskResourceType.INSPECTION,
            affected_files=primary_files[:2],
            priority=10,
        )
        graph.add_node(node_arch)

        # Subtask 2: Analyze Dependencies & Grounding
        node_deps = PersistentTaskNode(
            node_id=f"{gid}_step_2_deps",
            graph_id=gid,
            title="Analyze Dependencies & Affected Modules",
            description=f"Map dependency interactions and potential cascade effects",
            resource_type=TaskResourceType.DEPENDENCY_ANALYSIS,
            parent_id=node_arch.node_id,
            dependencies=[node_arch.node_id],
            affected_files=primary_files,
            required_symbols=primary_symbols,
            priority=20,
        )
        graph.add_node(node_deps)

        # Subtask 3: Structured Reasoning & Proposal Design
        node_reason = PersistentTaskNode(
            node_id=f"{gid}_step_3_reason",
            graph_id=gid,
            title="Design Patch & Critical Evaluation",
            description=f"Formulate grounded proposal and critical self-evaluation",
            resource_type=TaskResourceType.REASONING,
            parent_id=node_deps.node_id,
            dependencies=[node_deps.node_id],
            affected_files=primary_files,
            priority=30,
        )
        graph.add_node(node_reason)

        # Subtask 4: Patch Execution
        node_patch = PersistentTaskNode(
            node_id=f"{gid}_step_4_patch",
            graph_id=gid,
            title="Execute Verified Safe Patch",
            description=f"Apply planned mutations under safe execution boundary",
            resource_type=TaskResourceType.PATCH_EXECUTION,
            parent_id=node_reason.node_id,
            dependencies=[node_reason.node_id],
            affected_files=primary_files,
            priority=40,
        )
        graph.add_node(node_patch)

        # Subtask 5: Verification & Brain Evolution
        node_verify = PersistentTaskNode(
            node_id=f"{gid}_step_5_verify",
            graph_id=gid,
            title="Run Verification & Update Persistent Brain",
            description=f"Verify tests, update PPB knowledge records, and evolve state",
            resource_type=TaskResourceType.VERIFICATION,
            parent_id=node_patch.node_id,
            dependencies=[node_patch.node_id],
            affected_files=primary_files,
            priority=50,
        )
        graph.add_node(node_verify)

        # Persist graph to SQLite
        self.storage.save_graph(graph)
        return graph
