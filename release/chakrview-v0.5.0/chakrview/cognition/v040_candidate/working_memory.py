"""
Wave 581: Working Memory 2.0.

Architectural Guarantees:
- Bounded, task-scoped working memory.
- Does NOT pollute permanent storage or durable experience memory.
- Manages:
  - Active goals & subgoals
  - Intermediate reasoning variables
  - Unresolved subproblems
  - Relevant gathered evidence references
  - Task constraints (timeouts, max hops, memory limits)
- Clean task-boundary reset & checkpoint/restore serialization.
"""

from __future__ import annotations

import time
import json
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Set


@dataclass
class TaskConstraint:
    max_steps: int = 10
    max_time_seconds: float = 30.0
    allowed_domains: List[str] = field(default_factory=list)
    resource_limit_bytes: int = 50_000_000  # 50MB default working limit


@dataclass
class ActiveTaskState:
    task_id: str
    primary_goal: str
    active_subgoals: List[str] = field(default_factory=list)
    intermediate_variables: Dict[str, Any] = field(default_factory=dict)
    unresolved_questions: List[str] = field(default_factory=list)
    verified_evidence_refs: List[str] = field(default_factory=list)
    pending_actions: List[str] = field(default_factory=list)
    recent_results: List[Dict[str, Any]] = field(default_factory=list)
    constraints: TaskConstraint = field(default_factory=TaskConstraint)
    created_at: float = field(default_factory=time.time)
    step_count: int = 0


class WorkingMemory2:
    """
    Task-scoped working memory with bounded lifecycle and zero durable memory pollution.
    """

    def __init__(self, max_concurrent_tasks: int = 4):
        self.max_concurrent_tasks = max_concurrent_tasks
        self.active_tasks: Dict[str, ActiveTaskState] = {}

    def initialize_task(
        self,
        task_id: str,
        goal: str,
        constraints: Optional[TaskConstraint] = None,
    ) -> ActiveTaskState:
        if len(self.active_tasks) >= self.max_concurrent_tasks:
            # Evict oldest completed or stale task
            oldest_id = min(self.active_tasks.keys(), key=lambda k: self.active_tasks[k].created_at)
            del self.active_tasks[oldest_id]

        state = ActiveTaskState(
            task_id=task_id,
            primary_goal=goal,
            constraints=constraints or TaskConstraint(),
        )
        self.active_tasks[task_id] = state
        return state

    def set_variable(self, task_id: str, key: str, value: Any) -> None:
        if task_id in self.active_tasks:
            self.active_tasks[task_id].intermediate_variables[key] = value

    def get_variable(self, task_id: str, key: str, default: Any = None) -> Any:
        if task_id in self.active_tasks:
            return self.active_tasks[task_id].intermediate_variables.get(key, default)
        return default

    def push_result(self, task_id: str, action_name: str, result_data: Any) -> None:
        if task_id in self.active_tasks:
            st = self.active_tasks[task_id]
            st.step_count += 1
            st.recent_results.append({
                "step": st.step_count,
                "action": action_name,
                "result": result_data,
                "timestamp": time.time(),
            })
            # Bound history size
            if len(st.recent_results) > 20:
                st.recent_results = st.recent_results[-20:]

    def reset_task(self, task_id: str) -> None:
        """Explicit boundary reset: clears all ephemeral task state."""
        if task_id in self.active_tasks:
            del self.active_tasks[task_id]

    def checkpoint_state(self, task_id: str) -> str:
        """Serializes working state for checkpointing without durable memory side-effects."""
        if task_id not in self.active_tasks:
            raise KeyError(f"Task {task_id} not found in WorkingMemory2.")
        return json.dumps(asdict(self.active_tasks[task_id]))

    def restore_checkpoint(self, checkpoint_json: str) -> ActiveTaskState:
        data = json.loads(checkpoint_json)
        constraints = TaskConstraint(**data.pop("constraints"))
        state = ActiveTaskState(constraints=constraints, **data)
        self.active_tasks[state.task_id] = state
        return state
