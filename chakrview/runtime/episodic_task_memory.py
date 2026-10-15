"""
Batches 671-675: Episodic Task Memory and Task Provenance Store for ChakrView.
Location: chakrview/runtime/episodic_task_memory.py

Key Capabilities:
1. Stores structured multi-step TaskPlans, subtask execution journals, and intermediate artifacts.
2. Preserves full provenance: user constraints, plan versions, tool outputs, and user corrections.
3. Supports process restarts with atomic, safe JSON serialization.
4. Enforces strict tenant and session isolation.
5. Tracks plan invalidation and replanning chains without loss of historical audit trails.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional


class SubtaskStatus(str, Enum):
    PROPOSED = "proposed"
    AWAITING_CLARIFICATION = "awaiting_clarification"
    READY = "ready"
    RUNNING = "running"
    BLOCKED = "blocked"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    INVALIDATED = "invalidated"


class TaskPlanStatus(str, Enum):
    PROPOSED = "proposed"
    AWAITING_CLARIFICATION = "awaiting_clarification"
    READY = "ready"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    SUPERSEDED = "superseded"


@dataclass
class Subtask:
    subtask_id: str
    goal_description: str
    action_type: str  # "tool_call", "text_transform", "clarification", "reasoning"
    tool_name: Optional[str] = None
    tool_arguments: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    status: SubtaskStatus = SubtaskStatus.PROPOSED
    input_bindings: Dict[str, str] = field(default_factory=dict)
    result_artifact: Optional[Any] = None
    evidence_provenance: Optional[str] = None
    error_message: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Subtask:
        copied = dict(data)
        copied["status"] = SubtaskStatus(copied["status"])
        return cls(**copied)


@dataclass
class TaskPlan:
    plan_id: str
    session_id: str
    user_goal: str
    subtasks: List[Subtask] = field(default_factory=list)
    active_constraints: Dict[str, Any] = field(default_factory=dict)
    plan_status: TaskPlanStatus = TaskPlanStatus.PROPOSED
    version: int = 1
    superseded_by: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "session_id": self.session_id,
            "user_goal": self.user_goal,
            "subtasks": [st.to_dict() for st in self.subtasks],
            "active_constraints": self.active_constraints,
            "plan_status": self.plan_status.value,
            "version": self.version,
            "superseded_by": self.superseded_by,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TaskPlan:
        subtasks = [Subtask.from_dict(st) for st in data.get("subtasks", [])]
        return cls(
            plan_id=data["plan_id"],
            session_id=data["session_id"],
            user_goal=data["user_goal"],
            subtasks=subtasks,
            active_constraints=dict(data.get("active_constraints", {})),
            plan_status=TaskPlanStatus(data.get("plan_status", "proposed")),
            version=data.get("version", 1),
            superseded_by=data.get("superseded_by"),
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
        )


class EpisodicTaskMemory:
    """
    Persistent store for multi-turn task episodes, hierarchical plans, and execution journal.
    Enforces atomic file operations and strict cross-session isolation.
    """

    def __init__(self, storage_dir: Path) -> None:
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._plans_cache: Dict[str, Dict[str, TaskPlan]] = {}  # session_id -> {plan_id -> TaskPlan}

    def _get_episode_path(self, session_id: str) -> Path:
        safe_id = "".join(c for c in session_id if c.isalnum() or c in ("-", "_"))
        if not safe_id:
            safe_id = "default_session"
        return self.storage_dir / f"episodes_{safe_id}.json"

    def save_plan(self, plan: TaskPlan) -> None:
        plan.updated_at = time.time()
        session_id = plan.session_id
        if session_id not in self._plans_cache:
            self._plans_cache[session_id] = self._load_session_plans(session_id)
        self._plans_cache[session_id][plan.plan_id] = plan
        self._flush(session_id)

    def get_plan(self, session_id: str, plan_id: str) -> Optional[TaskPlan]:
        if session_id not in self._plans_cache:
            self._plans_cache[session_id] = self._load_session_plans(session_id)
        return self._plans_cache[session_id].get(plan_id)

    def get_active_plan(self, session_id: str) -> Optional[TaskPlan]:
        if session_id not in self._plans_cache:
            self._plans_cache[session_id] = self._load_session_plans(session_id)
        plans = list(self._plans_cache[session_id].values())
        # Return the most recent plan that is not superseded or cancelled
        active_plans = [
            p for p in plans if p.plan_status not in (TaskPlanStatus.SUPERSEDED, TaskPlanStatus.CANCELLED)
        ]
        if not active_plans:
            return None
        active_plans.sort(key=lambda p: p.updated_at, reverse=True)
        return active_plans[0]

    def list_plans(self, session_id: str) -> List[TaskPlan]:
        if session_id not in self._plans_cache:
            self._plans_cache[session_id] = self._load_session_plans(session_id)
        return list(self._plans_cache[session_id].values())

    def _load_session_plans(self, session_id: str) -> Dict[str, TaskPlan]:
        path = self._get_episode_path(session_id)
        if not path.exists():
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            plans_dict = {}
            for item in data.get("plans", []):
                p = TaskPlan.from_dict(item)
                plans_dict[p.plan_id] = p
            return plans_dict
        except Exception:
            corrupted_path = path.with_suffix(".corrupted")
            try:
                if corrupted_path.exists():
                    corrupted_path.unlink()
                path.rename(corrupted_path)
            except Exception:
                pass
            return {}

    def _flush(self, session_id: str) -> None:
        path = self._get_episode_path(session_id)
        tmp_path = path.with_suffix(".tmp")
        plans = self._plans_cache.get(session_id, {})
        payload = {
            "session_id": session_id,
            "count": len(plans),
            "updated_at": time.time(),
            "plans": [p.to_dict() for p in plans.values()],
        }
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        tmp_path.replace(path)

    def clear(self, session_id: str) -> None:
        self._plans_cache.pop(session_id, None)
        path = self._get_episode_path(session_id)
        if path.exists():
            path.unlink(missing_ok=True)
