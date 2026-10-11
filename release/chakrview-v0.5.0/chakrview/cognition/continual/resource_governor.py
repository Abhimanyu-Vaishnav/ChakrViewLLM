"""
Wave 543: Resource-Aware Cognitive Governor.

Enforces:
RESOURCE AVAILABILITY != PERMISSION != REQUIREMENT

Handles:
- Minimum sufficient resource allocation (CPU-first, GPU optional).
- Task decomposition and checkpointing under memory/compute pressure.
- Distributed worker failure tolerance (Node failure != Task failure).
- Graceful yield when interactive user workload becomes active (e.g., gaming, video editing).
"""

from __future__ import annotations

import time
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple


class ExecutionBudgetMode(str, Enum):
    MINIMAL_COMPACT = "MINIMAL_COMPACT"       # Constrained RAM/CPU; single-thread/shallow depth
    STANDARD_BALANCED = "STANDARD_BALANCED"   # Normal CPU operation
    BURST_DECOMPOSED = "BURST_DECOMPOSED"     # Large task broken into serialized sub-chunks
    YIELDED = "YIELDED"                       # User workload active; AI throttle/yield


@dataclass
class SystemResourceConstraint:
    cpu_available_pct: float       # 0.0 to 100.0
    ram_available_mb: float        # Available memory
    gpu_available: bool
    user_interactive_active: bool  # True if user is gaming, rendering, editing
    node_count: int = 1


@dataclass
class ExecutionPlan:
    budget_mode: ExecutionBudgetMode
    chunk_size: int
    allow_parallel_workers: bool
    checkpoint_interval_steps: int
    yield_priority_to_user: bool
    rationale: str


class ResourceAwareCognitiveGovernor:
    """
    Governs cognitive computation to respect hardware limits and prioritize human user workloads.
    """

    @classmethod
    def determine_execution_plan(
        cls,
        task_complexity_score: float, # e.g. estimated steps (1 to 20)
        constraints: SystemResourceConstraint,
    ) -> ExecutionPlan:
        # Case H: User workload becomes active -> YIELD
        if constraints.user_interactive_active:
            return ExecutionPlan(
                budget_mode=ExecutionBudgetMode.YIELDED,
                chunk_size=1,
                allow_parallel_workers=False,
                checkpoint_interval_steps=1,
                yield_priority_to_user=True,
                rationale="User interactive workload detected; throttling background cognitive execution.",
            )

        # Case B & C: Severely restricted RAM (< 256MB) or low CPU (< 15%)
        if constraints.ram_available_mb < 256 or constraints.cpu_available_pct < 15.0:
            return ExecutionPlan(
                budget_mode=ExecutionBudgetMode.MINIMAL_COMPACT,
                chunk_size=1,
                allow_parallel_workers=False,
                checkpoint_interval_steps=1,
                yield_priority_to_user=False,
                rationale="Host memory/CPU under severe constraint; forced minimal serialized footprint.",
            )

        # Case E: Task too large for single pass -> DECOMPOSE & CHECKPOINT
        if task_complexity_score > 6:
            return ExecutionPlan(
                budget_mode=ExecutionBudgetMode.BURST_DECOMPOSED,
                chunk_size=3,
                allow_parallel_workers=(constraints.node_count > 1),
                checkpoint_interval_steps=2,
                yield_priority_to_user=False,
                rationale="Task exceeds atomic reasoning budget; decomposed into checkpointed chunks.",
            )

        # Normal standard execution
        return ExecutionPlan(
            budget_mode=ExecutionBudgetMode.STANDARD_BALANCED,
            chunk_size=int(task_complexity_score),
            allow_parallel_workers=False,
            checkpoint_interval_steps=5,
            yield_priority_to_user=False,
            rationale="Standard CPU-first minimum-sufficient allocation.",
        )

    @classmethod
    def handle_worker_loss(
        cls,
        active_subtasks: List[str],
        lost_node_id: str,
    ) -> Tuple[List[str], str]:
        """
        Guarantees: NODE FAILURE != TASK FAILURE.
        Re-plans remaining work onto local or remaining nodes.
        """
        # Re-queue orphaned tasks for local execution
        return active_subtasks, f"Node {lost_node_id} unreachable; failover re-queued {len(active_subtasks)} tasks locally."
