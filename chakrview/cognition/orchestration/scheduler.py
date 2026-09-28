"""
Deterministic Task Scheduler for Step 28 Cognitive Orchestration.

Calculates topological dependency execution stages for agent roles within
deliberation rounds, guaranteeing deterministic result ordering.
"""

from typing import Dict, List, Set, Sequence
from collections import defaultdict, deque

from chakrview.cognition.federated.models import AgentRole


class DeterministicTaskScheduler:
    """
    Computes deterministic execution stages based on role dependencies.
    """

    # Canonical fallback role execution priority if no explicit dependency is given
    CANONICAL_ROLE_PRIORITY: Dict[AgentRole, int] = {
        AgentRole.PLANNER: 10,
        AgentRole.RESEARCHER: 20,
        AgentRole.ANALYST: 30,
        AgentRole.CRITIC: 40,
        AgentRole.VERIFIER: 50,
        AgentRole.SYNTHESIZER: 60,
        AgentRole.OBSERVER: 70,
    }

    @classmethod
    def schedule_roles(
        cls,
        roles: Sequence[AgentRole],
        dependencies: Optional[Dict[str, List[str]]] = None,
    ) -> List[List[AgentRole]]:
        """
        Partition roles into sequential execution stages.
        Within each stage, roles can execute concurrently, but stages execute strictly sequentially.
        """
        dependencies = dependencies or {}
        unique_roles = sorted(list(set(roles)), key=lambda r: cls.CANONICAL_ROLE_PRIORITY.get(r, 99))
        role_values = {r.value: r for r in unique_roles}

        # Build in-degree graph
        in_degree: Dict[str, int] = {r.value: 0 for r in unique_roles}
        adj_list: Dict[str, List[str]] = defaultdict(list)

        for dependent, prereqs in dependencies.items():
            if dependent in in_degree:
                for prereq in prereqs:
                    if prereq in in_degree:
                        adj_list[prereq].append(dependent)
                        in_degree[dependent] += 1

        # Kahn's algorithm for staged topological sort
        stages: List[List[AgentRole]] = []
        current_queue = [r_val for r_val, deg in in_degree.items() if deg == 0]
        # Sort for determinism
        current_queue.sort(key=lambda val: cls.CANONICAL_ROLE_PRIORITY.get(role_values[val], 99))

        visited_count = 0
        while current_queue:
            stage_roles = [role_values[val] for val in current_queue]
            stages.append(stage_roles)
            visited_count += len(stage_roles)

            next_queue = []
            for val in current_queue:
                for neighbor in adj_list[val]:
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        next_queue.append(neighbor)

            next_queue.sort(key=lambda v: cls.CANONICAL_ROLE_PRIORITY.get(role_values[v], 99))
            current_queue = next_queue

        # If cycles or orphan dependencies exist, append remaining unvisited deterministically
        if visited_count < len(unique_roles):
            unvisited = [r for r in unique_roles if not any(r in stage for stage in stages)]
            stages.append(unvisited)

        return stages
