"""
ChakrView Step 59: Repository Dependency Graph.

Builds a deterministic directed graph capturing:
- File-to-file import dependencies
- Caller-to-callee dependencies
- Test coverage / target dependencies
Provides upstream root-cause tracing: given a failing test or module C,
find all upstream producer modules (e.g. A -> B -> C).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Set, Tuple

from chakrview.cognition.repository.inspector import ModuleInspection


@dataclass
class DependencyEdge:
    source_module: str  # Consumer / Importer
    target_module: str  # Dependency / Producer
    edge_type: str      # "IMPORT", "CALL", "TESTS"
    symbols: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class RepositoryDependencyGraph:
    """
    Deterministic directed dependency graph of a repository.
    """

    def __init__(self) -> None:
        self.modules: Dict[str, ModuleInspection] = {}
        # Adjacency: module -> list of modules it depends on (outgoing edges)
        self.dependencies: Dict[str, Set[str]] = {}
        # Reverse Adjacency: module -> list of modules that depend on it (incoming edges)
        self.dependents: Dict[str, Set[str]] = {}
        self.edges: List[DependencyEdge] = []

    def build_from_inspections(self, inspections: Dict[str, ModuleInspection]) -> None:
        """Construct graph from AST inspections."""
        self.modules = inspections
        # Map module_name to rel_path
        name_to_path: Dict[str, str] = {insp.module_name: path for path, insp in inspections.items()}

        for path in inspections:
            self.dependencies[path] = set()
            self.dependents[path] = set()

        for path, insp in inspections.items():
            for imp in insp.imports:
                # Check if imported module is internal to repository
                if imp in name_to_path:
                    target_path = name_to_path[imp]
                    self.dependencies[path].add(target_path)
                    self.dependents[target_path].add(path)

                    symbols = [s for s, m in insp.imported_symbols.items() if m == imp]
                    edge_type = "TESTS" if insp.is_test else "IMPORT"
                    self.edges.append(
                        DependencyEdge(
                            source_module=path,
                            target_module=target_path,
                            edge_type=edge_type,
                            symbols=symbols,
                        )
                    )

    def get_upstream_dependencies(self, module_path: str) -> List[str]:
        """
        Return all transitive dependencies of module_path (modules it relies upon)
        in topological / distance order (BFS).
        """
        if module_path not in self.dependencies:
            return []

        visited: Set[str] = set()
        queue: deque[str] = deque([module_path])
        upstream: List[str] = []

        while queue:
            curr = queue.popleft()
            for dep in sorted(list(self.dependencies.get(curr, set()))):
                if dep not in visited and dep != module_path:
                    visited.add(dep)
                    upstream.append(dep)
                    queue.append(dep)

        return upstream

    def get_downstream_dependents(self, module_path: str) -> List[str]:
        """
        Return all transitive dependents of module_path (modules that rely upon it)
        in topological order.
        """
        if module_path not in self.dependents:
            return []

        visited: Set[str] = set()
        queue: deque[str] = deque([module_path])
        downstream: List[str] = []

        while queue:
            curr = queue.popleft()
            for dep in sorted(list(self.dependents.get(curr, set()))):
                if dep not in visited and dep != module_path:
                    visited.add(dep)
                    downstream.append(dep)
                    queue.append(dep)

        return downstream

    def find_dependency_chain(self, start_module: str, end_module: str) -> Optional[List[str]]:
        """Find directed path from start_module to end_module if exists."""
        if start_module == end_module:
            return [start_module]

        queue: deque[List[str]] = deque([[start_module]])
        visited: Set[str] = {start_module}

        while queue:
            path = queue.popleft()
            curr = path[-1]
            if curr == end_module:
                return path

            for neighbor in sorted(list(self.dependencies.get(curr, set()))):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(path + [neighbor])

        return None

    def to_dict(self) -> Dict[str, Any]:
        """Deterministic serialization of the dependency graph."""
        return {
            "modules": {k: v.to_dict() for k, v in self.modules.items()},
            "dependencies": {k: sorted(list(v)) for k, v in self.dependencies.items()},
            "dependents": {k: sorted(list(v)) for k, v in self.dependents.items()},
            "edges": [e.to_dict() for e in self.edges],
        }
