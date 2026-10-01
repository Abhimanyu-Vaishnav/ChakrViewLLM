"""
ChakrView Step 61: Repository Dependency Impact Analysis & Semantic Memory Revalidation.

Connects repository changes to dependency graph:
1. Traces directly and transitively affected modules (upstream/downstream).
2. Identifies affected tests that must be executed.
3. Revalidates stored semantic repository memories against live changes:
   - VALID: Repository context untouched or changes purely cosmetic/non-interfering.
   - CONDITIONALLY_VALID: Downstream consumers changed but root pattern & boundaries intact.
   - STALE: Upstream dependencies, root cause signatures, or dependencies mutated.
   - INVALID: Target module removed, language/framework mismatch, or boundary violated.
   - ABSTAIN: Ambiguity or conflicting changes detected.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Tuple

from chakrview.cognition.repository.state import RepositoryState
from chakrview.cognition.repository.change_detector import RepositoryDiff, ChangeCategory
from chakrview.cognition.repository.semantic_record import RepositorySemanticRecord
from chakrview.cognition.repository.memory_index import RepositoryMemoryIndex


class MemoryValidityStatus(Enum):
    VALID = auto()
    CONDITIONALLY_VALID = auto()
    STALE = auto()
    INVALID = auto()
    ABSTAIN = auto()


@dataclass
class MemoryRevalidationDecision:
    """Decision record determining if a previously stored semantic memory is still applicable."""
    memory_id: str
    status: MemoryValidityStatus
    confidence: float
    is_applicable: bool
    rationale: str
    affected_dependencies: List[str] = field(default_factory=list)
    boundary_violations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "status": self.status.name,
            "confidence": round(self.confidence, 4),
            "is_applicable": self.is_applicable,
            "rationale": self.rationale,
            "affected_dependencies": sorted(self.affected_dependencies),
            "boundary_violations": sorted(self.boundary_violations),
        }


@dataclass
class ImpactReport:
    """Complete impact analysis report for a repository diff."""
    directly_modified_modules: List[str] = field(default_factory=list)
    transitive_affected_modules: List[str] = field(default_factory=list)
    affected_tests: List[str] = field(default_factory=list)
    revalidated_memories: Dict[str, MemoryRevalidationDecision] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "directly_modified_modules": sorted(self.directly_modified_modules),
            "transitive_affected_modules": sorted(self.transitive_affected_modules),
            "affected_tests": sorted(self.affected_tests),
            "revalidated_memories": {m: dec.to_dict() for m, dec in sorted(self.revalidated_memories.items())},
        }


class RepositoryImpactAnalyzer:
    """
    Analyzes dependency propagation of changes and revalidates semantic memories.
    """

    @classmethod
    def analyze_impact(
        cls,
        state: RepositoryState,
        diff: RepositoryDiff,
        memory_index: Optional[RepositoryMemoryIndex] = None,
    ) -> ImpactReport:
        """Trace impact of diff through repository dependency graph and revalidate memories."""
        directly_modified = sorted(list(diff.affected_modules))
        transitive_affected: Set[str] = set()
        affected_tests: Set[str] = set()

        dep_graph = state.dependency_graph

        if dep_graph:
            for mod in directly_modified:
                # Downstream modules that rely on this modified module
                downstream = dep_graph.get_downstream_dependents(mod)
                for d in downstream:
                    transitive_affected.add(d)
                    if state.files.get(d) and state.files[d].is_test:
                        affected_tests.add(d)
                # Upstream dependencies
                upstream = dep_graph.get_upstream_dependencies(mod)
                for u in upstream:
                    transitive_affected.add(u)

            # Check all tests directly in modified
            for mod in directly_modified:
                if state.files.get(mod) and state.files[mod].is_test:
                    affected_tests.add(mod)

        # Revalidate memories in memory_index if provided
        revalidated: Dict[str, MemoryRevalidationDecision] = {}
        if memory_index:
            for memory_id, record in memory_index.all_records().items():
                decision = cls.revalidate_memory(state, diff, record, directly_modified, transitive_affected)
                revalidated[memory_id] = decision

        return ImpactReport(
            directly_modified_modules=directly_modified,
            transitive_affected_modules=sorted(list(transitive_affected)),
            affected_tests=sorted(list(affected_tests)),
            revalidated_memories=revalidated,
        )

    @classmethod
    def revalidate_memory(
        cls,
        state: RepositoryState,
        diff: RepositoryDiff,
        memory: RepositorySemanticRecord,
        directly_modified: List[str],
        transitive_affected: Set[str],
    ) -> MemoryRevalidationDecision:
        """
        Evaluate if a specific semantic memory is still valid given current state and diff.
        """
        # 1. Superseded check: if record is not active version, mark invalid
        if not memory.active_version:
            return MemoryRevalidationDecision(
                memory_id=memory.memory_id,
                status=MemoryValidityStatus.INVALID,
                confidence=0.0,
                is_applicable=False,
                rationale=f"Memory version v{memory.version} superseded by {memory.superseded_by}",
            )

        # 2. Language & Framework compatibility check
        if memory.language.lower() != state.language.lower():
            return MemoryRevalidationDecision(
                memory_id=memory.memory_id,
                status=MemoryValidityStatus.INVALID,
                confidence=0.0,
                is_applicable=False,
                rationale=f"Language mismatch: memory requires {memory.language}, repository is {state.language}",
            )

        # 3. Target module presence check
        mem_modules = set(memory.affected_modules)
        repo_files = set(state.files.keys())
        missing_modules = mem_modules - repo_files
        if missing_modules:
            return MemoryRevalidationDecision(
                memory_id=memory.memory_id,
                status=MemoryValidityStatus.INVALID,
                confidence=0.0,
                is_applicable=False,
                rationale=f"Required target module(s) missing from repository: {sorted(missing_modules)}",
            )

        # 4. Check if diff is purely cosmetic
        if diff.highest_category == ChangeCategory.COSMETIC:
            return MemoryRevalidationDecision(
                memory_id=memory.memory_id,
                status=MemoryValidityStatus.VALID,
                confidence=memory.confidence,
                is_applicable=True,
                rationale="Repository changes are purely cosmetic; memory remains completely valid",
            )

        # 5. Check if diff touches memory's affected modules or their dependencies
        intersect_direct = mem_modules & set(directly_modified)
        intersect_transitive = mem_modules & transitive_affected

        # Check known boundary conditions
        boundary_violations: List[str] = []
        for boundary in memory.known_boundaries:
            # Check if any changed file satisfies or conflicts with boundary condition
            for p in directly_modified:
                if boundary.lower() in p.lower():
                    boundary_violations.append(f"Boundary constraint triggered on {p}: '{boundary}'")

        if boundary_violations:
            return MemoryRevalidationDecision(
                memory_id=memory.memory_id,
                status=MemoryValidityStatus.INVALID,
                confidence=0.1,
                is_applicable=False,
                rationale=f"Boundary violation detected: {boundary_violations[0]}",
                boundary_violations=boundary_violations,
            )

        # If neither memory modules nor their dependencies are affected: VALID
        if not intersect_direct and not intersect_transitive:
            return MemoryRevalidationDecision(
                memory_id=memory.memory_id,
                status=MemoryValidityStatus.VALID,
                confidence=memory.confidence,
                is_applicable=True,
                rationale="Repository changes do not intersect with memory affected modules or dependency chain",
            )

        # If memory modules were directly modified with architectural or dependency changes: STALE
        if intersect_direct:
            has_dep_or_arch = any(
                diff.file_changes.get(p) and diff.file_changes[p].category in (ChangeCategory.DEPENDENCY, ChangeCategory.ARCHITECTURAL, ChangeCategory.BEHAVIORAL)
                for p in intersect_direct
            )
            if has_dep_or_arch:
                return MemoryRevalidationDecision(
                    memory_id=memory.memory_id,
                    status=MemoryValidityStatus.STALE,
                    confidence=max(0.2, memory.confidence - 0.4),
                    is_applicable=False,
                    rationale=f"Direct structural or dependency modification to memory module(s): {sorted(intersect_direct)}",
                    affected_dependencies=sorted(list(intersect_direct)),
                )

        # If only downstream consumers or tests were modified: CONDITIONALLY_VALID
        if intersect_transitive and not intersect_direct:
            return MemoryRevalidationDecision(
                memory_id=memory.memory_id,
                status=MemoryValidityStatus.CONDITIONALLY_VALID,
                confidence=max(0.5, memory.confidence - 0.1),
                is_applicable=True,
                rationale=f"Transitive dependents changed, but root module intact: {sorted(intersect_transitive)}",
                affected_dependencies=sorted(list(intersect_transitive)),
            )

        # Default fallback: CONDITIONALLY_VALID with cautious confidence
        return MemoryRevalidationDecision(
            memory_id=memory.memory_id,
            status=MemoryValidityStatus.CONDITIONALLY_VALID,
            confidence=max(0.4, memory.confidence - 0.2),
            is_applicable=True,
            rationale="Local modifications detected in target module; revalidation required",
            affected_dependencies=sorted(list(intersect_direct | intersect_transitive)),
        )
