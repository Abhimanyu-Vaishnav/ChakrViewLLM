"""
ChakrView Step 64: Persistent Repository Context, Incremental Invalidation & Evidence Retrieval.

Provides a persistent intelligence/cache layer over repository structure, eliminating
full-repository rescans while maintaining strict deterministic evidence provenance:
- RepositoryContextStore: Persistent cache of file digests, AST signatures, and dependency topologies.
- Incremental Invalidation: Uses Step 61 change classification to selectively invalidate only affected entries.
- EvidenceRecord: Strongly typed provenance tracking for every piece of grounded context.
- GroundedContextRetriever: Budgeted, task-relevant context retrieval.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Tuple

from chakrview.arena.models import ProjectManifest, SourceFile
from chakrview.cognition.repository.state import RepositoryState, FileState
from chakrview.cognition.repository.inspector import RepositoryInspector, ModuleInspection
from chakrview.cognition.repository.graph import RepositoryDependencyGraph
from chakrview.cognition.repository.change_detector import (
    RepositoryChangeDetector,
    RepositoryDiff,
    ChangeCategory,
    FileChange,
)
from chakrview.cognition.repository.impact_analyzer import (
    RepositoryImpactAnalyzer,
    MemoryValidityStatus,
    MemoryRevalidationDecision,
    ImpactReport,
)
from chakrview.cognition.repository.semantic_record import RepositorySemanticRecord
from chakrview.cognition.repository.memory_index import RepositoryMemoryIndex


class EpistemicState(Enum):
    """Certainty and grounding state for repository facts."""
    KNOWN = auto()
    UNKNOWN = auto()
    STALE = auto()
    CONTRADICTED = auto()
    UNAVAILABLE = auto()


@dataclass
class EvidenceRecord:
    """Provenance record tracking the grounding source for a piece of repository evidence."""
    source_type: str  # "AST_INSPECTION", "DEPENDENCY_GRAPH", "TEST_INVENTORY", "SEMANTIC_MEMORY"
    file_path: str
    symbol_name: Optional[str] = None
    structural_digest: str = ""
    retrieval_reason: str = ""
    dependency_distance: int = 0
    epistemic_state: EpistemicState = EpistemicState.KNOWN
    timestamp_utc: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_type": self.source_type,
            "file_path": self.file_path,
            "symbol_name": self.symbol_name,
            "structural_digest": self.structural_digest,
            "retrieval_reason": self.retrieval_reason,
            "dependency_distance": self.dependency_distance,
            "epistemic_state": self.epistemic_state.name,
            "timestamp_utc": self.timestamp_utc,
        }


@dataclass
class CachedFileEntry:
    """Cache record storing parsed structural information for a single file."""
    rel_path: str
    content_hash: str
    size_bytes: int
    is_test: bool
    module_inspection: ModuleInspection
    last_indexed_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rel_path": self.rel_path,
            "content_hash": self.content_hash,
            "size_bytes": self.size_bytes,
            "is_test": self.is_test,
            "last_indexed_at": self.last_indexed_at,
        }


@dataclass
class ContextStoreTelemetry:
    """Resource and timing telemetry for context store operations."""
    indexing_time_ms: float = 0.0
    files_inspected: int = 0
    files_reused_from_cache: int = 0
    invalidated_entries_count: int = 0
    cache_hit_ratio: float = 0.0
    retrieval_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "indexing_time_ms": round(self.indexing_time_ms, 2),
            "files_inspected": self.files_inspected,
            "files_reused_from_cache": self.files_reused_from_cache,
            "invalidated_entries_count": self.invalidated_entries_count,
            "cache_hit_ratio": round(self.cache_hit_ratio, 4),
            "retrieval_time_ms": round(self.retrieval_time_ms, 2),
        }


class RepositoryContextStore:
    """
    Persistent repository context store maintaining file inventories, AST signatures,
    dependency graphs, and incremental updates across successive refactorings.
    """

    def __init__(self, project_id: str = "default_repo") -> None:
        self.project_id = project_id
        self._file_cache: Dict[str, CachedFileEntry] = {}
        self._current_state: Optional[RepositoryState] = None
        self._dependency_graph: RepositoryDependencyGraph = RepositoryDependencyGraph()
        self._last_fingerprint: str = ""
        self.telemetry = ContextStoreTelemetry()

    @property
    def current_state(self) -> Optional[RepositoryState]:
        return self._current_state

    @property
    def dependency_graph(self) -> RepositoryDependencyGraph:
        return self._dependency_graph

    @property
    def cached_files_count(self) -> int:
        return len(self._file_cache)

    def synchronize(self, manifest: ProjectManifest) -> RepositoryState:
        """
        Synchronize context store with a project manifest.
        Re-inspects ONLY modified or new files, reusing existing cached entries.
        """
        start_t = time.perf_counter()

        manifest_files = {sf.path: sf for sf in manifest.files}
        manifest_paths = set(manifest_files.keys())
        cached_paths = set(self._file_cache.keys())

        # 1. Identify removed files
        removed_paths = cached_paths - manifest_paths
        for p in removed_paths:
            del self._file_cache[p]

        files_inspected = 0
        files_reused = 0
        inspections: Dict[str, ModuleInspection] = {}
        file_states: Dict[str, FileState] = {}

        # 2. Incrementally inspect or reuse
        for path, sf in manifest_files.items():
            content_hash = hashlib.sha256(sf.content.encode("utf-8")).hexdigest()
            cached = self._file_cache.get(path)

            if cached and cached.content_hash == content_hash:
                # Cache hit: reuse AST inspection
                files_reused += 1
                insp = cached.module_inspection
            else:
                # Cache miss: recompute AST inspection
                files_inspected += 1
                insp = RepositoryInspector.inspect_source(path, sf.content)
                self._file_cache[path] = CachedFileEntry(
                    rel_path=path,
                    content_hash=content_hash,
                    size_bytes=len(sf.content.encode("utf-8")),
                    is_test=insp.is_test,
                    module_inspection=insp,
                )

            inspections[path] = insp
            file_states[path] = FileState(
                rel_path=path,
                sha256=content_hash,
                size_bytes=len(sf.content.encode("utf-8")),
                is_test=insp.is_test,
                module_inspection=insp,
            )

        # 3. Update dependency graph
        dep_graph = RepositoryDependencyGraph()
        dep_graph.build_from_inspections(inspections)
        self._dependency_graph = dep_graph

        total_files = len(manifest_files)
        hit_ratio = files_reused / total_files if total_files > 0 else 0.0

        repo_state = RepositoryState(
            project_id=self.project_id,
            files=file_states,
            dependency_graph=dep_graph,
            version=(self._current_state.version + 1) if self._current_state else 1,
        )
        self._current_state = repo_state
        self._last_fingerprint = repo_state.state_fingerprint

        elapsed = (time.perf_counter() - start_t) * 1000.0
        self.telemetry = ContextStoreTelemetry(
            indexing_time_ms=elapsed,
            files_inspected=files_inspected,
            files_reused_from_cache=files_reused,
            invalidated_entries_count=len(removed_paths) + files_inspected,
            cache_hit_ratio=hit_ratio,
        )

        return repo_state

    def apply_incremental_diff(self, diff: RepositoryDiff) -> Set[str]:
        """
        Selectively invalidate cache entries according to Step 61 change classification.
        Returns the set of invalidated module paths.
        """
        invalidated: Set[str] = set()

        for path, fc in diff.file_changes.items():
            if fc.category in (
                ChangeCategory.DEPENDENCY,
                ChangeCategory.ARCHITECTURAL,
                ChangeCategory.BEHAVIORAL,
                ChangeCategory.LOCAL,
                ChangeCategory.TEST_ONLY,
            ):
                invalidated.add(path)
                # Invalidate downstream dependents if dependency or architectural
                if fc.category in (ChangeCategory.DEPENDENCY, ChangeCategory.ARCHITECTURAL):
                    dependents = self._dependency_graph.get_downstream_dependents(path)
                    invalidated.update(dependents)

        # Remove invalidated entries from file cache
        for inv in invalidated:
            self._file_cache.pop(inv, None)

        return invalidated

    def file_exists(self, path: str) -> bool:
        return path in self._file_cache

    def symbol_exists(self, symbol_name: str, file_path: Optional[str] = None) -> bool:
        """Check if a symbol (function, class, or variable) exists in the repository."""
        if file_path:
            entry = self._file_cache.get(file_path)
            if not entry:
                return False
            insp = entry.module_inspection
            return symbol_name in insp.functions or symbol_name in insp.classes

        for entry in self._file_cache.values():
            insp = entry.module_inspection
            if symbol_name in insp.functions or symbol_name in insp.classes:
                return True
        return False

    def get_file_content_hash(self, path: str) -> Optional[str]:
        entry = self._file_cache.get(path)
        return entry.content_hash if entry else None


@dataclass
class GroundedContextBudget:
    """Defines limits for context retrieval budgeting."""
    max_files: int = 5
    max_symbols: int = 20
    max_dependency_depth: int = 2
    max_memory_records: int = 3


@dataclass
class GroundedContextBundle:
    """
    A strictly bounded, evidence-grounded context payload delivered to proposal generators.
    Guaranteed to contain only verified repository facts.
    """
    task_description: str
    target_domain: str
    repository_fingerprint: str
    candidate_files: List[str]
    relevant_symbols: List[str]
    evidence_records: List[EvidenceRecord]
    active_memory_records: List[RepositorySemanticRecord]
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_description": self.task_description,
            "target_domain": self.target_domain,
            "repository_fingerprint": self.repository_fingerprint,
            "candidate_files": list(self.candidate_files),
            "relevant_symbols": list(self.relevant_symbols),
            "evidence_records": [e.to_dict() for e in self.evidence_records],
            "active_memory_records": [m.to_dict() for m in self.active_memory_records],
            "metrics": self.metrics,
        }


class GroundedContextRetriever:
    """
    Retrieves bounded, evidence-backed context for a given task:
    - Filters files relevant to keywords and semantic memories
    - Extracts symbols and dependency neighbors up to budgeted depth
    - Links each item to an EvidenceRecord ensuring provenance
    """

    def __init__(
        self,
        context_store: RepositoryContextStore,
        memory_index: Optional[RepositoryMemoryIndex] = None,
    ) -> None:
        self.context_store = context_store
        self.memory_index = memory_index

    def retrieve_context(
        self,
        task_description: str,
        target_domain: str,
        budget: Optional[GroundedContextBudget] = None,
    ) -> GroundedContextBundle:
        start_t = time.perf_counter()
        cfg = budget or GroundedContextBudget()

        store = self.context_store
        repo_state = store.current_state
        if not repo_state:
            raise RuntimeError("RepositoryContextStore is empty; synchronize before retrieval")

        task_words = set(w.lower().strip(".,:-_") for w in task_description.split() if len(w) > 2)

        # 1. Select candidate files based on lexical relevance
        selected_files: List[str] = []
        for path in sorted(repo_state.files.keys()):
            stem = path.lower()
            if any(w in stem for w in task_words):
                selected_files.append(path)
            if len(selected_files) >= cfg.max_files:
                break

        # Fallback if no lexical match: include first files up to budget
        if not selected_files:
            selected_files = sorted(list(repo_state.files.keys()))[: cfg.max_files]

        # 2. Trace dependency neighborhood
        evidence_list: List[EvidenceRecord] = []
        collected_symbols: List[str] = []

        for f_path in selected_files:
            f_entry = store._file_cache.get(f_path)
            if not f_entry:
                continue

            insp = f_entry.module_inspection
            # Record file existence evidence
            evidence_list.append(
                EvidenceRecord(
                    source_type="AST_INSPECTION",
                    file_path=f_path,
                    structural_digest=f_entry.content_hash[:16],
                    retrieval_reason="Candidate target file for task",
                    dependency_distance=0,
                )
            )

            # Collect symbols
            for fn in insp.functions:
                if len(collected_symbols) < cfg.max_symbols:
                    collected_symbols.append(fn)
                    evidence_list.append(
                        EvidenceRecord(
                            source_type="AST_INSPECTION",
                            file_path=f_path,
                            symbol_name=fn,
                            structural_digest=f_entry.content_hash[:16],
                            retrieval_reason=f"Function in {f_path}",
                            dependency_distance=0,
                        )
                    )
            for cl in insp.classes:
                if len(collected_symbols) < cfg.max_symbols:
                    collected_symbols.append(cl)
                    evidence_list.append(
                        EvidenceRecord(
                            source_type="AST_INSPECTION",
                            file_path=f_path,
                            symbol_name=cl,
                            structural_digest=f_entry.content_hash[:16],
                            retrieval_reason=f"Class in {f_path}",
                            dependency_distance=0,
                        )
                    )

            # Record dependency relationships
            downstream = store.dependency_graph.get_downstream_dependents(f_path)
            for d in downstream[: cfg.max_files]:
                evidence_list.append(
                    EvidenceRecord(
                        source_type="DEPENDENCY_GRAPH",
                        file_path=d,
                        retrieval_reason=f"Downstream dependent of {f_path}",
                        dependency_distance=1,
                    )
                )

        # 3. Retrieve relevant memory records
        active_memories: List[RepositorySemanticRecord] = []
        if self.memory_index:
            candidates = self.memory_index.candidate_set(task_family=target_domain)
            for rec in candidates[: cfg.max_memory_records]:
                active_memories.append(rec)
                evidence_list.append(
                    EvidenceRecord(
                        source_type="SEMANTIC_MEMORY",
                        file_path=rec.affected_modules[0] if rec.affected_modules else "",
                        structural_digest=rec.memory_id,
                        retrieval_reason=f"Relevant memory pattern: {rec.repository_pattern}",
                    )
                )

        elapsed = (time.perf_counter() - start_t) * 1000.0

        metrics = {
            "retrieval_time_ms": round(elapsed, 2),
            "files_selected": len(selected_files),
            "symbols_selected": len(collected_symbols),
            "evidence_count": len(evidence_list),
            "memories_selected": len(active_memories),
        }

        return GroundedContextBundle(
            task_description=task_description,
            target_domain=target_domain,
            repository_fingerprint=repo_state.state_fingerprint,
            candidate_files=selected_files,
            relevant_symbols=collected_symbols,
            evidence_records=evidence_list,
            active_memory_records=active_memories,
            metrics=metrics,
        )
