"""
ChakrView Step 61: Live Repository State, Deterministic Fingerprinting & Impact Models.

Captures:
- File inventory and SHA-256 content hashes
- Module-level AST signatures (imports, functions, classes, calls)
- Bidirectional dependency topology
- Test inventory and test-target associations
- Canonical cryptographic state fingerprint
Deterministic, CPU-first, serializable, and independent of neural weights.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from chakrview.arena.models import ProjectManifest, SourceFile
from chakrview.cognition.repository.inspector import RepositoryInspector, ModuleInspection
from chakrview.cognition.repository.graph import RepositoryDependencyGraph


@dataclass
class FileState:
    """State record for a single repository file."""
    rel_path: str
    sha256: str
    size_bytes: int
    is_test: bool
    module_inspection: Optional[ModuleInspection] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d


@dataclass
class RepositoryState:
    """
    Complete, deterministic snapshot of repository structure, files,
    AST semantics, and dependency graph.
    """
    project_id: str
    files: Dict[str, FileState] = field(default_factory=dict)
    dependency_graph: Optional[RepositoryDependencyGraph] = None
    language: str = "python"
    framework: str = "standard_library"
    version: int = 1
    state_fingerprint: str = ""

    def __post_init__(self) -> None:
        if not self.state_fingerprint and self.files:
            self.state_fingerprint = self.compute_fingerprint()

    def compute_fingerprint(self) -> str:
        """
        Compute deterministic SHA-256 fingerprint over all sorted file paths,
        hashes, and structural signatures.
        """
        hasher = hashlib.sha256()
        hasher.update(self.project_id.encode("utf-8"))
        hasher.update(self.language.encode("utf-8"))
        hasher.update(self.framework.encode("utf-8"))

        for path in sorted(self.files.keys()):
            f_state = self.files[path]
            hasher.update(path.encode("utf-8"))
            hasher.update(f_state.sha256.encode("utf-8"))
            if f_state.module_inspection:
                insp = f_state.module_inspection
                hasher.update(",".join(insp.imports).encode("utf-8"))
                hasher.update(",".join(insp.functions).encode("utf-8"))
                hasher.update(",".join(insp.classes).encode("utf-8"))

        if self.dependency_graph:
            for edge in sorted(self.dependency_graph.edges, key=lambda e: (e.source_module, e.target_module, e.edge_type)):
                hasher.update(f"{edge.source_module}->{edge.target_module}:{edge.edge_type}".encode("utf-8"))

        return hasher.hexdigest()

    @classmethod
    def from_manifest(
        cls,
        manifest: ProjectManifest,
        version: int = 1,
        language: str = "python",
        framework: str = "standard_library",
    ) -> RepositoryState:
        """Construct RepositoryState directly from a ProjectManifest."""
        file_states: Dict[str, FileState] = {}
        inspections: Dict[str, ModuleInspection] = {}

        for sf in manifest.files:
            content_bytes = sf.content.encode("utf-8")
            h = hashlib.sha256(content_bytes).hexdigest()
            insp = RepositoryInspector.inspect_source(sf.path, sf.content)
            inspections[sf.path] = insp

            file_states[sf.path] = FileState(
                rel_path=sf.path,
                sha256=h,
                size_bytes=len(content_bytes),
                is_test=insp.is_test,
                module_inspection=insp,
            )

        dep_graph = RepositoryDependencyGraph()
        dep_graph.build_from_inspections(inspections)

        state = cls(
            project_id=manifest.specification.project_id,
            files=file_states,
            dependency_graph=dep_graph,
            language=language,
            framework=framework,
            version=version,
        )
        state.state_fingerprint = state.compute_fingerprint()
        return state

    @classmethod
    def from_files(
        cls,
        project_id: str,
        files_dict: Dict[str, str],
        version: int = 1,
        language: str = "python",
        framework: str = "standard_library",
    ) -> RepositoryState:
        """Construct RepositoryState from path -> content string mapping."""
        file_states: Dict[str, FileState] = {}
        inspections: Dict[str, ModuleInspection] = {}

        for path, content in files_dict.items():
            content_bytes = content.encode("utf-8")
            h = hashlib.sha256(content_bytes).hexdigest()
            insp = RepositoryInspector.inspect_source(path, content)
            inspections[path] = insp

            file_states[path] = FileState(
                rel_path=path,
                sha256=h,
                size_bytes=len(content_bytes),
                is_test=insp.is_test,
                module_inspection=insp,
            )

        dep_graph = RepositoryDependencyGraph()
        dep_graph.build_from_inspections(inspections)

        state = cls(
            project_id=project_id,
            files=file_states,
            dependency_graph=dep_graph,
            language=language,
            framework=framework,
            version=version,
        )
        state.state_fingerprint = state.compute_fingerprint()
        return state

    def to_dict(self) -> Dict[str, Any]:
        """Deterministic serialization."""
        return {
            "project_id": self.project_id,
            "version": self.version,
            "language": self.language,
            "framework": self.framework,
            "state_fingerprint": self.state_fingerprint,
            "files": {p: fs.to_dict() for p, fs in sorted(self.files.items())},
            "dependency_graph": self.dependency_graph.to_dict() if self.dependency_graph else None,
        }
