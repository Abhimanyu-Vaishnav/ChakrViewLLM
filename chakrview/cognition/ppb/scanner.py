"""
ChakrView Step 79: Incremental, Resumable Project Scanner.

Divides large codebases into bounded units/chunks rather than requiring
the entire repository at once:
- Bounded batch size adapted from hardware resource policy
- Resumability: Checks persistent scan progress so already processed units are never redundantly re-scanned
- Interrupted or failed scans resume safely from last persisted unit
- Extracts structured KnowledgeRecords (MODULE, SYMBOL, DEPENDENCY) per file
- Deterministic progress reporting and explicit representation of partial understanding
"""

from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Set, Tuple

from chakrview.runtime.resource import RuntimeStrategy
from chakrview.cognition.repository.inspector import ModuleInspection, RepositoryInspector
from chakrview.cognition.ppb.brain import PersistentProjectBrain
from chakrview.cognition.ppb.models import (
    EpistemicStatus,
    KnowledgeRecord,
    KnowledgeRecordType,
    ProjectBrainState,
)


@dataclass
class ScanBatchProgress:
    """Telemetry and status for an incremental scan session."""
    total_files_discovered: int
    files_already_scanned: int
    files_processed_this_run: int
    files_skipped_unchanged: int
    chunks_completed: int
    is_fully_scanned: bool
    scan_fingerprint: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_files_discovered": self.total_files_discovered,
            "files_already_scanned": self.files_already_scanned,
            "files_processed_this_run": self.files_processed_this_run,
            "files_skipped_unchanged": self.files_skipped_unchanged,
            "chunks_completed": self.chunks_completed,
            "is_fully_scanned": self.is_fully_scanned,
            "scan_fingerprint": self.scan_fingerprint,
        }


class IncrementalProjectScanner:
    """
    Scans repositories in bounded, resource-adaptive, resumable batches.
    """

    def __init__(
        self,
        brain: PersistentProjectBrain,
        chunk_size: Optional[int] = None,
    ) -> None:
        self.brain = brain
        # Determine chunk size adapted to resource policy if not explicitly overridden
        if chunk_size is not None:
            self.chunk_size = max(1, chunk_size)
        else:
            strat = self.brain.resource_policy.strategy
            if strat == RuntimeStrategy.LOW_RESOURCE:
                self.chunk_size = 2
            elif strat == RuntimeStrategy.STANDARD:
                self.chunk_size = 5
            else:
                self.chunk_size = 10

    def discover_files(self, project_root: Optional[str] = None) -> List[str]:
        """
        Discovers all Python source and test files in project root, deterministically sorted.
        Ignores virtual environments, hidden folders, caches, and build artifacts.
        """
        root = Path(project_root or self.brain.project_identity.project_root).resolve()
        discovered: List[str] = []

        ignore_dirs = {
            ".git", ".venv", "venv", "__pycache__", ".pytest_cache",
            "build", "dist", ".gemini", "brain", "artifacts", "runs"
        }

        for current_root, dirs, files in os.walk(root):
            # Prune ignored directories in-place
            dirs[:] = [d for d in sorted(dirs) if d not in ignore_dirs and not d.startswith(".")]
            rel_dir = os.path.relpath(current_root, root)

            for f in sorted(files):
                if f.endswith(".py"):
                    rel_p = os.path.normpath(os.path.join(rel_dir, f) if rel_dir != "." else f).replace("\\", "/")
                    discovered.append(rel_p)

        return sorted(discovered)

    def scan_project(
        self,
        files_dict: Optional[Dict[str, str]] = None,
        max_chunks: Optional[int] = None,
        on_chunk_completed: Optional[Callable[[int, List[str]], None]] = None,
    ) -> ScanBatchProgress:
        """
        Incrementally scan project files in bounded batches.
        Can consume files from files_dict (memory/virtual filesystem) or live project_root.
        Resumes automatically using persistent scan progress.
        """
        project_id = self.brain.project_id
        already_scanned = self.brain.storage.get_scanned_files(project_id)

        # 1. Determine files to inspect
        if files_dict is not None:
            all_files = sorted(list(files_dict.keys()))
            file_getter = lambda p: files_dict[p]
        else:
            all_files = self.discover_files()
            root = Path(self.brain.project_identity.project_root)
            file_getter = lambda p: (root / p).read_text(encoding="utf-8", errors="replace")

        total_files = len(all_files)

        # 2. Filter out already scanned files whose content has not changed
        pending_files: List[Tuple[str, str, str]] = []  # (rel_path, content, sha256)
        skipped_count = 0

        for path in all_files:
            try:
                content = file_getter(path)
                sha = hashlib.sha256(content.encode("utf-8")).hexdigest()
                if path in already_scanned and already_scanned[path] == sha:
                    skipped_count += 1
                else:
                    pending_files.append((path, content, sha))
            except Exception:
                continue

        # 3. Process in bounded chunks
        chunks_done = 0
        processed_this_run = 0

        for i in range(0, len(pending_files), self.chunk_size):
            if max_chunks is not None and chunks_done >= max_chunks:
                break

            chunk_id = f"chunk_{chunks_done + 1}"
            chunk = pending_files[i : i + self.chunk_size]
            chunk_file_paths: List[str] = []

            for rel_path, content, sha in chunk:
                self._process_single_file(
                    rel_path=rel_path,
                    content=content,
                    sha=sha,
                    chunk_id=chunk_id,
                )
                self.brain.storage.record_scan_progress(
                    project_id=project_id,
                    file_path=rel_path,
                    content_sha256=sha,
                    source_chunk=chunk_id,
                    status="COMPLETED",
                )
                chunk_file_paths.append(rel_path)
                processed_this_run += 1

            chunks_done += 1
            if on_chunk_completed:
                on_chunk_completed(chunks_done, chunk_file_paths)

        # Check completeness
        updated_scanned = self.brain.storage.get_scanned_files(project_id)
        is_fully_scanned = len(updated_scanned) >= total_files and len(pending_files) == processed_this_run

        # Compute deterministic overall scan fingerprint
        hasher = hashlib.sha256()
        for p in sorted(updated_scanned.keys()):
            hasher.update(p.encode("utf-8"))
            hasher.update(updated_scanned[p].encode("utf-8"))
        final_fp = hasher.hexdigest()

        self.brain.storage.update_last_scan_fingerprint(project_id, final_fp)

        return ScanBatchProgress(
            total_files_discovered=total_files,
            files_already_scanned=len(already_scanned),
            files_processed_this_run=processed_this_run,
            files_skipped_unchanged=skipped_count,
            chunks_completed=chunks_done,
            is_fully_scanned=is_fully_scanned,
            scan_fingerprint=final_fp,
        )

    def _process_single_file(
        self,
        rel_path: str,
        content: str,
        sha: str,
        chunk_id: str,
    ) -> None:
        """Inspect file with AST and persist MODULE, SYMBOL, and DEPENDENCY knowledge records."""
        insp: ModuleInspection = RepositoryInspector.inspect_source(rel_path, content)
        project_id = self.brain.project_id

        # 1. Module level record
        mod_record_id = f"mod_{project_id}_{rel_path.replace('/', '_').replace('.', '_')}"
        mod_record = KnowledgeRecord(
            record_id=mod_record_id,
            project_id=project_id,
            record_type=KnowledgeRecordType.MODULE,
            file_path=rel_path,
            summary=f"Module {insp.module_name} in {rel_path}",
            details={
                "functions": insp.functions,
                "classes": insp.classes,
                "imports": insp.imports,
                "is_test": insp.is_test,
                "lines_count": len(content.splitlines()),
            },
            dependencies=insp.imports,
            related_symbols=insp.functions + insp.classes,
            evidence_ids=[f"ast_{sha[:12]}"],
            epistemic_status=EpistemicStatus.FACT,
            confidence=1.0,
            repo_fingerprint=sha[:16],
            source_chunk=chunk_id,
        )
        self.brain.store_knowledge(mod_record)

        # 2. Symbol level records for classes
        for cl in insp.classes:
            sym_id = f"sym_{project_id}_{rel_path.replace('/', '_')}_{cl}"
            sym_record = KnowledgeRecord(
                record_id=sym_id,
                project_id=project_id,
                record_type=KnowledgeRecordType.SYMBOL,
                file_path=rel_path,
                symbol_name=cl,
                summary=f"Class {cl} in {rel_path}",
                details={"kind": "class", "module": insp.module_name},
                dependencies=insp.imports,
                related_symbols=[f for f in insp.functions],
                evidence_ids=[f"ast_{sha[:12]}"],
                epistemic_status=EpistemicStatus.FACT,
                confidence=1.0,
                repo_fingerprint=sha[:16],
                source_chunk=chunk_id,
            )
            self.brain.store_knowledge(sym_record)

        # 3. Symbol level records for functions
        for fn in insp.functions:
            sym_id = f"sym_{project_id}_{rel_path.replace('/', '_')}_{fn}"
            sym_record = KnowledgeRecord(
                record_id=sym_id,
                project_id=project_id,
                record_type=KnowledgeRecordType.SYMBOL,
                file_path=rel_path,
                symbol_name=fn,
                summary=f"Function {fn} in {rel_path}",
                details={"kind": "function", "module": insp.module_name},
                dependencies=insp.imports,
                related_symbols=[],
                evidence_ids=[f"ast_{sha[:12]}"],
                epistemic_status=EpistemicStatus.FACT,
                confidence=1.0,
                repo_fingerprint=sha[:16],
                source_chunk=chunk_id,
            )
            self.brain.store_knowledge(sym_record)

        # 4. Dependency records
        if insp.imports:
            dep_id = f"dep_{project_id}_{rel_path.replace('/', '_')}"
            dep_record = KnowledgeRecord(
                record_id=dep_id,
                project_id=project_id,
                record_type=KnowledgeRecordType.DEPENDENCY,
                file_path=rel_path,
                summary=f"Imports for {rel_path}",
                details={"imports": insp.imports, "imported_symbols": insp.imported_symbols},
                dependencies=insp.imports,
                evidence_ids=[f"ast_{sha[:12]}"],
                epistemic_status=EpistemicStatus.FACT,
                confidence=1.0,
                repo_fingerprint=sha[:16],
                source_chunk=chunk_id,
            )
            self.brain.store_knowledge(dep_record)
