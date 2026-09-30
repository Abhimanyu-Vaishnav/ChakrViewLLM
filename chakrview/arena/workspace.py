"""
ChakrView Project Arena: Isolated Workspace Management.

Manages creation, layout, file writing, artifact collection, path traversal protection,
quota ceilings, diff tracking, and teardown of disposable project workspaces.
"""

from __future__ import annotations

import copy
import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Optional, List, Dict, Any

from chakrview.arena.models import (
    ProjectManifest,
    SourceFile,
    FileRole,
    PathTraversalError,
    WorkspaceQuotaExceededError,
    PatchDiff,
)

MAX_WORKSPACE_FILES = 50
MAX_WORKSPACE_BYTES = 10 * 1024 * 1024  # 10 MB


class IsolatedWorkspace:
    """
    Manages an isolated directory workspace for a project execution session.

    Directory structure:
        workspace_dir/
            project_manifest.json
            specification/
                project_spec.json
            source/
                [source files]
            tests/
                [test files]
            runtime/
                [runtime config and entrypoints]
            logs/
                [execution stdout/stderr logs]
            artifacts/
                [evaluation outputs, diffs, AST reports]
    """
    def __init__(
        self,
        project_manifest: ProjectManifest,
        base_dir: Optional[Path | str] = None,
        preserve: bool = False,
    ) -> None:
        self._initial_manifest = copy.deepcopy(project_manifest)
        self.manifest = copy.deepcopy(project_manifest)
        self.preserve = preserve
        self._temp_dir: Optional[tempfile.TemporaryDirectory] = None
        self.patch_history: List[PatchDiff] = []

        if base_dir:
            self.base_dir = Path(base_dir).resolve()
            self.base_dir.mkdir(parents=True, exist_ok=True)
            timestamp = int(time.time() * 1000)
            self.workspace_dir = (self.base_dir / f"workspace_{self.manifest.specification.project_id}_{timestamp}").resolve()
            self.workspace_dir.mkdir(parents=True, exist_ok=True)
        else:
            self._temp_dir = tempfile.TemporaryDirectory(prefix=f"arena_{self.manifest.specification.project_id}_")
            self.workspace_dir = Path(self._temp_dir.name).resolve()

        # Setup standard subdirectories
        self.spec_dir = (self.workspace_dir / "specification").resolve()
        self.source_dir = (self.workspace_dir / "source").resolve()
        self.test_dir = (self.workspace_dir / "tests").resolve()
        self.runtime_dir = (self.workspace_dir / "runtime").resolve()
        self.logs_dir = (self.workspace_dir / "logs").resolve()
        self.artifacts_dir = (self.workspace_dir / "artifacts").resolve()
        self.diffs_dir = (self.artifacts_dir / "diffs").resolve()

        for d in (self.spec_dir, self.source_dir, self.test_dir, self.runtime_dir, self.logs_dir, self.artifacts_dir, self.diffs_dir):
            d.mkdir(parents=True, exist_ok=True)

        self._setup_files()

    def _assert_within_workspace(self, target_path: Path) -> Path:
        """
        Verify that target_path resolves strictly inside workspace_dir.
        Raises PathTraversalError if target_path escapes the workspace boundary.
        """
        resolved = target_path.resolve()
        workspace_resolved = self.workspace_dir.resolve()
        try:
            resolved.relative_to(workspace_resolved)
        except ValueError:
            raise PathTraversalError(
                f"Path traversal detected: '{target_path}' resolves to '{resolved}' which is outside workspace '{workspace_resolved}'"
            )
        return resolved

    def _check_quotas(self, additional_bytes: int = 0) -> None:
        """Ensure file count and size quotas are strictly respected."""
        total_files = len(list(self.workspace_dir.rglob("*")))
        if total_files > MAX_WORKSPACE_FILES:
            raise WorkspaceQuotaExceededError(
                f"Workspace file count ({total_files}) exceeds quota ({MAX_WORKSPACE_FILES})"
            )
        
        total_bytes = sum(f.stat().st_size for f in self.workspace_dir.rglob("*") if f.is_file())
        if (total_bytes + additional_bytes) > MAX_WORKSPACE_BYTES:
            raise WorkspaceQuotaExceededError(
                f"Workspace disk usage ({total_bytes + additional_bytes} bytes) exceeds quota ({MAX_WORKSPACE_BYTES} bytes)"
            )

    def _setup_files(self) -> None:
        """Write project manifest, specification, and initial files to disk."""
        # 1. Write top-level project_manifest.json
        manifest_path = self._assert_within_workspace(self.workspace_dir / "project_manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(self.manifest.to_dict(), f, indent=2)

        # 2. Write specification
        spec_path = self._assert_within_workspace(self.spec_dir / "project_spec.json")
        with open(spec_path, "w", encoding="utf-8") as f:
            json.dump(self.manifest.specification.to_dict(), f, indent=2)

        # 3. Write source, test, and config files
        for sfile in self.manifest.files:
            if sfile.role in (FileRole.SOURCE, FileRole.INIT):
                dest_path = self.source_dir / sfile.path
            elif sfile.role == FileRole.TEST:
                dest_path = self.test_dir / sfile.path
            elif sfile.role == FileRole.CONFIG:
                dest_path = self.runtime_dir / sfile.path
            else:
                dest_path = self.workspace_dir / sfile.path

            safe_dest = self._assert_within_workspace(dest_path)
            safe_dest.parent.mkdir(parents=True, exist_ok=True)
            with open(safe_dest, "w", encoding="utf-8") as f:
                f.write(sfile.content)

        self._check_quotas()

    def write_source_file(self, rel_path: str, content: str) -> Path:
        """Write or overwrite a source file in the source directory with path confinement."""
        dest_path = self._assert_within_workspace(self.source_dir / rel_path)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        content_bytes = len(content.encode("utf-8"))
        self._check_quotas(additional_bytes=content_bytes)

        # Update manifest in memory
        self.manifest.upsert_file(SourceFile(path=rel_path, content=content, role=FileRole.SOURCE))

        with open(dest_path, "w", encoding="utf-8") as f:
            f.write(content)
        return dest_path

    def write_test_file(self, rel_path: str, content: str) -> Path:
        """Write or overwrite a test file in the test directory with path confinement."""
        dest_path = self._assert_within_workspace(self.test_dir / rel_path)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        content_bytes = len(content.encode("utf-8"))
        self._check_quotas(additional_bytes=content_bytes)

        # Update manifest in memory
        self.manifest.upsert_file(SourceFile(path=rel_path, content=content, role=FileRole.TEST))

        with open(dest_path, "w", encoding="utf-8") as f:
            f.write(content)
        return dest_path

    def apply_patch(self, rel_path: str, new_content: str) -> PatchDiff:
        """
        Modify an existing source file, record the unified diff, and store diff artifact.
        """
        file_path = self._assert_within_workspace(self.source_dir / rel_path)
        old_content = file_path.read_text(encoding="utf-8") if file_path.is_file() else ""

        self.write_source_file(rel_path, new_content)

        diff = PatchDiff(path=rel_path, old_content=old_content, new_content=new_content)
        self.patch_history.append(diff)

        # Save diff artifact to artifacts/diffs/
        diff_filename = f"patch_{len(self.patch_history):03d}_{rel_path.replace('/', '_')}.diff"
        diff_path = self.diffs_dir / diff_filename
        with open(diff_path, "w", encoding="utf-8") as f:
            f.write(diff.diff_text)

        return diff

    def get_source_file_path(self, rel_path: str) -> Path:
        return self._assert_within_workspace(self.source_dir / rel_path)

    def get_test_file_path(self, rel_path: str) -> Path:
        return self._assert_within_workspace(self.test_dir / rel_path)

    def list_source_files(self) -> List[Path]:
        return [p for p in self.source_dir.rglob("*.py") if p.is_file()]

    def list_test_files(self) -> List[Path]:
        return [p for p in self.test_dir.rglob("test_*.py") if p.is_file()]

    def reset_to_initial(self) -> None:
        """Wipe source, tests, logs, diffs directories and re-materialize from original manifest."""
        self.manifest = copy.deepcopy(self._initial_manifest)
        for d in (self.source_dir, self.test_dir, self.logs_dir, self.diffs_dir):
            if d.exists():
                shutil.rmtree(d, ignore_errors=True)
                d.mkdir(parents=True, exist_ok=True)
        self._setup_files()
        self.patch_history.clear()

    def cleanup(self) -> None:
        """Clean up workspace if not marked for preservation."""
        if self.preserve:
            return
        if self._temp_dir:
            try:
                self._temp_dir.cleanup()
            except Exception:
                pass
        elif self.workspace_dir.exists():
            try:
                shutil.rmtree(self.workspace_dir, ignore_errors=True)
            except Exception:
                pass

    def __enter__(self) -> IsolatedWorkspace:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.cleanup()
