"""
ChakrView Project Arena: Isolated Workspace Management.

Manages creation, layout, file writing, artifact collection, and teardown
of disposable project workspaces.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Optional, List

from chakrview.arena.models import ProjectManifest, SourceFile, FileRole


class IsolatedWorkspace:
    """
    Manages an isolated directory workspace for a project execution session.

    Directory structure:
        workspace_dir/
            specification/
                project_spec.json
            source/
                [source files]
            tests/
                [test files]
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
        self.manifest = project_manifest
        self.preserve = preserve
        self._temp_dir: Optional[tempfile.TemporaryDirectory] = None

        if base_dir:
            self.base_dir = Path(base_dir)
            self.base_dir.mkdir(parents=True, exist_ok=True)
            timestamp = int(time.time() * 1000)
            self.workspace_dir = self.base_dir / f"workspace_{self.manifest.specification.project_id}_{timestamp}"
            self.workspace_dir.mkdir(parents=True, exist_ok=True)
        else:
            self._temp_dir = tempfile.TemporaryDirectory(prefix=f"arena_{self.manifest.specification.project_id}_")
            self.workspace_dir = Path(self._temp_dir.name)

        # Setup standard subdirectories
        self.spec_dir = self.workspace_dir / "specification"
        self.source_dir = self.workspace_dir / "source"
        self.test_dir = self.workspace_dir / "tests"
        self.logs_dir = self.workspace_dir / "logs"
        self.artifacts_dir = self.workspace_dir / "artifacts"

        for d in (self.spec_dir, self.source_dir, self.test_dir, self.logs_dir, self.artifacts_dir):
            d.mkdir(parents=True, exist_ok=True)

        self._setup_files()

    def _setup_files(self) -> None:
        """Write project specification and files to disk."""
        # 1. Write specification
        spec_path = self.spec_dir / "project_spec.json"
        with open(spec_path, "w", encoding="utf-8") as f:
            json.dump(self.manifest.specification.to_dict(), f, indent=2)

        # 2. Write source and test files
        for sfile in self.manifest.files:
            if sfile.role in (FileRole.SOURCE, FileRole.INIT, FileRole.CONFIG):
                dest_path = self.source_dir / sfile.path
            elif sfile.role == FileRole.TEST:
                dest_path = self.test_dir / sfile.path
            else:
                dest_path = self.workspace_dir / sfile.path

            dest_path.parent.mkdir(parents=True, exist_ok=True)
            with open(dest_path, "w", encoding="utf-8") as f:
                f.write(sfile.content)

    def write_source_file(self, rel_path: str, content: str) -> Path:
        """Write or overwrite a source file in the source directory."""
        dest_path = self.source_dir / rel_path
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dest_path, "w", encoding="utf-8") as f:
            f.write(content)
        return dest_path

    def write_test_file(self, rel_path: str, content: str) -> Path:
        """Write or overwrite a test file in the test directory."""
        dest_path = self.test_dir / rel_path
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dest_path, "w", encoding="utf-8") as f:
            f.write(content)
        return dest_path

    def get_source_file_path(self, rel_path: str) -> Path:
        return self.source_dir / rel_path

    def get_test_file_path(self, rel_path: str) -> Path:
        return self.test_dir / rel_path

    def list_source_files(self) -> List[Path]:
        return [p for p in self.source_dir.rglob("*.py") if p.is_file()]

    def list_test_files(self) -> List[Path]:
        return [p for p in self.test_dir.rglob("test_*.py") if p.is_file()]

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
